# Module 05: Time-Slicing for Shared GPU Access

| | |
|---|---|
| **Time** | 3-5 hours |
| **Difficulty** | Intermediate |
| **Prerequisites** | Module 04 completed |

---

## Learning Objectives

By the end of this module, you will be able to:

- Explain how GPU time-slicing works at the CUDA context level
- Configure time-slicing through the NVIDIA device plugin ConfigMap
- Deploy multiple pods sharing a single GPU via time-slicing
- Understand the performance and isolation trade-offs
- Set up different time-slicing profiles for different node pools
- Monitor time-sliced GPU utilization and memory pressure

---

## Concepts

### What is GPU Time-Slicing?

GPU time-slicing allows multiple containers to share a single physical GPU by **context-switching** CUDA contexts. It is analogous to how a CPU time-slices processes, but at the GPU level.

```
Physical GPU (1x A100)
┌───────────────────────────────────────────────┐
│                                               │
│  Time ──►                                     │
│                                               │
│  ┌─────┐┌─────┐┌─────┐┌─────┐┌─────┐        │
│  │Pod A││Pod B││Pod C││Pod A││Pod B│  ...    │
│  │ ctx ││ ctx ││ ctx ││ ctx ││ ctx │         │
│  └─────┘└─────┘└─────┘└─────┘└─────┘        │
│                                               │
│  All pods see the full GPU memory             │
│  Context switch overhead: ~5-10%              │
│  NO memory isolation between pods             │
└───────────────────────────────────────────────┘
```

**How it works under the hood:**
1. The NVIDIA device plugin is configured with `replicas: N` per GPU.
2. It advertises `N x nvidia.com/gpu` resources per physical GPU to Kubernetes.
3. When multiple pods are scheduled, each gets a CUDA context on the same GPU.
4. The GPU driver time-slices between contexts using a round-robin scheduler.
5. Each pod sees the **full GPU memory** but shares compute time.

### Time-Slicing Configuration

The ConfigMap controls how many replicas (virtual GPUs) each physical GPU exposes:

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: time-slicing-config
  namespace: gpu-operator
data:
  any: |
    version: v1
    sharing:
      timeSlicing:
        renameByDefault: false          # Keep nvidia.com/gpu name
        failRequestsGreaterThanOne: false
        resources:
          - name: nvidia.com/gpu
            replicas: 4                 # 1 physical GPU = 4 virtual GPUs
```

**Key parameters:**

| Parameter | Default | Description |
|---|---|---|
| `replicas` | 1 | Number of virtual GPUs per physical GPU |
| `renameByDefault` | false | If true, advertise as `nvidia.com/gpu.shared` instead of `nvidia.com/gpu` |
| `failRequestsGreaterThanOne` | false | If true, reject pods requesting > 1 GPU (prevents exclusive allocation) |

### Profile-Based Configuration

Different node pools can have different time-slicing settings. Use node labels to select profiles:

```yaml
data:
  # Development nodes: 8 slices per GPU (many small workloads)
  dev: |
    version: v1
    sharing:
      timeSlicing:
        renameByDefault: true
        resources:
          - name: nvidia.com/gpu
            replicas: 8

  # Inference nodes: 4 slices per GPU
  inference: |
    version: v1
    sharing:
      timeSlicing:
        resources:
          - name: nvidia.com/gpu
            replicas: 4

  # Training nodes: no sharing (1:1)
  training: |
    version: v1
    sharing:
      timeSlicing:
        resources:
          - name: nvidia.com/gpu
            replicas: 1
```

```bash
# Assign profiles via node labels
kubectl label node dev-node-1 nvidia.com/device-plugin.config=dev
kubectl label node inf-node-1 nvidia.com/device-plugin.config=inference
kubectl label node train-node-1 nvidia.com/device-plugin.config=training
```

### Performance Characteristics

Time-slicing introduces overhead from CUDA context switching. The impact depends on workload type:

| Workload Type | Overhead | Recommendation |
|---|---|---|
| **Inference (batch)** | 5-10% | Good fit -- models are memory-bound, not compute-bound |
| **Inference (real-time)** | 10-20% | Acceptable for < 4 replicas; watch latency P99 |
| **Training (small models)** | 10-15% | OK for fine-tuning; use dedicated GPU for pre-training |
| **Training (large models)** | 20-30%+ | NOT recommended -- use MIG or dedicated GPUs |
| **Jupyter notebooks** | Minimal | Excellent fit -- interactive, bursty workloads |
| **CI/CD GPU tests** | 5-10% | Excellent fit -- short-lived, low utilization |

**Memory considerations:**
- All pods share the **entire GPU memory pool**.
- There is **no memory isolation** -- one pod can OOM-kill others.
- Total memory usage across all pods cannot exceed physical VRAM.
- Monitor with `nvidia-smi` or DCGM to prevent memory pressure.

### Time-Slicing vs. Alternatives

| Aspect | Time-Slicing | MIG | MPS | vGPU |
|---|---|---|---|---|
| GPU support | All NVIDIA | A100/A30/H100 | All NVIDIA | Requires vGPU license |
| Memory isolation | No | Yes (hardware) | No | Yes (hypervisor) |
| Compute isolation | No | Yes (hardware) | Partial | Yes (hypervisor) |
| Max partitions | Unlimited | 7 | 48 | Varies |
| Configuration | ConfigMap only | Node drain required | Daemon config | Host-level |
| Overhead | 5-10% | Near-zero | 3-5% | 10-15% |
| Cost | Free | Free (GPU cost) | Free | License fee |

---

## Hands-On Lab

### Prerequisites Check

```bash
# GPU Operator running
kubectl get pods -n gpu-operator | grep device-plugin

# Check current GPU capacity
kubectl get nodes -o custom-columns=NAME:.metadata.name,GPUs:.status.allocatable.nvidia\\.com/gpu
```

### Exercise 1: Enable Time-Slicing

**Goal:** Configure 4x time-slicing and verify increased GPU capacity.

**Step 1:** Apply the time-slicing ConfigMap
```bash
kubectl apply -f manifests/time-slicing-config.yaml
```

**Step 2:** Reference the ConfigMap in the GPU Operator
```bash
# If using Helm:
helm upgrade gpu-operator nvidia/gpu-operator \
  --namespace gpu-operator \
  --set devicePlugin.config.name=time-slicing-config \
  --set devicePlugin.config.default=any \
  --wait

# Or patch the ClusterPolicy directly:
kubectl patch clusterpolicy cluster-policy \
  --type merge \
  -p '{"spec":{"devicePlugin":{"config":{"name":"time-slicing-config","default":"any"}}}}'
```

**Step 3:** Restart the device plugin to pick up changes
```bash
kubectl rollout restart daemonset/nvidia-device-plugin-daemonset -n gpu-operator
kubectl rollout status daemonset/nvidia-device-plugin-daemonset -n gpu-operator
```

**Step 4:** Verify GPU capacity increased
```bash
kubectl get nodes -o custom-columns=NAME:.metadata.name,GPUs:.status.allocatable.nvidia\\.com/gpu
# With 2 physical GPUs and replicas=4, you should see 8
```

### Exercise 2: Deploy Multiple Pods on Shared GPUs

**Goal:** Run 4 pods sharing 1 physical GPU.

```bash
# Deploy 4 inference pods
kubectl apply -f - <<EOF
apiVersion: apps/v1
kind: Deployment
metadata:
  name: shared-gpu-inference
spec:
  replicas: 4
  selector:
    matchLabels:
      app: shared-inference
  template:
    metadata:
      labels:
        app: shared-inference
    spec:
      containers:
        - name: inference
          image: nvcr.io/nvidia/cuda:12.3.1-base-ubuntu22.04
          command: ["bash", "-c", "while true; do nvidia-smi; sleep 30; done"]
          resources:
            limits:
              nvidia.com/gpu: 1    # Each gets 1 time-sliced virtual GPU
EOF

# Verify all 4 pods are running
kubectl get pods -l app=shared-inference

# Check that they share the same physical GPU
for pod in $(kubectl get pods -l app=shared-inference -o name); do
  echo "--- $pod ---"
  kubectl exec $pod -- nvidia-smi --query-gpu=gpu_uuid --format=csv,noheader
done
# All pods should show the same GPU UUID
```

### Exercise 3: Monitor Time-Sliced Usage

**Goal:** Observe GPU utilization and memory with multiple tenants.

```bash
# Watch real-time GPU usage on the node
kubectl exec -it <device-plugin-pod> -n gpu-operator -- nvidia-smi dmon -s pucvmet -d 5

# Check per-process GPU usage
kubectl exec -it <any-gpu-pod> -- nvidia-smi pmon -s um -d 5
```

---

## Key Terminology

| Term | Definition |
|---|---|
| **Time-Slicing** | Sharing a GPU by context-switching between CUDA processes at the driver level |
| **CUDA Context** | Per-process GPU execution environment containing memory allocations and kernel state |
| **Replicas** | Number of virtual GPUs advertised per physical GPU in the device plugin |
| **Context Switch** | Saving one process's GPU state and loading another's (~5-10% overhead) |
| **Oversubscription** | Allocating more virtual GPUs than physical GPUs, relying on time-sharing |
| **renameByDefault** | Config option to advertise time-sliced GPUs as `nvidia.com/gpu.shared` |

---

## Common Mistakes

| Mistake | Symptom | Fix |
|---|---|---|
| Setting replicas too high | Severe performance degradation, OOM | Start with 4, benchmark, increase cautiously |
| No memory monitoring | OOM kills with no warning | Set up DCGM alerts for memory > 80% |
| Using time-slicing for large training | 2-3x slower training | Use dedicated GPUs or MIG for training |
| Forgetting to restart device plugin | Old config still active | `kubectl rollout restart daemonset/nvidia-device-plugin-daemonset` |
| Mixing time-slicing with MIG | Config conflict | Use one or the other per GPU (set `migStrategy: none` for time-slicing) |

---

## Self-Check Questions

1. How does time-slicing differ from MIG at the hardware level? What is not isolated?
2. If you set `replicas: 8` on a node with 2 GPUs, how many `nvidia.com/gpu` resources does Kubernetes see?
3. Why is time-slicing a poor choice for large model training but excellent for Jupyter notebooks?
4. What happens if 4 pods sharing a GPU collectively try to use more memory than the GPU has?
5. How would you set up a cluster with dedicated training GPUs and shared inference GPUs?

---

## You Know You Have Completed This Module When...

- [ ] Time-slicing ConfigMap applied and device plugin restarted
- [ ] Node reports increased GPU count (physical x replicas)
- [ ] Multiple pods running on the same physical GPU
- [ ] You understand the performance implications and when NOT to use time-slicing
- [ ] You can configure per-node-pool time-slicing profiles
- [ ] Self-check questions answered confidently

---

## Troubleshooting

### Common Issues

**Issue: GPU count unchanged after applying ConfigMap**
```bash
# Check ConfigMap exists
kubectl get configmap time-slicing-config -n gpu-operator

# Verify device plugin is using the ConfigMap
kubectl describe daemonset nvidia-device-plugin-daemonset -n gpu-operator | grep -A3 "config"

# Check device plugin logs
kubectl logs -n gpu-operator -l app=nvidia-device-plugin-daemonset --tail=30
```

**Issue: Pods OOM-killed on shared GPU**
```bash
# Check GPU memory usage
nvidia-smi --query-gpu=memory.used,memory.total --format=csv

# Set CUDA_MEM_LIMIT environment variable in your pods to prevent runaway allocations
env:
  - name: PYTORCH_CUDA_ALLOC_CONF
    value: "max_split_size_mb:512"
```

**Issue: Context switching causes latency spikes**
- Reduce the number of replicas
- Move latency-sensitive workloads to dedicated GPUs
- Consider MPS for inference workloads (lower context switch overhead)

---

**Next: [Module 06 - Distributed Training](../06-scheduling-and-quotas/)**
