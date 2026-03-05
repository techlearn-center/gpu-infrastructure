# Module 03: GPU Device Plugin and Scheduling

| | |
|---|---|
| **Time** | 3-5 hours |
| **Difficulty** | Intermediate |
| **Prerequisites** | Module 02 completed, GPU Operator running |

---

## Learning Objectives

By the end of this module, you will be able to:

- Explain how the Kubernetes device plugin framework exposes GPU resources
- Configure GPU resource requests and limits in pod specs
- Use nodeSelector and nodeAffinity for GPU-type-specific scheduling
- Implement topology-aware scheduling for multi-GPU workloads
- Set up ResourceQuotas and LimitRanges to govern GPU allocation per namespace
- Understand the scheduling flow from pod creation to GPU binding

---

## Concepts

### How Kubernetes Discovers GPUs

Kubernetes does not natively understand GPUs. The **device plugin framework** bridges this gap:

```
┌──────────────────────────────────────────────────────────┐
│                    Kubernetes Node                        │
│                                                          │
│  ┌──────────────┐    gRPC     ┌───────────────────────┐ │
│  │  kubelet      │◄──────────►│ NVIDIA Device Plugin   │ │
│  │               │            │                         │ │
│  │ - Registers   │            │ - Enumerates GPUs       │ │
│  │   nvidia.com/ │            │ - Health checks         │ │
│  │   gpu         │            │ - Allocates on request  │ │
│  └──────┬───────┘            └───────────┬─────────────┘ │
│         │                                │               │
│         │ reports to                     │ reads          │
│         ▼                                ▼               │
│  ┌──────────────┐              ┌─────────────────────┐  │
│  │  API Server   │              │  nvidia-smi / NVML  │  │
│  │  (node status)│              │  (GPU hardware)     │  │
│  └──────────────┘              └─────────────────────┘  │
└──────────────────────────────────────────────────────────┘
```

**Flow:**
1. The NVIDIA Device Plugin DaemonSet runs on every GPU node.
2. It enumerates GPUs via NVML and registers `nvidia.com/gpu` with kubelet over gRPC.
3. Kubelet reports GPU capacity/allocatable to the API server.
4. The scheduler can now place pods that request `nvidia.com/gpu`.
5. When a pod is assigned to a node, kubelet tells the device plugin which GPU to allocate.
6. The device plugin provides the container with the GPU device file and environment variables.

### GPU Resource Requests and Limits

In Kubernetes, GPUs are an **extended resource**. Unlike CPU and memory, they have special rules:

```yaml
resources:
  limits:
    nvidia.com/gpu: 1    # Request exactly 1 GPU
```

**Important rules:**
- You can only specify `limits`, not `requests` (they are always equal for extended resources).
- GPUs are **not divisible** -- you request whole GPUs (1, 2, 4, etc.).
- A pod requesting 0 GPUs will not have access to any GPU.
- GPUs are **exclusive by default** -- no two pods share a GPU (unless time-slicing or MIG is configured).
- If a node has 4 GPUs and 3 are allocated, only pods requesting 1 GPU can schedule.

### Scheduling Flow

```
Pod Created               Scheduler                     Node
    │                         │                           │
    │  PodSpec:               │                           │
    │  nvidia.com/gpu: 2      │                           │
    ├────────────────────────►│                           │
    │                         │  Filter: nodes with       │
    │                         │  >= 2 available GPUs      │
    │                         │                           │
    │                         │  Score: prefer nodes with │
    │                         │  matching topology,       │
    │                         │  fewer allocated GPUs     │
    │                         │                           │
    │                         ├──────────────────────────►│
    │                         │  Bind pod to node         │
    │                         │                           │
    │                         │              kubelet asks  │
    │                         │              device plugin │
    │                         │              for 2 GPUs    │
    │                         │                           │
    │                         │              Device plugin │
    │                         │              returns GPU   │
    │                         │              UUIDs + env   │
```

### Node Selectors and Affinity

GPU Feature Discovery (from Module 02) labels nodes with hardware properties. Use these labels to schedule pods on specific GPU types.

**nodeSelector (simple):**
```yaml
spec:
  nodeSelector:
    nvidia.com/gpu.product: "NVIDIA-A100-SXM4-80GB"
```

**nodeAffinity (advanced):**
```yaml
spec:
  affinity:
    nodeAffinity:
      requiredDuringSchedulingIgnoredDuringExecution:
        nodeSelectorTerms:
          - matchExpressions:
              - key: nvidia.com/gpu.product
                operator: In
                values:
                  - "NVIDIA-A100-SXM4-80GB"
                  - "NVIDIA-H100-SXM5-80GB"
      preferredDuringSchedulingIgnoredDuringExecution:
        - weight: 100
          preference:
            matchExpressions:
              - key: nvidia.com/gpu.memory
                operator: Gt
                values:
                  - "40000"   # Prefer GPUs with > 40 GB VRAM
```

### Topology-Aware Scheduling

For multi-GPU training, GPU placement matters. Two GPUs connected via NVLink can communicate at 600 GB/s, while PCIe-connected GPUs are limited to 64 GB/s.

```yaml
spec:
  # Topology Manager policy must be set on kubelet
  # kubelet --topology-manager-policy=best-effort
  containers:
    - name: training
      resources:
        limits:
          nvidia.com/gpu: 4    # Scheduler + Topology Manager
                                # try to place all 4 on same NUMA/NVLink domain
```

**Kubelet Topology Manager policies:**
| Policy | Behavior |
|---|---|
| `none` | No topology awareness (default) |
| `best-effort` | Try to align GPUs on same NUMA node; schedule anyway if not possible |
| `restricted` | Only schedule if GPUs can be aligned on same NUMA node |
| `single-numa-node` | Strict: all resources must come from a single NUMA node |

### ResourceQuotas and LimitRanges

In multi-tenant clusters, control GPU allocation per namespace:

```yaml
# Limit total GPUs per namespace
apiVersion: v1
kind: ResourceQuota
metadata:
  name: gpu-quota
  namespace: team-ml
spec:
  hard:
    requests.nvidia.com/gpu: "8"     # Team gets max 8 GPUs
    limits.nvidia.com/gpu: "8"
    pods: "20"                        # Max 20 pods
```

```yaml
# Default GPU request per container
apiVersion: v1
kind: LimitRange
metadata:
  name: gpu-limits
  namespace: team-ml
spec:
  limits:
    - type: Container
      default:
        nvidia.com/gpu: "1"
      defaultRequest:
        nvidia.com/gpu: "1"
      max:
        nvidia.com/gpu: "4"          # No single container gets > 4 GPUs
```

---

## Hands-On Lab

### Prerequisites Check

```bash
# Verify GPU Operator is running
kubectl get pods -n gpu-operator
kubectl get clusterpolicy cluster-policy -o jsonpath='{.status.state}'

# Check available GPU resources
kubectl get nodes -o custom-columns=NAME:.metadata.name,GPUs:.status.allocatable.nvidia\\.com/gpu
```

### Exercise 1: Schedule GPU Workloads

**Goal:** Deploy pods with GPU requests and observe scheduling behavior.

**Step 1:** Deploy a single-GPU pod
```bash
kubectl apply -f - <<EOF
apiVersion: v1
kind: Pod
metadata:
  name: single-gpu-pod
spec:
  restartPolicy: OnFailure
  containers:
    - name: cuda-check
      image: nvcr.io/nvidia/cuda:12.3.1-base-ubuntu22.04
      command: ["nvidia-smi"]
      resources:
        limits:
          nvidia.com/gpu: 1
EOF

kubectl wait --for=condition=Ready pod/single-gpu-pod --timeout=60s
kubectl logs single-gpu-pod
```

**Step 2:** Deploy a multi-GPU pod
```bash
kubectl apply -f - <<EOF
apiVersion: v1
kind: Pod
metadata:
  name: multi-gpu-pod
spec:
  restartPolicy: OnFailure
  containers:
    - name: cuda-check
      image: nvcr.io/nvidia/cuda:12.3.1-base-ubuntu22.04
      command: ["bash", "-c", "nvidia-smi -L && echo 'GPU count:' && nvidia-smi --query-gpu=index --format=csv,noheader | wc -l"]
      resources:
        limits:
          nvidia.com/gpu: 2
EOF
```

**Step 3:** Observe what happens when GPUs are exhausted
```bash
# Try to schedule more GPUs than available
kubectl apply -f - <<EOF
apiVersion: v1
kind: Pod
metadata:
  name: greedy-gpu-pod
spec:
  restartPolicy: OnFailure
  containers:
    - name: cuda-check
      image: nvcr.io/nvidia/cuda:12.3.1-base-ubuntu22.04
      command: ["nvidia-smi"]
      resources:
        limits:
          nvidia.com/gpu: 100
EOF

# This pod should stay Pending
kubectl get pod greedy-gpu-pod
kubectl describe pod greedy-gpu-pod | grep -A5 "Events"
# Look for: "Insufficient nvidia.com/gpu"
```

### Exercise 2: GPU-Specific Scheduling with Labels

**Goal:** Use GFD labels to target specific GPU types.

```bash
# List GPU labels on your node
kubectl get nodes -o json | jq '.items[].metadata.labels | to_entries[] | select(.key | startswith("nvidia.com"))'

# Deploy a pod that targets A100 GPUs only
kubectl apply -f - <<EOF
apiVersion: v1
kind: Pod
metadata:
  name: a100-only-pod
spec:
  restartPolicy: OnFailure
  nodeSelector:
    nvidia.com/gpu.product: "NVIDIA-A100-SXM4-80GB"
  containers:
    - name: training
      image: nvcr.io/nvidia/pytorch:23.12-py3
      command: ["python", "-c", "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"]
      resources:
        limits:
          nvidia.com/gpu: 1
EOF
```

### Exercise 3: Set Up GPU Quotas

**Goal:** Implement namespace-level GPU quotas for multi-tenancy.

```bash
# Create a namespace for a team
kubectl create namespace team-data-science

# Apply a ResourceQuota
kubectl apply -f - <<EOF
apiVersion: v1
kind: ResourceQuota
metadata:
  name: gpu-quota
  namespace: team-data-science
spec:
  hard:
    requests.nvidia.com/gpu: "4"
    limits.nvidia.com/gpu: "4"
EOF

# Verify the quota
kubectl get resourcequota -n team-data-science
kubectl describe resourcequota gpu-quota -n team-data-science
```

---

## Key Terminology

| Term | Definition |
|---|---|
| **Device Plugin** | gRPC server that registers extended resources (like GPUs) with kubelet |
| **Extended Resource** | Non-native Kubernetes resource (e.g., `nvidia.com/gpu`) that is integer, non-divisible, and non-overcommittable |
| **GFD Labels** | Node labels set by GPU Feature Discovery describing GPU hardware properties |
| **Topology Manager** | Kubelet component that aligns resource allocation (CPU, memory, GPU) to NUMA topology |
| **ResourceQuota** | Namespace-scoped limit on total resource consumption |
| **LimitRange** | Namespace-scoped default and max resource limits per container |

---

## Common Mistakes

| Mistake | Symptom | Fix |
|---|---|---|
| Setting `requests` instead of `limits` for GPUs | Validation error or ignored | Extended resources only use `limits` (request = limit automatically) |
| Requesting fractional GPUs (e.g., 0.5) | Pod rejected | GPUs are integers; use time-slicing (Module 05) for sharing |
| Ignoring topology for multi-GPU pods | 50% slower training | Enable Topology Manager with `best-effort` or `restricted` policy |
| No ResourceQuota on shared clusters | One team monopolizes all GPUs | Apply quotas per namespace |
| Wrong GFD label in nodeSelector | Pod stuck Pending forever | Check actual labels with `kubectl get node -o json` |

---

## Self-Check Questions

1. How does the NVIDIA device plugin communicate with kubelet? What protocol?
2. Why can you only set `limits` (not `requests`) for `nvidia.com/gpu`?
3. A node has 4 GPUs, 3 are allocated. Can a pod requesting 2 GPUs schedule on this node? Why?
4. What is the difference between `nodeSelector` and `nodeAffinity`? When would you use each?
5. How does the Topology Manager improve multi-GPU training performance?

---

## You Know You Have Completed This Module When...

- [ ] You can deploy pods with GPU resource limits and verify GPU access
- [ ] You can use GFD labels with nodeSelector and nodeAffinity
- [ ] You can explain the full scheduling flow from pod creation to GPU binding
- [ ] You have configured ResourceQuotas and LimitRanges for GPU namespaces
- [ ] You understand topology-aware scheduling and its performance implications
- [ ] Self-check questions answered confidently

---

## Troubleshooting

### Common Issues

**Issue: Pod stuck in Pending with "Insufficient nvidia.com/gpu"**
```bash
# Check allocatable GPUs on all nodes
kubectl get nodes -o custom-columns=NAME:.metadata.name,GPUs:.status.allocatable.nvidia\\.com/gpu

# Check what is currently consuming GPUs
kubectl get pods --all-namespaces -o json | jq '.items[] | select(.spec.containers[].resources.limits["nvidia.com/gpu"] != null) | {name: .metadata.name, ns: .metadata.namespace, gpus: .spec.containers[].resources.limits["nvidia.com/gpu"]}'
```

**Issue: GPU not visible inside container**
```bash
# Check NVIDIA_VISIBLE_DEVICES env var
kubectl exec <pod> -- env | grep NVIDIA
# Should show: NVIDIA_VISIBLE_DEVICES=<uuid>

# Verify container runtime is configured for GPU passthrough
kubectl exec <pod> -- nvidia-smi
```

**Issue: GFD labels missing from node**
```bash
kubectl get pods -n gpu-operator -l app=gpu-feature-discovery
kubectl logs -n gpu-operator -l app=gpu-feature-discovery --tail=20
```

---

**Next: [Module 04 - Multi-Instance GPU (MIG)](../04-time-slicing/)**
