# Module 02: NVIDIA GPU Operator on Kubernetes

| | |
|---|---|
| **Time** | 3-5 hours |
| **Difficulty** | Beginner-Intermediate |
| **Prerequisites** | Module 01 completed, kubectl and Helm installed |

---

## Learning Objectives

By the end of this module, you will be able to:

- Explain why the GPU Operator exists and what problems it solves
- Deploy the NVIDIA GPU Operator on a Kubernetes cluster using Helm
- Understand every component the operator manages (driver, toolkit, device plugin, DCGM, GFD, MIG manager)
- Verify a working GPU Operator installation with validation pods
- Troubleshoot common GPU Operator deployment failures
- Configure the ClusterPolicy CRD for production environments

---

## Concepts

### The Problem: GPU Software Stack Complexity

Running GPU workloads on Kubernetes requires a complex software stack on every node:

```
Application (PyTorch, TensorFlow, Triton)
        |
   CUDA Libraries (cuDNN, cuBLAS, NCCL)
        |
   NVIDIA Container Toolkit (libnvidia-container)
        |
   NVIDIA Device Plugin for Kubernetes
        |
   NVIDIA Driver (kernel module)
        |
   GPU Hardware (A100, H100, etc.)
```

Without the GPU Operator, you must manually install and maintain:
1. NVIDIA drivers on every GPU node
2. NVIDIA Container Toolkit for GPU-aware containers
3. Kubernetes device plugin DaemonSet for `nvidia.com/gpu` resource
4. DCGM exporter for monitoring
5. GPU Feature Discovery for node labels
6. MIG Manager for partitioning

When you scale to 50+ GPU nodes across multiple Kubernetes clusters, manual management becomes impossible. Driver updates require rolling reboots, version mismatches cause silent failures, and new nodes need provisioning.

### The Solution: NVIDIA GPU Operator

The GPU Operator is a **Kubernetes Operator** that automates the entire NVIDIA software stack. You deploy one Helm chart, and it manages everything:

```
┌─────────────────────────────────────────────────┐
│               NVIDIA GPU Operator                │
│                                                   │
│  ClusterPolicy CRD                               │
│  ┌─────────────────────────────────────────────┐ │
│  │  Driver Manager    → installs/updates driver │ │
│  │  Toolkit          → container runtime hooks  │ │
│  │  Device Plugin    → exposes nvidia.com/gpu   │ │
│  │  DCGM Exporter    → Prometheus metrics       │ │
│  │  GPU Feature Disc. → node labels             │ │
│  │  MIG Manager      → MIG partitioning         │ │
│  │  Validator        → health checks            │ │
│  │  NFD              → Node Feature Discovery   │ │
│  └─────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────┘
```

### GPU Operator Components in Detail

| Component | What It Does | DaemonSet? | Why It Matters |
|---|---|---|---|
| **NVIDIA Driver** | Compiles and loads the GPU kernel module | Yes | Without it, the kernel cannot talk to the GPU |
| **Container Toolkit** | Injects GPU device files and libraries into containers | Yes | Enables `--gpus all` in container runtimes |
| **Device Plugin** | Registers `nvidia.com/gpu` as a schedulable Kubernetes resource | Yes | Without it, `kubectl` cannot schedule GPU pods |
| **DCGM Exporter** | Exposes GPU metrics (utilization, memory, temp, ECC) as Prometheus metrics | Yes | Required for monitoring and alerting |
| **GPU Feature Discovery (GFD)** | Labels nodes with GPU properties (`nvidia.com/gpu.product=A100`) | Yes | Enables nodeAffinity and nodeSelector for GPU-specific scheduling |
| **MIG Manager** | Partitions A100/H100 GPUs into isolated MIG instances | Yes | Required for multi-tenant GPU sharing with isolation |
| **Node Feature Discovery (NFD)** | Generic node labeling (CPU features, PCIe, etc.) | Yes | Foundation for GFD |
| **Validator** | Runs smoke tests to confirm the GPU stack is healthy | Pod | Catches misconfiguration before workloads fail |

### ClusterPolicy CRD

The `ClusterPolicy` is the single source of truth for GPU Operator configuration. It controls which components are enabled, their versions, and runtime parameters.

```yaml
apiVersion: nvidia.com/v1
kind: ClusterPolicy
metadata:
  name: cluster-policy
spec:
  driver:
    enabled: true          # Set false if drivers are pre-installed
    version: "545.23.08"
  toolkit:
    enabled: true
  devicePlugin:
    enabled: true
    config:
      name: time-slicing-config   # Reference to ConfigMap
      default: any
  dcgmExporter:
    enabled: true
    serviceMonitor:
      enabled: true        # Auto-create Prometheus ServiceMonitor
  migManager:
    enabled: true
    config:
      name: mig-config     # Reference to MIG ConfigMap
  gfd:
    enabled: true
  validator:
    enabled: true
```

---

## Hands-On Lab

### Prerequisites Check

```bash
# Kubernetes cluster (kind, minikube, EKS, GKE, AKS)
kubectl version --client
kubectl get nodes

# Helm 3.x
helm version

# (Optional) Check for existing GPU resources
kubectl get nodes -o json | jq '.items[].status.capacity'
```

### Exercise 1: Deploy the GPU Operator

**Goal:** Install the NVIDIA GPU Operator on your cluster.

**Step 1:** Add the NVIDIA Helm repository
```bash
helm repo add nvidia https://helm.ngc.nvidia.com/nvidia
helm repo update
```

**Step 2:** Create the namespace
```bash
kubectl create namespace gpu-operator
```

**Step 3:** Install the GPU Operator
```bash
helm install gpu-operator nvidia/gpu-operator \
  --namespace gpu-operator \
  --set driver.enabled=true \
  --set toolkit.enabled=true \
  --set devicePlugin.enabled=true \
  --set dcgmExporter.enabled=true \
  --set migManager.enabled=true \
  --set gfd.enabled=true \
  --set nfd.enabled=true \
  --wait --timeout 10m
```

**Step 4:** Verify the installation
```bash
# All pods should be Running or Completed
kubectl get pods -n gpu-operator

# Check the ClusterPolicy status
kubectl get clusterpolicy cluster-policy -o jsonpath='{.status.state}'
# Expected: ready

# Verify GPU resources are advertised
kubectl get nodes -o json | jq '.items[].status.allocatable | select(.["nvidia.com/gpu"])'
```

### Exercise 2: Validate GPU Access

**Goal:** Confirm that GPU workloads can schedule and run.

**Step 1:** Deploy a GPU verification pod
```bash
kubectl apply -f - <<EOF
apiVersion: v1
kind: Pod
metadata:
  name: gpu-verify
spec:
  restartPolicy: OnFailure
  containers:
    - name: cuda-vectoradd
      image: nvcr.io/nvidia/k8s/cuda-sample:vectoradd-cuda12.3.1
      resources:
        limits:
          nvidia.com/gpu: 1
EOF
```

**Step 2:** Wait for completion and check logs
```bash
kubectl wait --for=condition=Ready pod/gpu-verify --timeout=120s
kubectl logs gpu-verify

# Expected output includes: "Test PASSED"
```

**Step 3:** Inspect GPU node labels created by GFD
```bash
kubectl get node <your-node> -o json | jq '.metadata.labels | with_entries(select(.key | startswith("nvidia.com")))'
```

Example labels:
```
nvidia.com/gpu.product = NVIDIA-A100-SXM4-80GB
nvidia.com/gpu.memory = 81920
nvidia.com/gpu.count = 8
nvidia.com/gpu.family = ampere
nvidia.com/mig.capable = true
nvidia.com/cuda.driver.major = 12
```

### Exercise 3: Customize the ClusterPolicy

**Goal:** Modify the GPU Operator configuration for a production environment.

**Step 1:** Export the current ClusterPolicy
```bash
kubectl get clusterpolicy cluster-policy -o yaml > cluster-policy-backup.yaml
```

**Step 2:** Apply custom settings via Helm upgrade
```bash
helm upgrade gpu-operator nvidia/gpu-operator \
  --namespace gpu-operator \
  --set devicePlugin.config.name=time-slicing-config \
  --set dcgmExporter.serviceMonitor.enabled=true \
  --set dcgmExporter.env[0].name=DCGM_EXPORTER_LISTEN \
  --set dcgmExporter.env[0].value=":9400" \
  --set validator.env[0].name=WITH_WORKLOAD \
  --set validator.env[0].value="true" \
  --wait --timeout 10m
```

**Step 3:** Verify the updated configuration
```bash
kubectl get clusterpolicy cluster-policy -o yaml | grep -A 5 "devicePlugin"
kubectl get clusterpolicy cluster-policy -o yaml | grep -A 5 "dcgmExporter"
```

---

## Key Terminology

| Term | Definition |
|---|---|
| **GPU Operator** | Kubernetes Operator that automates the NVIDIA GPU software stack lifecycle |
| **ClusterPolicy** | Custom Resource (CRD) that configures all GPU Operator components |
| **Device Plugin** | DaemonSet that registers `nvidia.com/gpu` as a Kubernetes extended resource |
| **DCGM** | Data Center GPU Manager -- NVIDIA's low-level GPU monitoring framework |
| **GFD** | GPU Feature Discovery -- labels nodes with GPU hardware properties |
| **NFD** | Node Feature Discovery -- general-purpose node labeling framework |
| **Container Toolkit** | Runtime hooks that inject GPU devices and libraries into containers |

---

## Architecture Reference

```
                    ┌─────────────┐
                    │  Helm Chart │
                    └──────┬──────┘
                           │ creates
                    ┌──────▼──────┐
                    │ ClusterPolicy│
                    └──────┬──────┘
                           │ reconciles
              ┌────────────┼────────────┐
              │            │            │
        ┌─────▼─────┐ ┌───▼────┐ ┌────▼─────┐
        │  DaemonSets│ │  Pods  │ │ConfigMaps│
        └─────┬─────┘ └───┬────┘ └──────────┘
              │            │
    ┌─────────┼─────────┐  │
    │         │         │  │
┌───▼──┐ ┌───▼──┐ ┌───▼──┐ ┌──▼───┐
│Driver│ │Plugin│ │ DCGM │ │Valid.│
└──────┘ └──────┘ └──────┘ └──────┘
```

---

## Common Mistakes

| Mistake | Symptom | Fix |
|---|---|---|
| Installing on nodes without GPUs | Driver pods CrashLoopBackOff | Use `nodeSelector` or `tolerations` to target only GPU nodes |
| Driver version mismatch with host | `nvidia-smi` fails inside containers | Set `driver.enabled=false` if host has pre-installed drivers |
| Missing containerd config | Device plugin cannot see GPUs | Ensure `toolkit.env` points to correct containerd socket path |
| Not waiting for operator readiness | GPU pods stuck in Pending | Wait for `ClusterPolicy` status = `ready` before deploying workloads |
| Forgetting ServiceMonitor for DCGM | No GPU metrics in Prometheus | Set `dcgmExporter.serviceMonitor.enabled=true` |

---

## Self-Check Questions

1. What problem does the GPU Operator solve that you cannot solve with manual installation?
2. Name all seven components the GPU Operator manages. What happens if you disable the device plugin?
3. What is the ClusterPolicy CRD and why is it the central configuration point?
4. How do GFD labels enable GPU-specific scheduling? Give an example nodeSelector.
5. Your GPU pods are stuck in Pending after installing the operator. What three things do you check first?

---

## You Know You Have Completed This Module When...

- [ ] GPU Operator is deployed and all pods are Running
- [ ] ClusterPolicy status shows `ready`
- [ ] A CUDA sample pod runs successfully and logs `Test PASSED`
- [ ] You can list GPU node labels created by GFD
- [ ] You understand every component and can explain why each is necessary
- [ ] Self-check questions answered confidently

---

## Troubleshooting

### Common Issues

**Issue: Driver pod stuck in Init or CrashLoopBackOff**
```bash
kubectl logs -n gpu-operator -l app=nvidia-driver-daemonset --tail=50
# Common causes:
# - Secure Boot enabled (disable or use pre-compiled drivers)
# - Kernel headers not installed on host
# - Incompatible driver version for your GPU
```

**Issue: Device plugin not registering GPU resources**
```bash
kubectl logs -n gpu-operator -l app=nvidia-device-plugin-daemonset --tail=50
kubectl describe node <gpu-node> | grep -A5 "Allocatable"
# Look for nvidia.com/gpu in the output
```

**Issue: DCGM exporter showing no metrics**
```bash
kubectl port-forward -n gpu-operator svc/nvidia-dcgm-exporter 9400:9400
curl http://localhost:9400/metrics | head -20
# If empty, check DCGM pod logs for NVML errors
```

---

**Next: [Module 03 - GPU Device Plugin and Scheduling](../03-gpu-sharing-strategies/)**
