# Module 04: Multi-Instance GPU (MIG) Partitioning

| | |
|---|---|
| **Time** | 3-5 hours |
| **Difficulty** | Intermediate |
| **Prerequisites** | Module 03 completed, A100/A30/H100 GPU (or simulation) |

---

## Learning Objectives

By the end of this module, you will be able to:

- Explain what MIG is, why it exists, and which GPUs support it
- List MIG profiles for A100-80GB and calculate resource allocations
- Configure the MIG Manager via ConfigMap and node labels
- Choose between "single" and "mixed" MIG strategies
- Deploy workloads that request specific MIG instances
- Compare MIG vs. time-slicing vs. MPS for GPU sharing

---

## Concepts

### What is MIG?

Multi-Instance GPU (MIG) is a hardware-level partitioning feature on NVIDIA A100, A30, and H100 GPUs. It divides a single physical GPU into up to 7 isolated **GPU Instances (GIs)**, each with:

- **Dedicated streaming multiprocessors (SMs)** -- guaranteed compute
- **Dedicated memory** -- isolated HBM with separate memory controllers
- **Dedicated L2 cache** -- no cache thrashing between tenants
- **Separate error isolation** -- an ECC error in one instance does not affect others

```
Physical A100-80GB GPU
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐              │
│  │ 1g.10gb  │ │ 1g.10gb  │ │ 1g.10gb  │ │ 1g.10gb  │  ... x 7    │
│  │ 1 SM     │ │ 1 SM     │ │ 1 SM     │ │ 1 SM     │              │
│  │ 10 GB    │ │ 10 GB    │ │ 10 GB    │ │ 10 GB    │              │
│  │ isolated │ │ isolated │ │ isolated │ │ isolated │              │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘              │
│                                                                     │
│  OR                                                                 │
│                                                                     │
│  ┌────────────────────┐ ┌────────────────────┐                     │
│  │     3g.40gb        │ │     3g.40gb        │                     │
│  │   3 SMs, 40 GB     │ │   3 SMs, 40 GB     │                     │
│  │   isolated         │ │   isolated         │                     │
│  └────────────────────┘ └────────────────────┘                     │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### MIG Profiles (A100-80GB)

| Profile | SMs | Memory | L2 Cache | Max Instances | Use Case |
|---|---|---|---|---|---|
| **1g.10gb** | 1/7 | 10 GB | 1/7 | 7 | Small inference, CI/CD GPU tests |
| **1g.10gb+me** | 1/7 | 10 GB + media | 1/7 | 1* | Video transcoding |
| **2g.20gb** | 2/7 | 20 GB | 2/7 | 3 | Medium inference, fine-tuning small models |
| **3g.40gb** | 3/7 | 40 GB | 3/7 | 2 | Training medium models, batch inference |
| **4g.40gb** | 4/7 | 40 GB | 4/7 | 1 | Large training (with 3g.40gb on remaining SMs) |
| **7g.80gb** | 7/7 | 80 GB | 7/7 | 1 | Full GPU (MIG disabled effectively) |

\*The `+me` profiles include the media engine for video decode/encode.

### MIG Strategies: Single vs. Mixed

| Strategy | Resource Name | How It Works | Best For |
|---|---|---|---|
| **single** | `nvidia.com/gpu` | All instances on a GPU use the same profile. Each instance appears as 1 GPU. | Homogeneous workloads (all inference, all training) |
| **mixed** | `nvidia.com/mig-<profile>` | Different profiles coexist on one GPU. Pods request specific profiles. | Multi-tenant clusters with varied workload sizes |

**Single strategy example:**
```yaml
# All 7 instances are 1g.10gb
# Pod requests nvidia.com/gpu: 1  (gets one 1g.10gb instance)
resources:
  limits:
    nvidia.com/gpu: 1
```

**Mixed strategy example:**
```yaml
# GPU has 1x 3g.40gb + 2x 1g.10gb
# Pod requests a specific profile
resources:
  limits:
    nvidia.com/mig-3g.40gb: 1    # Gets the 3g.40gb instance
```

### MIG vs. Time-Slicing vs. MPS

| Feature | MIG | Time-Slicing | MPS |
|---|---|---|---|
| **Isolation** | Full (memory, compute, cache) | None (shared everything) | Partial (shared memory) |
| **Memory protection** | Hardware-enforced | None | None |
| **Supported GPUs** | A100, A30, H100 only | Any NVIDIA GPU | Any NVIDIA GPU |
| **Max partitions** | 7 (A100) | Unlimited (software) | 48 clients |
| **Overhead** | Near-zero | 5-10% context switch | 3-5% |
| **Reconfiguration** | Requires draining pods | ConfigMap + restart | Dynamic |
| **Best for** | Production multi-tenancy | Dev/test sharing | Inference serving |

**Decision tree:**
1. Need hard memory isolation? **Use MIG** (if GPU supports it)
2. GPU does not support MIG? **Use time-slicing** for dev/test, **MPS** for inference
3. Need maximum flexibility? **Use time-slicing** (any GPU, easy to reconfigure)

---

## Hands-On Lab

### Prerequisites Check

```bash
# Verify MIG capability
kubectl get nodes -o json | jq '.items[].metadata.labels["nvidia.com/mig.capable"]'
# Should return "true" for A100/A30/H100 nodes

# Check current MIG status
nvidia-smi mig -lgip    # List GPU Instance profiles
nvidia-smi mig -lgi     # List active GPU Instances
```

### Exercise 1: Enable MIG and Create Instances

**Goal:** Enable MIG mode and partition a GPU.

**Step 1:** Enable MIG mode on a GPU (requires exclusive access)
```bash
# Enable MIG on GPU 0
sudo nvidia-smi -i 0 -mig 1

# Verify MIG is enabled
nvidia-smi -i 0 --query-gpu=mig.mode.current --format=csv,noheader
# Expected: Enabled
```

**Step 2:** Create MIG instances
```bash
# Create 7x 1g.10gb instances (A100-80GB)
sudo nvidia-smi mig -i 0 -cgi 19,19,19,19,19,19,19 -C

# List instances
nvidia-smi mig -lgi
nvidia-smi mig -lci
```

**Step 3:** Verify partitions
```bash
nvidia-smi
# You should see 7 MIG devices listed under the GPU
```

### Exercise 2: Configure MIG Manager on Kubernetes

**Goal:** Use the GPU Operator's MIG Manager for declarative partitioning.

**Step 1:** Apply the MIG ConfigMap
```bash
kubectl apply -f manifests/mig-config.yaml
```

**Step 2:** Label a node with the desired MIG profile
```bash
# Apply the "all-1g.10gb" profile
kubectl label node <gpu-node> nvidia.com/mig.config=all-1g.10gb --overwrite

# The MIG Manager will:
# 1. Drain GPU pods from the node
# 2. Destroy existing MIG instances
# 3. Create new instances matching the profile
# 4. Un-cordon the node
```

**Step 3:** Watch the MIG Manager reconfigure
```bash
kubectl logs -n gpu-operator -l app=nvidia-mig-manager -f

# Verify new GPU resources
kubectl get node <gpu-node> -o json | jq '.status.allocatable | to_entries[] | select(.key | contains("nvidia"))'
```

### Exercise 3: Deploy Workloads on MIG Instances

**Goal:** Schedule pods that target specific MIG profiles.

```bash
# Deploy a small inference pod on a 1g.10gb instance
kubectl apply -f - <<EOF
apiVersion: v1
kind: Pod
metadata:
  name: mig-inference
spec:
  restartPolicy: OnFailure
  containers:
    - name: inference
      image: nvcr.io/nvidia/cuda:12.3.1-base-ubuntu22.04
      command: ["nvidia-smi"]
      resources:
        limits:
          nvidia.com/mig-1g.10gb: 1
EOF

kubectl logs mig-inference
# The nvidia-smi output should show only the 1g.10gb partition
```

---

## Key Terminology

| Term | Definition |
|---|---|
| **MIG (Multi-Instance GPU)** | Hardware partitioning feature that splits one GPU into up to 7 isolated instances |
| **GPU Instance (GI)** | A MIG partition with dedicated SMs and memory |
| **Compute Instance (CI)** | A subdivision of a GI (usually 1:1 with GI) |
| **MIG Profile** | Specification like `1g.10gb` defining SM count and memory for an instance |
| **MIG Strategy** | `single` (homogeneous) or `mixed` (heterogeneous) profile assignment |
| **MIG Manager** | GPU Operator component that declaratively manages MIG configurations |

---

## Common Mistakes

| Mistake | Symptom | Fix |
|---|---|---|
| Enabling MIG on a non-supported GPU (T4, V100) | Error: "MIG is not supported" | MIG requires A100, A30, or H100 |
| Forgetting to drain pods before MIG reconfiguration | GPU reset fails | Let the MIG Manager handle draining, or drain manually first |
| Using `nvidia.com/gpu` with mixed strategy | Pod gets wrong-sized instance | Use `nvidia.com/mig-<profile>` for mixed strategy |
| Creating incompatible profile combinations | nvidia-smi error | Check valid profile combinations in NVIDIA documentation |
| Not setting GFD's `MIG_STRATEGY` env var | Node labels incorrect | Set `gfd.env.GFD_MIG_STRATEGY=mixed` in ClusterPolicy |

---

## Self-Check Questions

1. What makes MIG fundamentally different from time-slicing? Name three isolation guarantees.
2. How many 1g.10gb instances can you create on an A100-80GB? What about 2g.20gb?
3. When should you use the "single" vs. "mixed" MIG strategy?
4. What happens to running pods when you change a node's MIG configuration?
5. A team needs 5 isolated GPU partitions with at least 15 GB memory each. What MIG profile and GPU would you recommend?

---

## You Know You Have Completed This Module When...

- [ ] You can explain MIG isolation guarantees (compute, memory, cache, ECC)
- [ ] You can list MIG profiles and their resource allocations for A100
- [ ] You have configured the MIG Manager via ConfigMap and node labels
- [ ] You can deploy pods targeting specific MIG instances
- [ ] You can articulate when to use MIG vs. time-slicing vs. MPS
- [ ] Self-check questions answered confidently

---

## Troubleshooting

### Common Issues

**Issue: "MIG mode change requires GPU reset"**
```bash
# All GPU processes must be stopped first
sudo fuser -v /dev/nvidia*         # Find processes using GPU
sudo nvidia-smi -i 0 -r            # Reset GPU
sudo nvidia-smi -i 0 -mig 1       # Then enable MIG
```

**Issue: MIG Manager not reconfiguring after label change**
```bash
kubectl logs -n gpu-operator -l app=nvidia-mig-manager --tail=50
# Common causes:
# - Pods still running on the GPU (drain first)
# - Invalid profile name in label
# - ConfigMap not applied
```

**Issue: Pod requesting MIG resource stuck Pending**
```bash
# Check available MIG resources
kubectl get node <gpu-node> -o json | jq '.status.allocatable | to_entries[] | select(.key | contains("mig"))'

# Ensure GFD strategy matches
kubectl get pods -n gpu-operator -l app=gpu-feature-discovery -o yaml | grep MIG_STRATEGY
```

---

**Next: [Module 05 - Time-Slicing for Shared GPU Access](../05-cuda-container-toolkit/)**
