# Capstone Project: Production GPU Infrastructure on Kubernetes

## Overview

This capstone project combines everything from Modules 01-10 into a single, production-grade GPU infrastructure deployment. You will design, deploy, and operate a multi-tenant GPU cluster that supports distributed training, shared inference, monitoring, autoscaling, and cost optimization.

This is the project you showcase to hiring managers to demonstrate hands-on GPU infrastructure expertise.

---

## The Challenge

Build a complete GPU Kubernetes cluster that serves two teams:

- **Team ML** -- runs large distributed training jobs (DDP/FSDP) on dedicated A100 GPUs
- **Team Product** -- runs inference APIs and interactive notebooks on shared GPUs

Your cluster must handle all of the following simultaneously:

| Requirement | Module Reference |
|---|---|
| GPU fundamentals applied to capacity planning | Module 01 |
| GPU Operator deployed and validated | Module 02 |
| Device plugin scheduling with nodeAffinity | Module 03 |
| MIG partitioning for inference isolation | Module 04 |
| Time-slicing for development notebooks | Module 05 |
| DDP training job running across GPUs | Module 06 |
| DCGM + Prometheus + Grafana monitoring | Module 07 |
| Spot instances with checkpointing for training | Module 08 |
| NCCL-optimized multi-node communication | Module 09 |
| Priority classes, autoscaling, quotas | Module 10 |

---

## Architecture

Design your solution with the following components:

```
┌──────────────────────────────────────────────────────────────────────┐
│                  Capstone GPU Cluster                                 │
│                                                                       │
│  Node Pools:                                                          │
│  ┌─────────────────┐  ┌─────────────────┐  ┌────────────────────┐   │
│  │ Training Pool    │  │ Inference Pool   │  │ System Pool         │   │
│  │ 2+ nodes         │  │ 2+ nodes         │  │ 1 node              │   │
│  │ 8x A100 each     │  │ 4x T4 each      │  │ No GPUs             │   │
│  │ On-demand / Spot │  │ MIG + time-slice │  │ Control plane pods  │   │
│  │ Dedicated GPUs   │  │ Shared GPUs      │  │                     │   │
│  │ Priority: HIGH   │  │ Priority: MED    │  │                     │   │
│  └─────────────────┘  └─────────────────┘  └────────────────────┘   │
│                                                                       │
│  Namespaces:                                                          │
│  ┌─────────────────┐  ┌─────────────────┐                            │
│  │ team-ml          │  │ team-product     │                            │
│  │ Quota: 16 GPUs   │  │ Quota: 8 GPUs   │                            │
│  │ Training jobs    │  │ Inference APIs   │                            │
│  │ Jupyter (shared) │  │ Notebooks       │                            │
│  └─────────────────┘  └─────────────────┘                            │
│                                                                       │
│  Platform Services:                                                   │
│  ┌─────────┐ ┌────────────┐ ┌──────────┐ ┌─────────┐               │
│  │GPU Oper.│ │ Prometheus │ │ Grafana  │ │Autoscale│               │
│  │+ DCGM   │ │ + Alerts   │ │Dashboard │ │ (CA+KEDA)│               │
│  └─────────┘ └────────────┘ └──────────┘ └─────────┘               │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Requirements

### Must Have (70% of grade)

- [ ] **GPU Operator** deployed with ClusterPolicy validated (`status: ready`)
- [ ] **Two namespaces** with ResourceQuotas limiting GPU allocation per team
- [ ] **Three PriorityClasses** (production-inference > training > development) with preemption
- [ ] **MIG configuration** on at least one node (e.g., 7x 1g.10gb for inference)
- [ ] **Time-slicing** on at least one node (e.g., 4 replicas for notebooks)
- [ ] **DDP training job** running across 2+ GPUs with checkpointing
- [ ] **Monitoring stack**: DCGM Exporter + Prometheus + Grafana with GPU dashboard
- [ ] **At least 3 alerting rules**: GPU idle, thermal throttling, ECC errors
- [ ] **Cluster Autoscaler** configured for GPU node pools (min: 0, max: N)
- [ ] **Documentation** explaining every component and how they fit together

### Nice to Have (30% of grade)

- [ ] **FSDP or DeepSpeed** training example (in addition to DDP)
- [ ] **KEDA** scaling inference pods based on queue depth or request count
- [ ] **Spot instance** node pool with SIGTERM checkpoint handler
- [ ] **Cost report** showing estimated monthly spend and optimization opportunities
- [ ] **NCCL benchmark** results comparing intra-node vs. inter-node bandwidth
- [ ] **CI/CD pipeline** that validates GPU Operator health before deploying workloads
- [ ] **Disaster recovery plan** for single GPU failure, node failure, and cluster failure
- [ ] **Custom pynvml exporter** deployed alongside DCGM for comparison

---

## Getting Started

### Step 1: Set Up Your Environment

```bash
# Clone the repo and set up
cd gpu-infrastructure
cp .env.example .env
pip install -r requirements.txt

# Option A: Local cluster (Kind + Docker Compose)
docker compose up -d

# Option B: Cloud cluster (EKS/GKE/AKS)
# Provision a cluster with GPU node pools
# See Module 02 for GPU Operator installation
```

### Step 2: Deploy Core Infrastructure

```bash
# Deploy GPU Operator
helm repo add nvidia https://helm.ngc.nvidia.com/nvidia
helm install gpu-operator nvidia/gpu-operator \
  --namespace gpu-operator --create-namespace \
  --wait --timeout 10m

# Apply MIG configuration
kubectl apply -f manifests/mig-config.yaml

# Apply time-slicing configuration
kubectl apply -f manifests/time-slicing-config.yaml

# Deploy monitoring stack
docker compose up -d prometheus grafana dcgm-exporter
```

### Step 3: Create Namespaces and Quotas

```bash
# Create team namespaces with GPU quotas
kubectl create namespace team-ml
kubectl create namespace team-product

kubectl apply -f - <<EOF
apiVersion: v1
kind: ResourceQuota
metadata:
  name: gpu-quota
  namespace: team-ml
spec:
  hard:
    requests.nvidia.com/gpu: "16"
    limits.nvidia.com/gpu: "16"
---
apiVersion: v1
kind: ResourceQuota
metadata:
  name: gpu-quota
  namespace: team-product
spec:
  hard:
    requests.nvidia.com/gpu: "8"
    limits.nvidia.com/gpu: "8"
EOF
```

### Step 4: Deploy Workloads

```bash
# Run distributed training
torchrun --nproc_per_node=4 src/training/distributed_training.py \
  --epochs 5 --batch-size 64 --mixed-precision

# Run GPU metrics exporter
python src/monitoring/gpu_metrics.py
```

### Step 5: Validate

```bash
bash capstone/validation/validate.sh
```

---

## Evaluation Criteria

| Criteria | Weight | What We Look For |
|---|---|---|
| **Functionality** | 30% | All must-have items work end-to-end |
| **Architecture** | 20% | Clean separation of concerns, proper node pools, namespace isolation |
| **Monitoring** | 15% | GPU-specific dashboards, meaningful alerts, SLO tracking |
| **Security** | 15% | Quotas enforced, RBAC per team, no over-permissive access |
| **Automation** | 10% | Autoscaling works, minimal manual steps, reproducible setup |
| **Documentation** | 10% | Architecture diagram, component explanations, operational runbook |

---

## Solution

The `solution/` directory contains a reference implementation. Try to complete the capstone yourself first -- that is what builds real skills and interview confidence.

```bash
ls capstone/solution/
```

---

## Showcasing to Hiring Managers

When you complete this capstone:

1. **Fork this repo** to your personal GitHub
2. **Add your solution** with detailed commit messages showing your thought process
3. **Include screenshots** of your Grafana dashboards, `nvidia-smi` output, and training logs
4. **Write an architecture decision record** explaining why you chose each component
5. **Record a 5-minute demo video** walking through the cluster (optional but impressive)
6. **Reference it on your resume**: "Designed and deployed production GPU infrastructure on Kubernetes supporting distributed training (DDP/FSDP), multi-tenant isolation (MIG + time-slicing), and autoscaling with monitoring"

### Interview Talking Points

From this capstone, you can speak confidently about:

- **GPU Operator**: "I deployed and configured the NVIDIA GPU Operator to automate driver, toolkit, device plugin, and monitoring across all GPU nodes."
- **MIG vs. Time-Slicing**: "I used MIG for production inference requiring memory isolation and time-slicing for development workloads where flexibility matters more than isolation."
- **Distributed Training**: "I implemented DDP with NCCL for multi-GPU training, including automatic checkpointing for spot instance resilience."
- **Monitoring**: "I built GPU observability with DCGM Exporter, Prometheus, and Grafana, with alerts for thermal throttling, ECC errors, and idle GPUs."
- **Cost Optimization**: "I reduced GPU costs by 60% through right-sizing (T4 for inference vs. A100 for training), spot instances, and idle detection."
- **Multi-tenancy**: "I implemented namespace-level GPU quotas, priority-based preemption, and fair-share scheduling across teams."

See [docs/portfolio-guide.md](../docs/portfolio-guide.md) for more detailed guidance.
