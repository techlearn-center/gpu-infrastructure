# Module 10: Production GPU Infrastructure (Autoscaling, Preemption, Quotas)

| | |
|---|---|
| **Time** | 3-5 hours |
| **Difficulty** | Advanced |
| **Prerequisites** | Modules 01-09 completed |

---

## Learning Objectives

By the end of this module, you will be able to:

- Design a production-grade GPU Kubernetes cluster end-to-end
- Implement GPU-aware autoscaling (Cluster Autoscaler + KEDA)
- Configure priority classes and preemption for GPU workloads
- Set up multi-tenant GPU quotas with fair-share scheduling
- Build a complete observability stack (metrics, logs, alerts, SLOs)
- Plan capacity, handle failures, and implement disaster recovery

---

## Concepts

### Production GPU Cluster Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Production GPU Cluster                                │
│                                                                          │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────┐                 │
│  │ Ingress /   │  │   API Layer  │  │  Training       │                 │
│  │ Load Balancer│  │  (Triton,   │  │  Orchestrator   │                 │
│  │              │  │   vLLM)     │  │  (Kubeflow,     │                 │
│  └──────┬───────┘  └──────┬──────┘  │   Volcano)      │                 │
│         │                  │         └───────┬─────────┘                 │
│         ▼                  ▼                 ▼                           │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                    Kubernetes Control Plane                       │   │
│  │  ┌──────────┐ ┌────────────┐ ┌───────────┐ ┌─────────────────┐ │   │
│  │  │Scheduler │ │ Autoscaler │ │ GPU Oper. │ │ Priority/Quota  │ │   │
│  │  │(topology │ │ (CA + KEDA │ │ (driver,  │ │ Controller      │ │   │
│  │  │ aware)   │ │  + VPA)    │ │  plugin,  │ │                 │ │   │
│  │  │          │ │            │ │  DCGM)    │ │                 │ │   │
│  │  └──────────┘ └────────────┘ └───────────┘ └─────────────────┘ │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                                                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                  │
│  │ GPU Node Pool│  │ GPU Node Pool│  │ CPU Node Pool │                  │
│  │ (Training)   │  │ (Inference)  │  │ (System/Util) │                  │
│  │ 8x A100 each │  │ 4x T4 each  │  │ No GPUs       │                  │
│  │ On-demand    │  │ Spot         │  │ Spot           │                  │
│  │ Dedicated    │  │ Time-sliced  │  │               │                  │
│  └──────────────┘  └──────────────┘  └──────────────┘                  │
│                                                                          │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                    Observability Layer                            │   │
│  │  Prometheus + Grafana + AlertManager + Loki + Jaeger             │   │
│  └──────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```

### GPU-Aware Autoscaling

**Cluster Autoscaler** adds/removes GPU nodes based on pending pods:

```yaml
# Cluster Autoscaler config for GPU node pools
apiVersion: apps/v1
kind: Deployment
metadata:
  name: cluster-autoscaler
  namespace: kube-system
spec:
  template:
    spec:
      containers:
        - name: cluster-autoscaler
          command:
            - ./cluster-autoscaler
            - --cloud-provider=aws
            - --nodes=0:16:gpu-training-pool     # Min 0, Max 16 nodes
            - --nodes=0:32:gpu-inference-pool
            - --scale-down-delay-after-add=10m    # Wait 10m after scale-up
            - --scale-down-unneeded-time=10m
            - --scale-down-utilization-threshold=0.3
            - --skip-nodes-with-system-pods=false
            - --balance-similar-node-groups=true
            - --expendable-pods-priority-cutoff=-10
            - --max-graceful-termination-sec=600  # 10m for checkpoints
```

**KEDA** (Kubernetes Event-Driven Autoscaling) for inference:

```yaml
# Scale inference pods based on request queue depth
apiVersion: keda.sh/v1alpha1
kind: ScaledObject
metadata:
  name: inference-scaler
spec:
  scaleTargetRef:
    name: inference-deployment
  minReplicaCount: 1
  maxReplicaCount: 16
  triggers:
    - type: prometheus
      metadata:
        serverAddress: http://prometheus:9090
        metricName: inference_queue_depth
        query: sum(inference_request_queue_length)
        threshold: "10"    # Scale up when queue > 10
  advanced:
    horizontalPodAutoscalerConfig:
      behavior:
        scaleUp:
          stabilizationWindowSeconds: 60    # Wait 1 min
          policies:
            - type: Pods
              value: 4         # Add max 4 pods at a time
              periodSeconds: 60
        scaleDown:
          stabilizationWindowSeconds: 300   # Wait 5 min before scale-down
```

### Priority Classes and Preemption

In a shared GPU cluster, not all workloads are equal. Priority classes determine which workloads can preempt others when GPUs are scarce:

```yaml
# Critical inference (never preempted)
apiVersion: scheduling.k8s.io/v1
kind: PriorityClass
metadata:
  name: gpu-critical-inference
value: 1000000
globalDefault: false
preemptionPolicy: PreemptLowerPriority
description: "Production inference -- cannot be preempted"

---
# Training (can preempt dev, can be preempted by inference)
apiVersion: scheduling.k8s.io/v1
kind: PriorityClass
metadata:
  name: gpu-training
value: 100000
preemptionPolicy: PreemptLowerPriority
description: "Training jobs -- preempt dev workloads"

---
# Development (lowest priority, preempted by everything)
apiVersion: scheduling.k8s.io/v1
kind: PriorityClass
metadata:
  name: gpu-development
value: 1000
preemptionPolicy: Never         # Never preempt others
description: "Dev notebooks -- always preemptable"
```

**Usage:**
```yaml
spec:
  priorityClassName: gpu-training
  containers:
    - name: trainer
      resources:
        limits:
          nvidia.com/gpu: 4
```

**Preemption flow:**
1. High-priority pod cannot schedule (no GPUs available).
2. Scheduler identifies lower-priority pods that could be evicted.
3. Lower-priority pods receive SIGTERM and have `terminationGracePeriodSeconds` to save state.
4. After eviction, high-priority pod schedules on freed GPUs.

### Multi-Tenant GPU Quotas

```yaml
# Per-team GPU quotas
---
apiVersion: v1
kind: Namespace
metadata:
  name: team-nlp
  labels:
    team: nlp

---
apiVersion: v1
kind: ResourceQuota
metadata:
  name: gpu-quota
  namespace: team-nlp
spec:
  hard:
    requests.nvidia.com/gpu: "16"    # 16 GPUs max
    limits.nvidia.com/gpu: "16"
    persistentvolumeclaims: "20"
    pods: "50"

---
# Fair-share scheduling with Volcano
apiVersion: scheduling.volcano.sh/v1beta1
kind: Queue
metadata:
  name: team-nlp
spec:
  reclaimable: true
  weight: 3           # 3x the share of a weight-1 team
  capability:
    nvidia.com/gpu: 16
```

### Capacity Planning

**Sizing formula:**
```
Required GPUs = sum(workloads) + buffer

Training:
  Active training jobs x GPUs per job x (1 / utilization target)
  Example: 3 jobs x 8 GPUs x (1 / 0.85) = 28 GPUs

Inference:
  Peak QPS x latency per request x GPUs per replica / batch_size
  Example: 1000 QPS x 0.05s x 1 GPU / 32 batch = 1.6 GPUs (round to 2)

Development:
  Active developers x (1 GPU average) x (1 / time-slicing factor)
  Example: 10 devs x 1 GPU / 4 (time-slicing) = 3 GPUs

Buffer: 20% for bursts and failures
Total: (28 + 2 + 3) x 1.2 = 40 GPUs = 5 DGX nodes
```

### Failure Handling and Disaster Recovery

| Failure Type | Detection | Recovery |
|---|---|---|
| **Single GPU failure** | DCGM XID/ECC alerts | Drain node, replace GPU, reinstate |
| **Node failure** | Node NotReady event | Reschedule pods on healthy nodes |
| **Network failure** | NCCL timeout alerts | Restart training from checkpoint |
| **Spot preemption** | Cloud provider event | Resume from checkpoint on new nodes |
| **Driver crash** | nvidia-smi error | GPU Operator restarts driver pod |
| **Cluster-wide outage** | All nodes down | Restore from etcd backup, redeploy |

**Health check implementation:**
```yaml
# Readiness probe that checks GPU health
containers:
  - name: inference
    readinessProbe:
      exec:
        command:
          - bash
          - -c
          - "nvidia-smi > /dev/null 2>&1 && python -c 'import torch; torch.cuda.current_device()'"
      initialDelaySeconds: 30
      periodSeconds: 30
    livenessProbe:
      exec:
        command:
          - bash
          - -c
          - "nvidia-smi > /dev/null 2>&1"
      initialDelaySeconds: 60
      periodSeconds: 60
      failureThreshold: 3
```

### SLOs for GPU Infrastructure

| SLO | Target | How to Measure |
|---|---|---|
| **GPU Availability** | 99.9% | Time GPUs are schedulable / total time |
| **Scheduling Latency** | P99 < 60s | Time from pod creation to GPU binding |
| **Training Job SLA** | 99% completion | Jobs completed / jobs submitted |
| **Inference Latency** | P99 < 100ms | Request latency at the service level |
| **GPU Utilization** | > 70% cluster-wide | Average DCGM_FI_DEV_GPU_UTIL |
| **Cost Efficiency** | < $X per GPU-hour | Total spend / total allocated GPU-hours |

---

## Hands-On Lab

### Exercise 1: Configure Priority-Based Preemption

**Goal:** Set up a priority system where training preempts development workloads.

```bash
# Create priority classes
kubectl apply -f - <<EOF
apiVersion: scheduling.k8s.io/v1
kind: PriorityClass
metadata:
  name: gpu-production
value: 1000000
preemptionPolicy: PreemptLowerPriority
---
apiVersion: scheduling.k8s.io/v1
kind: PriorityClass
metadata:
  name: gpu-development
value: 1000
preemptionPolicy: Never
EOF

# Deploy a low-priority pod consuming GPUs
kubectl apply -f - <<EOF
apiVersion: v1
kind: Pod
metadata:
  name: dev-notebook
spec:
  priorityClassName: gpu-development
  terminationGracePeriodSeconds: 30
  containers:
    - name: jupyter
      image: nvcr.io/nvidia/pytorch:23.12-py3
      command: ["sleep", "infinity"]
      resources:
        limits:
          nvidia.com/gpu: 1
EOF

# Now deploy a high-priority pod -- it should preempt the dev pod
kubectl apply -f - <<EOF
apiVersion: v1
kind: Pod
metadata:
  name: production-training
spec:
  priorityClassName: gpu-production
  containers:
    - name: trainer
      image: nvcr.io/nvidia/pytorch:23.12-py3
      command: ["sleep", "infinity"]
      resources:
        limits:
          nvidia.com/gpu: 1
EOF

# Watch the dev pod get evicted
kubectl get events --sort-by=.metadata.creationTimestamp | tail -10
```

### Exercise 2: Set Up GPU Autoscaling

**Goal:** Configure Cluster Autoscaler for GPU node pools.

```bash
# For EKS:
eksctl create nodegroup \
  --cluster gpu-cluster \
  --name gpu-training-spot \
  --instance-types p4d.24xlarge \
  --capacity-type SPOT \
  --nodes-min 0 \
  --nodes-max 8 \
  --node-labels "workload-type=training"

# For GKE:
gcloud container node-pools create gpu-training \
  --cluster gpu-cluster \
  --machine-type a2-highgpu-8g \
  --accelerator type=nvidia-tesla-a100,count=8 \
  --enable-autoscaling \
  --min-nodes 0 \
  --max-nodes 8 \
  --spot
```

### Exercise 3: Implement Complete Observability

**Goal:** Deploy the full monitoring stack for production.

```bash
# Start all services
docker compose up -d

# Verify endpoints
curl -s http://localhost:9090/api/v1/targets | jq '.data.activeTargets | length'  # Prometheus
curl -s http://localhost:9400/metrics | head -5                                      # DCGM
curl -s http://localhost:3000/api/health                                            # Grafana

# Import GPU dashboard in Grafana
# Dashboard ID: 12239 (NVIDIA DCGM Exporter Dashboard)
```

---

## Key Terminology

| Term | Definition |
|---|---|
| **Cluster Autoscaler** | Kubernetes controller that adds/removes nodes based on pending pods |
| **KEDA** | Kubernetes Event-Driven Autoscaling -- scales workloads based on event sources |
| **PriorityClass** | Kubernetes resource that assigns scheduling priority and preemption behavior |
| **Preemption** | Evicting lower-priority pods to make room for higher-priority ones |
| **Fair-Share Scheduling** | Allocating resources proportionally to configured weights per queue/team |
| **Volcano** | Kubernetes batch scheduling system with GPU-aware fair-share queues |
| **SLO** | Service Level Objective -- target metric for infrastructure reliability |
| **PDB** | PodDisruptionBudget -- limits how many pods can be evicted simultaneously |

---

## Common Mistakes

| Mistake | Symptom | Fix |
|---|---|---|
| No priority classes | Training jobs wait behind idle notebooks | Define 3+ priority tiers |
| Autoscaler max too low | Pods stuck Pending despite budget | Set max-nodes high enough for peak demand |
| No terminationGracePeriodSeconds | Preempted pods lose work | Set to 300-600s and implement SIGTERM checkpoint handler |
| Single node pool for all workloads | Cannot apply different policies | Create separate pools for training, inference, and dev |
| No PodDisruptionBudget for inference | Rolling updates cause downtime | Set `minAvailable: 1` PDB for inference deployments |

---

## Self-Check Questions

1. Design a priority hierarchy for a cluster shared by ML training, production inference, and development. What values would you assign?
2. How does the Cluster Autoscaler decide when to add a GPU node? When to remove one? What is the risk of too-aggressive scale-down?
3. A training job is preempted by a higher-priority inference deployment. What happens to the training job's progress?
4. You need 40 A100 GPUs for training, 8 T4s for inference, and 4 GPUs for development. Design the node pool configuration.
5. What SLOs would you set for a production GPU cluster? How would you measure them?

---

## You Know You Have Completed This Module When...

- [ ] You can design a production GPU cluster with separate node pools
- [ ] Priority classes and preemption are configured and tested
- [ ] GPU autoscaling works for both training and inference
- [ ] Multi-tenant quotas and fair-share scheduling are in place
- [ ] Full observability stack is deployed with GPU-specific dashboards and alerts
- [ ] You can articulate capacity planning and failure recovery strategies
- [ ] Self-check questions answered confidently

---

## Troubleshooting

### Common Issues

**Issue: Cluster Autoscaler not scaling up for GPU pods**
```bash
# Check autoscaler logs
kubectl logs -n kube-system -l app=cluster-autoscaler --tail=50

# Common causes:
# - Max nodes already reached
# - Instance type not available in the region
# - IAM permissions missing for node group scaling
# - Pod requests resources that no node pool can satisfy
```

**Issue: Preemption not working**
```bash
# Verify priority classes exist
kubectl get priorityclass

# Check if the lower-priority pod is eligible for preemption
kubectl describe pod <high-priority-pod> | grep -A10 "Events"
# Look for: "Preempting" events

# Ensure preemptionPolicy is PreemptLowerPriority (not Never)
```

**Issue: Inference latency spikes during scale-up**
- Set `minReplicaCount > 0` in KEDA ScaledObject to keep warm replicas
- Use PodDisruptionBudget to prevent too many replicas going down at once
- Pre-pull GPU container images on nodes to avoid cold-start pulls

---

**Next: [Capstone Project](../../capstone/)**
