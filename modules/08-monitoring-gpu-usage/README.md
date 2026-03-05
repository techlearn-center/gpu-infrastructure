# Module 08: GPU Cost Optimization

| | |
|---|---|
| **Time** | 3-5 hours |
| **Difficulty** | Advanced |
| **Prerequisites** | Module 07 completed |

---

## Learning Objectives

By the end of this module, you will be able to:

- Analyze GPU costs across AWS, GCP, and Azure and choose the most cost-effective option
- Implement spot/preemptible instances for fault-tolerant GPU workloads
- Right-size GPU allocations based on actual utilization data
- Set up idle GPU detection and automatic scale-down
- Design scheduling policies that maximize GPU utilization
- Calculate and report GPU cost-per-training-run and cost-per-inference

---

## Concepts

### The GPU Cost Problem

GPU compute is the single largest expense in ML infrastructure. A single H100 instance costs ~$30-40/hour on-demand. A 64-GPU training run can cost $50,000+ per week.

```
Example: Training a 13B parameter model

   On-Demand (p5.48xlarge, 8x H100):
   $98.32/hr x 168 hours (1 week) = $16,518

   Spot Instances (70% discount):
   $29.50/hr x 168 hours = $4,956
   Savings: $11,562 per week

   Right-Sizing (A100 instead of H100, if sufficient):
   $32.77/hr x 200 hours = $6,554
   Savings: $9,964 per week

   Idle Elimination (actual utilization was 60%):
   Should have used 60% of the time = $9,911
   Wasted: $6,607 per week
```

### Cloud GPU Pricing Comparison

| GPU | AWS Instance | On-Demand $/hr | Spot $/hr | GCP $/hr | Azure $/hr |
|---|---|---|---|---|---|
| **T4** | g4dn.xlarge | $0.526 | $0.158 | $0.350 | $0.526 |
| **A10G** | g5.xlarge | $1.006 | $0.302 | -- | -- |
| **A100 40GB** | p4d.24xlarge (8) | $32.77 | $9.83 | $26.44 | $28.99 |
| **A100 80GB** | p4de.24xlarge (8) | $40.97 | $12.29 | $32.48 | $35.62 |
| **H100** | p5.48xlarge (8) | $98.32 | $29.50 | $76.14 | $83.52 |

*Prices approximate as of 2025. Check cloud provider pricing pages for current rates.*

### Strategy 1: Spot/Preemptible Instances

Spot instances offer 60-90% discounts but can be reclaimed with 2 minutes notice.

**Best practices for GPU spot instances:**
```yaml
# Kubernetes node pool with spot instances
apiVersion: v1
kind: Node
metadata:
  labels:
    cloud.google.com/gke-preemptible: "true"    # GKE
    # OR
    eks.amazonaws.com/capacityType: SPOT         # EKS

# Toleration for spot node taint
spec:
  tolerations:
    - key: "cloud.google.com/gke-preemptible"
      operator: "Equal"
      value: "true"
      effect: "NoSchedule"
```

**Checkpoint strategy for spot instances:**
```python
# Save checkpoint every N steps to survive preemption
if step % CHECKPOINT_INTERVAL == 0:
    save_checkpoint(model, optimizer, step, loss)

# On startup, resume from latest checkpoint
checkpoint = load_latest_checkpoint()
if checkpoint:
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    start_step = checkpoint["step"]
```

**Spot instance suitability:**
| Workload | Spot-Friendly? | Why |
|---|---|---|
| Training with checkpoints | Yes | Resume from last checkpoint on interruption |
| Batch inference | Yes | Stateless, can retry failed batches |
| Real-time inference | No | Interruption causes downtime |
| Hyperparameter search | Yes | Experiments are independent |
| Interactive notebooks | No | Losing GPU disrupts developer workflow |

### Strategy 2: Right-Sizing GPUs

The most common waste: using an H100 when a T4 would suffice.

**Right-sizing decision tree:**
```
What is your workload?
│
├── Inference only?
│   ├── Model < 10 GB VRAM? ──► T4 ($0.53/hr)
│   ├── Model < 20 GB VRAM? ──► A10G ($1.01/hr)
│   └── Model < 40 GB VRAM? ──► A100 40GB ($4.10/hr per GPU)
│
├── Fine-tuning?
│   ├── Model < 7B params? ──► A10G or L40S
│   ├── Model < 13B params? ──► A100 80GB
│   └── Model > 13B params? ──► Multi-GPU A100 or H100
│
└── Pre-training?
    ├── Need maximum throughput? ──► H100 (best FLOPS/$)
    └── Budget-constrained? ──► A100 80GB (mature, cheaper)
```

**Using monitoring data for right-sizing:**
```promql
# Average GPU utilization over the past 7 days
avg_over_time(DCGM_FI_DEV_GPU_UTIL[7d])

# Peak memory usage over the past 7 days
max_over_time(DCGM_FI_DEV_FB_USED[7d])

# If avg utilization < 30% and peak memory < 50% of total:
# The GPU is oversized -- consider downgrading
```

### Strategy 3: Idle GPU Detection and Scale-Down

```yaml
# Cluster Autoscaler configuration for GPU node pools
apiVersion: v1
kind: ConfigMap
metadata:
  name: cluster-autoscaler-config
data:
  # Scale down GPU nodes after 10 minutes of idleness
  scale-down-delay-after-add: "10m"
  scale-down-unneeded-time: "10m"
  scale-down-utilization-threshold: "0.3"

  # GPU-specific: never scale below 0 (allow full scale-down)
  # but scale up aggressively for queued GPU pods
  max-graceful-termination-sec: "600"   # 10 min for checkpoint
```

**Custom idle GPU detector:**
```python
# Alert if GPU is allocated but < 10% utilized for > 30 minutes
ALERT_QUERY = """
  DCGM_FI_DEV_GPU_UTIL < 10
  and on(pod)
  kube_pod_status_phase{phase="Running"}
"""
```

### Strategy 4: Scheduling Optimization

**Bin-packing:** Schedule workloads to fill GPUs completely before allocating new nodes.

```yaml
# Kubernetes scheduler profile for GPU bin-packing
apiVersion: kubescheduler.config.k8s.io/v1
kind: KubeSchedulerConfiguration
profiles:
  - schedulerName: gpu-scheduler
    plugins:
      score:
        enabled:
          - name: NodeResourcesFit
        disabled:
          - name: NodeResourcesBalancedAllocation
    pluginConfig:
      - name: NodeResourcesFit
        args:
          scoringStrategy:
            type: MostAllocated    # Prefer nodes that are already busy
```

**Time-of-day scheduling:** Run batch jobs during off-peak hours.

```yaml
# CronJob for batch inference during cheap hours (nights/weekends)
apiVersion: batch/v1
kind: CronJob
metadata:
  name: batch-inference
spec:
  schedule: "0 22 * * *"    # Run at 10 PM daily
  jobTemplate:
    spec:
      template:
        spec:
          containers:
            - name: inference
              resources:
                limits:
                  nvidia.com/gpu: 4
```

### Strategy 5: Mixed-Precision and Quantization

Reduce GPU memory and compute requirements to use smaller/fewer GPUs:

| Technique | Memory Savings | Speed Impact | Quality Impact |
|---|---|---|---|
| **FP16 mixed precision** | ~50% | 2-3x faster | Minimal |
| **BF16 mixed precision** | ~50% | 2-3x faster | Minimal |
| **INT8 quantization** | ~75% | 3-4x faster (inference) | Small accuracy loss |
| **INT4 quantization** | ~87.5% | 4-8x faster (inference) | Moderate accuracy loss |
| **FP8** (H100 only) | ~75% | 2-4x faster | Minimal |

---

## Hands-On Lab

### Exercise 1: Calculate Your GPU Costs

**Goal:** Audit current GPU spending and identify optimization opportunities.

```bash
# Query GPU utilization over the past week
curl -s "http://prometheus:9090/api/v1/query?query=avg_over_time(DCGM_FI_DEV_GPU_UTIL[7d])"

# Calculate cost waste
# If average utilization is 40% on a $32.77/hr A100 instance:
# Effective cost per useful GPU-hour = $32.77 / 0.40 = $81.93
# vs. optimal at 85% utilization: $32.77 / 0.85 = $38.55
# Waste: $81.93 - $38.55 = $43.38/hr per instance
```

### Exercise 2: Implement Spot Instance Checkpointing

**Goal:** Make training resilient to spot instance preemption.

```python
# Add to your training loop (see src/training/distributed_training.py)
import signal

def handle_preemption(signum, frame):
    """Save checkpoint on SIGTERM (spot preemption signal)."""
    print("Preemption detected! Saving checkpoint...")
    save_checkpoint(model, optimizer, epoch, step)
    sys.exit(0)

signal.signal(signal.SIGTERM, handle_preemption)
```

### Exercise 3: Set Up Idle GPU Alerts

**Goal:** Create Prometheus alerts for underutilized GPUs.

```bash
# Apply alerting rules
kubectl apply -f - <<EOF
apiVersion: monitoring.coreos.com/v1
kind: PrometheusRule
metadata:
  name: gpu-cost-alerts
spec:
  groups:
    - name: gpu-cost
      rules:
        - alert: GPUUnderutilized
          expr: avg_over_time(DCGM_FI_DEV_GPU_UTIL[30m]) < 20
          for: 30m
          labels:
            severity: warning
          annotations:
            summary: "GPU {{ \$labels.gpu }} averaging {{ \$value }}% utilization. Consider right-sizing or time-slicing."
EOF
```

---

## Key Terminology

| Term | Definition |
|---|---|
| **Spot Instance** | Cloud VM available at steep discount but can be reclaimed with short notice |
| **Right-Sizing** | Matching GPU type and count to actual workload requirements |
| **Bin-Packing** | Scheduling strategy that fills existing nodes before provisioning new ones |
| **Idle Detection** | Monitoring for allocated-but-unused GPU resources |
| **Quantization** | Reducing model precision (FP32 -> INT8) to use less memory and compute |
| **Cluster Autoscaler** | Kubernetes component that adds/removes nodes based on pending pod demand |

---

## Common Mistakes

| Mistake | Symptom | Fix |
|---|---|---|
| Using on-demand for fault-tolerant workloads | 3-5x higher cost | Switch to spot with checkpointing |
| H100 for inference of small models | $98/hr for a $0.53/hr workload | Right-size to T4 or A10G |
| No idle detection | Paying for GPUs running nothing | Alert on < 10% utilization for > 15 minutes |
| Not using mixed precision | 2x more memory, 2x slower | Enable AMP (automatic mixed precision) |
| Manual scaling | Over-provisioning to handle peak | Use Cluster Autoscaler with GPU node pools |

---

## Self-Check Questions

1. Your training job costs $16,000/week on on-demand H100s. How much could you save with spot instances? What risk does this introduce?
2. A model uses 12 GB VRAM for inference. Which GPU offers the best cost efficiency?
3. Your Prometheus dashboard shows average GPU utilization of 25% across 8 A100 GPUs. What are three actions you would take?
4. Why is checkpointing essential for spot instance training? How frequently should you checkpoint?
5. When is it actually cheaper to use on-demand H100s instead of spot A100s? (Hint: think about training time.)

---

## You Know You Have Completed This Module When...

- [ ] You can compare GPU pricing across cloud providers
- [ ] You have implemented spot instance support with checkpointing
- [ ] You can right-size GPUs based on utilization and memory data
- [ ] Idle GPU alerts are configured and tested
- [ ] You can calculate cost-per-training-run and identify savings opportunities
- [ ] Self-check questions answered confidently

---

## Troubleshooting

### Common Issues

**Issue: Spot instance preempted mid-training, no checkpoint**
- Implement SIGTERM handler for graceful checkpoint save
- Set checkpoint interval to every 15-30 minutes
- Use cloud-specific preemption notices (AWS: 2-minute warning, GCP: 30-second warning)

**Issue: Cluster Autoscaler not scaling down GPU nodes**
```bash
# Check autoscaler status
kubectl get configmap cluster-autoscaler-status -n kube-system -o yaml

# Common causes:
# - Pods without a controller (standalone pods prevent scale-down)
# - PodDisruptionBudget blocking eviction
# - Local storage (emptyDir) preventing eviction
```

**Issue: Cannot determine if GPU is right-sized**
- Collect at least 7 days of utilization data
- Check both average and P99 utilization
- Memory utilization is the hard constraint; compute utilization shows efficiency

---

**Next: [Module 09 - Multi-Node GPU Clusters](../09-cost-optimization/)**
