# Module 06: Distributed Training (DDP, FSDP, DeepSpeed)

| | |
|---|---|
| **Time** | 3-5 hours |
| **Difficulty** | Intermediate-Advanced |
| **Prerequisites** | Module 05 completed, PyTorch basics |

---

## Learning Objectives

By the end of this module, you will be able to:

- Explain why distributed training is necessary for large models
- Implement PyTorch Distributed Data Parallel (DDP) across multiple GPUs
- Understand data parallelism vs. model parallelism vs. pipeline parallelism
- Configure NCCL for efficient GPU-to-GPU communication
- Compare DDP, FSDP, and DeepSpeed for different model sizes
- Launch distributed training jobs on Kubernetes with torchrun

---

## Concepts

### Why Distributed Training?

A single A100-80GB can hold models up to ~13B parameters for training (with FP16 + AdamW). For larger models or faster training, you must distribute the work across multiple GPUs.

**Scaling math:**
```
Single GPU:
  LLaMA 7B, batch_size=8, A100-80GB
  Throughput: ~2,400 tokens/sec
  Time for 1 epoch (100B tokens): ~11,500 hours (480 days)

8x A100 DDP:
  Same model, effective batch_size=64
  Throughput: ~18,000 tokens/sec (7.5x scaling, ~94% efficiency)
  Time for 1 epoch: ~1,540 hours (64 days)

64x A100 (8 nodes x 8 GPUs):
  Throughput: ~130,000 tokens/sec (85% scaling)
  Time for 1 epoch: ~213 hours (9 days)
```

### Parallelism Strategies

```
┌─────────────────────────────────────────────────────────────┐
│                    PARALLELISM STRATEGIES                    │
├──────────────────┬──────────────────┬───────────────────────┤
│  Data Parallel   │ Model Parallel   │ Pipeline Parallel     │
│  (DDP)           │ (Tensor)         │                       │
│                  │                  │                       │
│  Each GPU has    │  Each GPU has    │  Each GPU has         │
│  FULL model      │  PART of each    │  DIFFERENT layers     │
│  DIFFERENT data  │  layer           │  SAME data flows      │
│                  │                  │  through stages       │
│  ┌───┐ ┌───┐    │  ┌─┐┌─┐         │  ┌───┐               │
│  │GPU│ │GPU│    │  │A││B│  = 1    │  │ L1│ GPU 0         │
│  │ 0 │ │ 1 │    │  │ ││ │  layer  │  ├───┤               │
│  │   │ │   │    │  └─┘└─┘         │  │ L2│ GPU 1         │
│  │ALL│ │ALL│    │                  │  ├───┤               │
│  │lay│ │lay│    │  Needed for      │  │ L3│ GPU 2         │
│  │ers│ │ers│    │  layers too      │  ├───┤               │
│  └───┘ └───┘    │  large for 1 GPU │  │ L4│ GPU 3         │
│                  │                  │  └───┘               │
│  Scale: data     │  Scale: model    │  Scale: model depth  │
│  Best: < 13B     │  Best: > 100B    │  Best: very deep     │
└──────────────────┴──────────────────┴───────────────────────┘
```

### PyTorch DDP (Distributed Data Parallel)

DDP is the **most common** distributed training approach. Every GPU holds a complete copy of the model and processes a different mini-batch. Gradients are synchronized via all-reduce after each backward pass.

```
Forward Pass:
  GPU 0: model(batch_0) --> loss_0
  GPU 1: model(batch_1) --> loss_1
  GPU 2: model(batch_2) --> loss_2
  GPU 3: model(batch_3) --> loss_3

Backward Pass + All-Reduce:
  GPU 0: loss_0.backward() --> grads_0 ─┐
  GPU 1: loss_1.backward() --> grads_1 ─┤── NCCL All-Reduce ──> avg_grads
  GPU 2: loss_2.backward() --> grads_2 ─┤   (all GPUs get same result)
  GPU 3: loss_3.backward() --> grads_3 ─┘

Optimizer Step:
  All GPUs: optimizer.step(avg_grads)
  All GPUs now have identical weights
```

**Key DDP code pattern:**
```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# 1. Initialize process group (NCCL backend for GPUs)
dist.init_process_group(backend="nccl")

# 2. Set local GPU
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)

# 3. Wrap model in DDP
model = MyModel().to(local_rank)
model = DDP(model, device_ids=[local_rank])

# 4. Use DistributedSampler for data sharding
sampler = DistributedSampler(dataset, shuffle=True)
loader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)

# 5. Training loop (standard, DDP handles gradient sync)
for epoch in range(epochs):
    sampler.set_epoch(epoch)  # Critical for proper shuffling
    for batch in loader:
        loss = model(batch)
        loss.backward()       # Gradients synced automatically via NCCL
        optimizer.step()
        optimizer.zero_grad()

# 6. Cleanup
dist.destroy_process_group()
```

### FSDP (Fully Sharded Data Parallel)

FSDP shards model parameters, gradients, and optimizer states across GPUs. Each GPU only holds 1/N of the model at rest, then gathers parameters on-demand during forward/backward.

**Memory comparison (7B model, 8 GPUs, FP16):**
| Approach | Per-GPU Memory | Total Memory |
|---|---|---|
| **DDP** | ~70 GB (full model + optimizer) | 560 GB |
| **FSDP** | ~9 GB (1/8 of everything) | 70 GB |

```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

model = FSDP(
    MyModel(),
    auto_wrap_policy=size_based_auto_wrap_policy,
    mixed_precision=MixedPrecision(
        param_dtype=torch.float16,
        reduce_dtype=torch.float16,
        buffer_dtype=torch.float16,
    ),
)
```

**Use FSDP when:**
- Model does not fit in a single GPU's memory with DDP
- You have 13B+ parameter models
- You want to avoid the complexity of DeepSpeed

### DeepSpeed ZeRO

DeepSpeed's ZeRO (Zero Redundancy Optimizer) provides three stages of memory optimization:

| Stage | What is sharded | Memory savings | Communication overhead |
|---|---|---|---|
| **ZeRO-1** | Optimizer states | ~4x | Minimal |
| **ZeRO-2** | + Gradients | ~8x | Low |
| **ZeRO-3** | + Parameters | ~N x (linear with GPUs) | Higher |

```python
# DeepSpeed config (ds_config.json)
{
  "train_batch_size": 256,
  "gradient_accumulation_steps": 4,
  "fp16": {"enabled": true},
  "zero_optimization": {
    "stage": 2,
    "offload_optimizer": {"device": "cpu"},
    "contiguous_gradients": true,
    "overlap_comm": true
  }
}
```

### NCCL Configuration

NCCL (NVIDIA Collective Communications Library) is the backbone of all GPU-to-GPU communication. Proper NCCL tuning is critical for scaling.

| Environment Variable | Purpose | Recommended Value |
|---|---|---|
| `NCCL_DEBUG` | Logging level | `INFO` (debug), `WARN` (prod) |
| `NCCL_SOCKET_IFNAME` | Network interface for communication | `eth0` or `ib0` |
| `NCCL_IB_DISABLE` | Disable InfiniBand | `0` (enable IB if available) |
| `NCCL_P2P_LEVEL` | Peer-to-peer communication level | `NVL` (NVLink) |
| `NCCL_ALGO` | Collective algorithm | `Ring` or `Tree` |
| `NCCL_PROTO` | Protocol | `Simple` or `LL128` |

---

## Hands-On Lab

### Prerequisites Check

```bash
# Verify PyTorch and CUDA
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.cuda.is_available()}')"

# Check GPU count
python -c "import torch; print(f'GPUs: {torch.cuda.device_count()}')"

# Verify NCCL
python -c "import torch.distributed as dist; print('NCCL available:', dist.is_nccl_available())"
```

### Exercise 1: Single-Node DDP Training

**Goal:** Run DDP training across multiple GPUs on one node.

```bash
# Run the distributed training example with 4 GPUs
torchrun --nproc_per_node=4 \
  src/training/distributed_training.py \
  --epochs 5 \
  --batch-size 64 \
  --mixed-precision \
  --log-interval 5
```

**Observe:**
- Each process gets a different `RANK` and `LOCAL_RANK`
- Throughput scales roughly linearly with GPU count
- All processes report the same final loss (gradients are synchronized)

### Exercise 2: Deploy DDP on Kubernetes

**Goal:** Run distributed training as a Kubernetes Job.

```bash
kubectl apply -f - <<EOF
apiVersion: batch/v1
kind: Job
metadata:
  name: ddp-training
spec:
  parallelism: 1
  template:
    spec:
      restartPolicy: Never
      containers:
        - name: trainer
          image: nvcr.io/nvidia/pytorch:23.12-py3
          command:
            - torchrun
            - --nproc_per_node=4
            - /workspace/src/training/distributed_training.py
            - --epochs=3
            - --batch-size=64
            - --mixed-precision
          resources:
            limits:
              nvidia.com/gpu: 4
          env:
            - name: NCCL_DEBUG
              value: "INFO"
          volumeMounts:
            - name: training-code
              mountPath: /workspace/src
      volumes:
        - name: training-code
          configMap:
            name: training-code
EOF
```

### Exercise 3: Benchmark Scaling Efficiency

**Goal:** Measure how throughput scales from 1 to N GPUs.

```bash
# 1 GPU
torchrun --nproc_per_node=1 src/training/distributed_training.py --epochs 2 --batch-size 64

# 2 GPUs
torchrun --nproc_per_node=2 src/training/distributed_training.py --epochs 2 --batch-size 64

# 4 GPUs
torchrun --nproc_per_node=4 src/training/distributed_training.py --epochs 2 --batch-size 64

# Calculate scaling efficiency:
# efficiency = (throughput_N / (throughput_1 * N)) * 100
```

---

## Key Terminology

| Term | Definition |
|---|---|
| **DDP** | Distributed Data Parallel -- each GPU has full model copy, data is sharded |
| **FSDP** | Fully Sharded Data Parallel -- model, gradients, and optimizer are sharded across GPUs |
| **DeepSpeed ZeRO** | Memory optimization library from Microsoft with three sharding stages |
| **NCCL** | NVIDIA Collective Communications Library -- GPU-to-GPU communication primitives |
| **All-Reduce** | Collective operation that sums (or averages) tensors across all GPUs |
| **torchrun** | PyTorch's distributed launch utility that sets RANK, LOCAL_RANK, WORLD_SIZE |
| **DistributedSampler** | Splits a dataset across ranks so each GPU processes unique data |
| **Gradient Accumulation** | Simulating larger batch sizes by accumulating gradients over multiple forward passes |

---

## Common Mistakes

| Mistake | Symptom | Fix |
|---|---|---|
| Forgetting `sampler.set_epoch(epoch)` | Data not properly shuffled across ranks | Always call `set_epoch()` at the start of each epoch |
| Not scaling learning rate with world_size | Convergence issues | Multiply base LR by world_size (linear scaling rule) |
| Using `model.state_dict()` instead of `model.module.state_dict()` | Checkpoint includes DDP wrapper keys | Access `.module` for the underlying model |
| Wrong NCCL_SOCKET_IFNAME | Timeout during init_process_group | Set to your cluster's network interface (check `ip addr`) |
| Not using `pin_memory=True` in DataLoader | 10-30% slower data transfer | Always set `pin_memory=True` for GPU training |

---

## Self-Check Questions

1. What is the difference between data parallelism and model parallelism? When would you use each?
2. In DDP, what happens if one GPU's gradient computation is slower than the others?
3. Why does FSDP use less memory per GPU than DDP? What is the trade-off?
4. What is the linear scaling rule for learning rate, and why is it necessary?
5. You have a 70B parameter model and 8x A100-80GB. Can you use DDP? What should you use instead?

---

## You Know You Have Completed This Module When...

- [ ] You can run DDP training across multiple GPUs with `torchrun`
- [ ] You understand the all-reduce gradient synchronization mechanism
- [ ] You can explain when to use DDP vs. FSDP vs. DeepSpeed
- [ ] You can configure NCCL environment variables for optimal performance
- [ ] You can deploy a distributed training job on Kubernetes
- [ ] Self-check questions answered confidently

---

## Troubleshooting

### Common Issues

**Issue: `RuntimeError: NCCL communicator was aborted`**
```bash
# Usually a timeout issue. Increase timeout:
dist.init_process_group(backend="nccl", timeout=timedelta(minutes=30))

# Or check NCCL networking:
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=eth0
```

**Issue: Processes hang during init_process_group**
```bash
# Verify all processes can reach the master:
# On each node: ping <MASTER_ADDR>
# Check firewall allows MASTER_PORT (default 29500)

# Ensure WORLD_SIZE matches actual number of processes
echo "RANK=$RANK WORLD_SIZE=$WORLD_SIZE MASTER_ADDR=$MASTER_ADDR"
```

**Issue: Training throughput does not scale linearly**
- Check GPU interconnect: `nvidia-smi topo -m` (NVLink >> PCIe)
- Ensure data loading is not the bottleneck: increase `num_workers`
- Profile with `torch.profiler` to find communication overhead

---

**Next: [Module 07 - GPU Monitoring](../07-multi-node-training/)**
