# Module 09: Multi-Node GPU Clusters (NCCL, InfiniBand, Network Topology)

| | |
|---|---|
| **Time** | 3-5 hours |
| **Difficulty** | Advanced |
| **Prerequisites** | Module 08 completed |

---

## Learning Objectives

By the end of this module, you will be able to:

- Design multi-node GPU cluster topologies for distributed training
- Configure NCCL for optimal performance across nodes
- Understand InfiniBand vs. RoCE vs. TCP networking for GPU clusters
- Implement topology-aware pod scheduling in Kubernetes
- Diagnose and resolve multi-node communication bottlenecks
- Plan GPU cluster networking for different scale requirements

---

## Concepts

### Why Multi-Node GPU Clusters?

Single-node GPU servers max out at 8 GPUs (DGX A100/H100). Training frontier models requires hundreds or thousands of GPUs across multiple nodes.

```
Scale requirements:
  GPT-3 (175B params):     ~1,000 A100 GPUs, ~34 days
  LLaMA 2 (70B params):    ~2,048 A100 GPUs, ~21 days
  GPT-4 (est. >1T params): ~25,000 A100 GPUs, ~90 days

Hardware:
  1 DGX A100 node = 8x A100 + 8x NVLink + 8x 200Gb InfiniBand
  128 nodes = 1,024 GPUs
  Interconnect bandwidth between nodes is the critical bottleneck
```

### GPU Cluster Network Topology

```
                    ┌─────────────────────────────┐
                    │       Spine Switches          │
                    │  (InfiniBand / RoCE fabric)   │
                    └──┬──────┬──────┬──────┬──────┘
                       │      │      │      │
              ┌────────▼──┐ ┌─▼──────┐ ┌───▼─────┐ ┌──▼───────┐
              │Leaf SW 0  │ │Leaf 1  │ │Leaf 2   │ │Leaf 3    │
              └──┬──┬──┬──┘ └──┬──┬──┘ └──┬──┬──┘ └──┬──┬──┘
                 │  │  │       │  │       │  │       │  │
            ┌────▼┐┌▼──▼┐ ┌───▼┐┌▼──▼┐ ┌──▼┐┌▼──▼┐ ┌─▼─┐┌▼──▼┐
            │Node││Node│ │Node││Node│ │Node││Node│ │Node││Node│
            │ 0  ││ 1  │ │ 2  ││ 3  │ │ 4  ││ 5  │ │ 6  ││ 7  │
            │8GPU││8GPU│ │8GPU││8GPU│ │8GPU││8GPU│ │8GPU││8GPU│
            └────┘└────┘ └────┘└────┘ └────┘└────┘ └────┘└────┘

            └─── Rail 0 ───┘  └─── Rail 1 ───┘  ... (fat-tree topology)
```

**Inside a single node (DGX A100):**
```
GPU 0 ──NVSwitch──► GPU 1 ──NVSwitch──► GPU 2 ──NVSwitch──► GPU 3
  │                   │                   │                   │
  ▼ NVSwitch          ▼                   ▼                   ▼
GPU 4 ──NVSwitch──► GPU 5 ──NVSwitch──► GPU 6 ──NVSwitch──► GPU 7
  │                   │                   │                   │
  ▼ HCA               ▼ HCA               ▼ HCA               ▼ HCA
  IB Port 0           IB Port 1           IB Port 2           IB Port 3

Each NVLink: 300 GB/s (bidirectional)
Each IB port: 200 Gb/s = 25 GB/s
Intra-node: 12x faster than inter-node
```

### Networking Technologies

| Technology | Bandwidth | Latency | Use Case |
|---|---|---|---|
| **NVLink 4 (intra-node)** | 900 GB/s total | <1 us | GPU-to-GPU within a node |
| **NVSwitch (intra-node)** | All-to-all NVLink | <1 us | Full mesh within DGX |
| **InfiniBand HDR** | 200 Gb/s per port | 0.6 us | Inter-node GPU communication |
| **InfiniBand NDR** | 400 Gb/s per port | 0.5 us | Latest DGX H100 clusters |
| **RoCE v2** | 100-400 Gb/s | 1-2 us | Ethernet-based RDMA (cheaper) |
| **TCP/IP** | 10-100 Gb/s | 50-100 us | Fallback, not recommended for training |

### NCCL Configuration for Multi-Node

NCCL (NVIDIA Collective Communications Library) handles all GPU-to-GPU communication. Proper configuration is critical for multi-node scaling.

**Essential NCCL environment variables:**

| Variable | Description | Recommended Value |
|---|---|---|
| `NCCL_DEBUG` | Logging verbosity | `INFO` (debug) / `WARN` (prod) |
| `NCCL_SOCKET_IFNAME` | Network interface | `ib0` (InfiniBand) or `eth0` (Ethernet) |
| `NCCL_IB_DISABLE` | Disable InfiniBand | `0` (use IB when available) |
| `NCCL_IB_HCA` | InfiniBand HCA devices | `mlx5_0,mlx5_1,...` |
| `NCCL_IB_GID_INDEX` | GID index for RoCE | `3` (RoCEv2 default) |
| `NCCL_NET_GDR_LEVEL` | GPUDirect RDMA level | `5` (full support) |
| `NCCL_P2P_LEVEL` | Peer-to-peer NVLink | `NVL` |
| `NCCL_ALGO` | Collective algorithm | `Ring` or `Tree` |
| `NCCL_CROSS_NIC` | Allow cross-NIC communication | `1` |
| `NCCL_BUFFSIZE` | Communication buffer size | `4194304` (4 MB) |

**NCCL algorithm selection:**
```
Ring All-Reduce:
  GPU0 → GPU1 → GPU2 → GPU3 → GPU0
  Best for: small messages, low GPU count
  Latency: 2(N-1) steps

Tree All-Reduce:
       GPU0
      /    \
   GPU1    GPU2
   /
  GPU3
  Best for: large messages, high GPU count
  Latency: 2*log2(N) steps
```

### GPUDirect Technologies

| Technology | Description | Benefit |
|---|---|---|
| **GPUDirect P2P** | Direct GPU-to-GPU transfer over NVLink/PCIe | Skip CPU/system memory |
| **GPUDirect RDMA** | Direct GPU-to-network transfer | Skip CPU for inter-node comms |
| **GPUDirect Storage** | Direct GPU-to-NVMe transfer | Fast data loading from SSD |

```
Without GPUDirect RDMA:
  GPU → PCIe → CPU Memory → NIC → Network → NIC → CPU Memory → PCIe → GPU
  Latency: ~50 us, Bandwidth: limited by CPU memory copy

With GPUDirect RDMA:
  GPU → NIC → Network → NIC → GPU
  Latency: ~2 us, Bandwidth: full IB/RoCE line rate
```

### Topology-Aware Scheduling in Kubernetes

For multi-node training, co-locate GPUs that share the fastest interconnect:

```yaml
# Pod affinity to co-locate training pods on same rack
apiVersion: v1
kind: Pod
metadata:
  name: training-worker
  labels:
    training-job: llm-pretraining
spec:
  affinity:
    podAffinity:
      requiredDuringSchedulingIgnoredDuringExecution:
        - labelSelector:
            matchLabels:
              training-job: llm-pretraining
          topologyKey: topology.kubernetes.io/zone
    podAntiAffinity:
      preferredDuringSchedulingIgnoredDuringExecution:
        - weight: 100
          podAffinityTerm:
            labelSelector:
              matchLabels:
                training-job: llm-pretraining
            topologyKey: kubernetes.io/hostname  # Different nodes
  containers:
    - name: trainer
      resources:
        limits:
          nvidia.com/gpu: 8    # All GPUs on this node
```

### Multi-Node Training with Kubeflow MPI Operator

```yaml
apiVersion: kubeflow.org/v2beta1
kind: MPIJob
metadata:
  name: multi-node-training
spec:
  slotsPerWorker: 8    # 8 GPUs per worker
  runPolicy:
    cleanPodPolicy: Running
  mpiReplicaSpecs:
    Launcher:
      replicas: 1
      template:
        spec:
          containers:
            - name: launcher
              image: nvcr.io/nvidia/pytorch:23.12-py3
              command:
                - mpirun
                - --allow-run-as-root
                - -np 16         # Total GPUs (2 nodes x 8)
                - -npernode 8
                - python
                - /workspace/train.py
    Worker:
      replicas: 2              # 2 nodes
      template:
        spec:
          containers:
            - name: worker
              image: nvcr.io/nvidia/pytorch:23.12-py3
              resources:
                limits:
                  nvidia.com/gpu: 8
                  rdma/rdma_shared_device_a: 1   # InfiniBand RDMA
```

---

## Hands-On Lab

### Prerequisites Check

```bash
# Check node count and GPU capacity
kubectl get nodes -o custom-columns=NAME:.metadata.name,GPUs:.status.allocatable.nvidia\\.com/gpu

# Check network interfaces on GPU nodes
kubectl exec -it <gpu-pod> -- ip addr | grep -E "^[0-9]+:|inet "

# Check NCCL availability
python -c "import torch.distributed as dist; print('NCCL:', dist.is_nccl_available())"
```

### Exercise 1: Profile NCCL Communication

**Goal:** Measure GPU-to-GPU bandwidth and latency.

```bash
# Run NCCL tests (all-reduce benchmark)
kubectl apply -f - <<EOF
apiVersion: batch/v1
kind: Job
metadata:
  name: nccl-test
spec:
  template:
    spec:
      restartPolicy: Never
      containers:
        - name: nccl-test
          image: nvcr.io/nvidia/pytorch:23.12-py3
          command:
            - bash
            - -c
            - |
              apt-get update && apt-get install -y nccl-tests
              /usr/bin/all_reduce_perf -b 8 -e 128M -f 2 -g 4
          resources:
            limits:
              nvidia.com/gpu: 4
          env:
            - name: NCCL_DEBUG
              value: "INFO"
EOF

kubectl logs -f job/nccl-test
# Look for: busbw (bus bandwidth) -- this is the effective throughput
```

### Exercise 2: Configure Multi-Node Communication

**Goal:** Set up NCCL for inter-node training.

```bash
# Apply NCCL configuration as environment variables
kubectl apply -f - <<EOF
apiVersion: v1
kind: ConfigMap
metadata:
  name: nccl-config
data:
  NCCL_DEBUG: "INFO"
  NCCL_SOCKET_IFNAME: "eth0"
  NCCL_IB_DISABLE: "0"
  NCCL_NET_GDR_LEVEL: "5"
  NCCL_P2P_LEVEL: "NVL"
EOF
```

### Exercise 3: Benchmark Scaling Across Nodes

**Goal:** Measure training throughput scaling from 1 to N nodes.

```bash
# Single node (8 GPUs)
torchrun --nnodes=1 --nproc_per_node=8 train.py

# Two nodes (16 GPUs)
# Node 0:
torchrun --nnodes=2 --node_rank=0 --nproc_per_node=8 \
  --master_addr=10.0.0.1 --master_port=29500 train.py
# Node 1:
torchrun --nnodes=2 --node_rank=1 --nproc_per_node=8 \
  --master_addr=10.0.0.1 --master_port=29500 train.py

# Expected scaling efficiency:
# 1 node (8 GPUs):   100%
# 2 nodes (16 GPUs):  90-95% (NVLink intra-node, IB inter-node)
# 4 nodes (32 GPUs):  85-90%
# 8 nodes (64 GPUs):  80-88%
```

---

## Key Terminology

| Term | Definition |
|---|---|
| **NCCL** | NVIDIA Collective Communications Library -- optimized GPU-to-GPU communication |
| **InfiniBand (IB)** | High-bandwidth, low-latency network fabric for HPC and GPU clusters |
| **RoCE** | RDMA over Converged Ethernet -- InfiniBand-like RDMA on Ethernet hardware |
| **RDMA** | Remote Direct Memory Access -- transfers data without CPU involvement |
| **GPUDirect** | NVIDIA technology for direct data paths between GPUs, NICs, and storage |
| **NVSwitch** | NVIDIA switch chip enabling all-to-all NVLink within a node |
| **Fat-Tree Topology** | Network topology where bandwidth increases toward the spine (non-blocking) |
| **All-Reduce** | Collective operation that aggregates (sums/averages) data across all participants |
| **Bus Bandwidth** | Effective collective communication throughput (excludes protocol overhead) |

---

## Common Mistakes

| Mistake | Symptom | Fix |
|---|---|---|
| Using TCP instead of InfiniBand | 10-50x slower inter-node communication | Configure NCCL to use IB: `NCCL_IB_DISABLE=0` |
| Wrong NCCL_SOCKET_IFNAME | Timeout during init_process_group | Set to the correct interface: `ib0` or `eth0` |
| Ignoring network topology | Non-uniform communication latency | Use pod affinity to co-locate on same rack/switch |
| Not enabling GPUDirect RDMA | Extra CPU-memory hops for every transfer | Install MOFED driver, set `NCCL_NET_GDR_LEVEL=5` |
| Mixed InfiniBand and Ethernet nodes | Degraded to lowest common denominator | Ensure all GPU nodes use the same network fabric |

---

## Self-Check Questions

1. Why is NVLink ~12x faster than InfiniBand for GPU communication? When does IB matter?
2. What is GPUDirect RDMA and how does it reduce latency for multi-node training?
3. A training job scales 95% efficiently from 8 to 16 GPUs but only 70% from 16 to 32. What is the likely bottleneck?
4. Explain the difference between Ring and Tree all-reduce algorithms. When is each better?
5. How would you design a Kubernetes scheduling policy to ensure multi-node training pods are co-located on the same InfiniBand fabric?

---

## You Know You Have Completed This Module When...

- [ ] You can explain GPU cluster network topology (NVLink, NVSwitch, InfiniBand, spine-leaf)
- [ ] You can configure NCCL environment variables for optimal multi-node performance
- [ ] You understand GPUDirect technologies (P2P, RDMA, Storage)
- [ ] You can run NCCL benchmarks and interpret bus bandwidth results
- [ ] You can design topology-aware scheduling for multi-node training on Kubernetes
- [ ] Self-check questions answered confidently

---

## Troubleshooting

### Common Issues

**Issue: NCCL timeout during multi-node init**
```bash
# Check connectivity between nodes
ping <other-node-ip>

# Verify NCCL can reach the master
export NCCL_DEBUG=INFO
# Look for: "NCCL INFO Channel 00 : 0[0] -> 1[1]" type messages

# Common fixes:
export NCCL_SOCKET_IFNAME=eth0   # Or ib0 for InfiniBand
export NCCL_IB_DISABLE=1         # Temporarily disable IB to test TCP
```

**Issue: Low inter-node bandwidth in NCCL tests**
```bash
# Check IB link speed
ibstat | grep Rate
# Should show: 200 Gb/sec (HDR) or 400 Gb/sec (NDR)

# Verify GPUDirect is working
nvidia-smi topo -m
# Look for "NV#" connections (NVLink) and "SYS" for cross-node
```

**Issue: Training hangs at gradient synchronization**
- One node may have a hardware issue (bad GPU, bad NIC)
- Check NCCL_DEBUG=INFO logs for which rank is stuck
- Verify all nodes have the same PyTorch + NCCL versions

---

**Next: [Module 10 - Production GPU Infrastructure](../10-production-gpu-cluster/)**
