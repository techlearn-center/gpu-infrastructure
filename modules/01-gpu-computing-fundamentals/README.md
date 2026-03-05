# Module 01: GPU Computing Fundamentals

| | |
|---|---|
| **Time** | 3-5 hours |
| **Difficulty** | Beginner |
| **Prerequisites** | Docker installed, basic terminal knowledge |

---

## Learning Objectives

By the end of this module, you will be able to:

- Explain the difference between CPU and GPU architectures and why GPUs dominate parallel workloads
- Identify CUDA cores, Tensor Cores, and the GPU memory hierarchy
- Map NVIDIA GPU product lines (T4, V100, A100, H100, L40S) to appropriate workloads
- Read and interpret `nvidia-smi` output to assess GPU health and utilization
- Understand how GPU computing concepts translate to Kubernetes scheduling decisions

---

## Concepts

### CPU vs GPU Architecture

A CPU is designed for **low-latency, serial execution**. A modern server CPU has 32-128 cores, each capable of complex branching, out-of-order execution, and large caches. It excels at tasks where individual operations are complex but few run simultaneously.

A GPU is designed for **high-throughput, parallel execution**. An NVIDIA A100 has **6,912 CUDA cores** grouped into Streaming Multiprocessors (SMs). Each core is simpler than a CPU core, but thousands run in lockstep to process massive data arrays.

```
CPU (Serial)                          GPU (Parallel)
+---------+                           +---+---+---+---+---+---+
| Core 0  | --> Complex task          | 0 | 1 | 2 | 3 | 4 | 5 |  ... x 6912
+---------+                           +---+---+---+---+---+---+
| Core 1  | --> Complex task          All cores execute same instruction
+---------+                           on different data (SIMT model)
     ...
| Core 63 | --> Complex task
+---------+
```

**When to use GPUs:**
- Matrix multiplications (neural network training and inference)
- Batch image/video processing
- Monte Carlo simulations and scientific computing
- Any workload with massive data parallelism

**When CPUs are better:**
- Sequential business logic
- I/O-bound tasks (web servers, databases)
- Workloads with heavy branching

### CUDA Cores and Tensor Cores

| Component | Purpose | Example |
|---|---|---|
| **CUDA Cores** | General-purpose parallel compute (FP32, FP64, INT) | Matrix math, simulations |
| **Tensor Cores** | Specialized matrix-multiply-accumulate (FP16, BF16, TF32, INT8, FP8) | Deep learning training and inference |
| **RT Cores** | Ray tracing acceleration (not relevant for ML) | Graphics rendering |

**Tensor Cores** are the key differentiator for AI/ML workloads. A single A100 Tensor Core performs a 4x4 matrix multiply-accumulate in one clock cycle. With 432 Tensor Cores, the A100 delivers **312 TFLOPS** of FP16 Tensor performance versus 19.5 TFLOPS of FP32 CUDA-core performance -- a 16x advantage for deep learning.

### GPU Memory Hierarchy

Understanding the memory hierarchy is critical for optimizing GPU workloads and sizing Kubernetes resource requests.

```
Level           Size          Latency      Bandwidth
-------         ---------     ---------    -----------
Registers       256 KB/SM     1 cycle      ~19 TB/s
L1 / Shared     192 KB/SM     ~28 cycles   ~19 TB/s
L2 Cache        40 MB (A100)  ~200 cycles  ~5 TB/s
HBM (VRAM)      80 GB (A100)  ~400 cycles  2.0 TB/s (HBM2e)
System (PCIe)   Host RAM      ~10K cycles  64 GB/s (Gen4 x16)
NVLink          Peer GPU      ~600 cycles  600 GB/s (NVLink 4)
```

**Key insight for Kubernetes:** When you set `resources.limits: nvidia.com/gpu: 1`, the pod gets exclusive access to the **entire HBM memory** of that GPU. This is why GPU memory is the most common bottleneck -- a model that needs 45 GB of VRAM cannot share a 40 GB A100 with anything else.

### NVIDIA GPU Product Line for Data Centers

| GPU | CUDA Cores | Tensor Cores | Memory | Use Case | Cloud Instance |
|---|---|---|---|---|---|
| **T4** | 2,560 | 320 (Gen 2) | 16 GB GDDR6 | Inference, light training | AWS g4dn, GCP n1-standard + T4 |
| **V100** | 5,120 | 640 (Gen 1) | 32 GB HBM2 | Training (legacy) | AWS p3, GCP a2 |
| **A10G** | 9,216 | 288 (Gen 3) | 24 GB GDDR6X | Inference, graphics | AWS g5 |
| **A100** | 6,912 | 432 (Gen 3) | 40/80 GB HBM2e | Training + inference | AWS p4d, GCP a2-ultragpu |
| **H100** | 16,896 | 528 (Gen 4) | 80 GB HBM3 | Large model training | AWS p5, GCP a3-highgpu |
| **L40S** | 18,176 | 568 (Gen 4) | 48 GB GDDR6X | Inference, fine-tuning | AWS g6 |

**Choosing the right GPU:**
- **Inference only?** T4 or L40S (lower cost, sufficient memory)
- **Fine-tuning < 7B params?** A10G or L40S
- **Pre-training large models?** A100 80 GB or H100 (need HBM bandwidth)
- **Budget-sensitive?** T4 spot instances for batch inference

### Understanding nvidia-smi

`nvidia-smi` is the primary CLI tool for GPU monitoring. Every GPU infrastructure engineer must read its output fluently.

```
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 545.23.08    Driver Version: 545.23.08    CUDA Version: 12.3                 |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|=========================================+========================+======================|
|   0  NVIDIA A100-SXM4-80GB         On  | 00000000:00:04.0  Off |                    0 |
| N/A   34C    P0              62W / 400W |    1024MiB / 81920MiB  |     15%      Default |
+-----------------------------------------+------------------------+----------------------+
```

| Field | What it tells you | Why it matters |
|---|---|---|
| **GPU-Util** | % of time at least one SM kernel is running | Low util = GPU is idle (wasting money) |
| **Memory-Usage** | HBM allocation vs. capacity | Determines if another workload can fit |
| **Temp** | Junction temperature in Celsius | > 83 C triggers thermal throttling |
| **Pwr:Usage/Cap** | Current power draw vs. limit | High power = heavy compute |
| **Perf** | Performance state (P0=max, P8=idle) | P8 means GPU is underutilized |
| **ECC** | Error-correcting code errors | Non-zero = hardware degradation |

```bash
# Useful nvidia-smi commands for infrastructure engineers
nvidia-smi                                     # Full dashboard
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv
nvidia-smi topo -m                             # Show NVLink / PCIe topology
nvidia-smi mig -lgip                           # List MIG-capable profiles
nvidia-smi dmon -s pucvmet -d 1                # Continuous monitoring (1s interval)
```

---

## Hands-On Lab

### Prerequisites Check

```bash
# Check Docker is running
docker --version
docker compose version

# Check you have the project cloned
ls modules/01-gpu-computing-fundamentals/

# (Optional) If you have an NVIDIA GPU:
nvidia-smi
```

### Exercise 1: Explore GPU Architecture with nvidia-smi

**Goal:** Learn to read GPU information and identify key parameters.

**Step 1:** If you have an NVIDIA GPU, run the following and annotate each field:
```bash
nvidia-smi

# Export as structured data
nvidia-smi --query-gpu=index,name,driver_version,memory.total,memory.used,memory.free,utilization.gpu,utilization.memory,temperature.gpu,power.draw,power.limit --format=csv,noheader,nounits
```

**Step 2:** If you do NOT have a GPU, study this sample output and answer the questions below:
```
GPU 0: NVIDIA A100-SXM4-80GB
  Memory: 2048 MiB / 81920 MiB (2.5%)
  Utilization: 45%
  Temperature: 52 C
  Power: 185W / 400W
```

**Questions:**
1. How much free VRAM is available?
2. Could you run a second workload requiring 40 GB VRAM alongside this one?
3. Is the GPU thermally throttling?
4. What does 45% utilization suggest about the workload?

### Exercise 2: GPU Topology and Interconnects

**Goal:** Understand how GPUs are connected (PCIe vs. NVLink) and why it matters for multi-GPU training.

```bash
# Show GPU-to-GPU connectivity
nvidia-smi topo -m

# Expected output shows NVLink (NV#) or PCIe (PIX/PHB/SYS) between GPUs
# NVLink: 600 GB/s (bidirectional, 4th gen)
# PCIe Gen4 x16: 64 GB/s (bidirectional)
```

**Key takeaway:** Multi-GPU training performance depends heavily on interconnect bandwidth. NVLink is ~10x faster than PCIe for gradient synchronization. In Kubernetes, use topology-aware scheduling to keep related GPUs on the same NVLink domain.

### Exercise 3: Memory Estimation for Common Models

**Goal:** Practice estimating GPU memory requirements for real workloads.

**Memory estimation formula (training with AdamW):**
```
Total VRAM = Model Parameters x Bytes per Param x Multiplier

Where Multiplier accounts for:
  - Model weights:          1x  (FP16 = 2 bytes/param)
  - Gradients:              1x
  - Optimizer states:       2x  (AdamW: mean + variance)
  - Activations:            ~1x (depends on batch size)
  --------------------------------------------------
  Total:                    ~5x params (FP16)

Example: LLaMA 7B
  7 billion params x 2 bytes x 5 = 70 GB
  Fits on: A100 80 GB (tight) or 2x A100 40 GB with model parallelism
```

**Exercise:** Calculate VRAM requirements for:
1. ResNet-50 (25.6M parameters) -- training with FP32
2. BERT-base (110M parameters) -- training with FP16
3. GPT-2 (1.5B parameters) -- inference only with FP16

---

## Key Terminology

| Term | Definition |
|---|---|
| **CUDA Core** | Basic parallel processing unit on NVIDIA GPUs, executes one floating-point or integer operation per clock |
| **Tensor Core** | Specialized hardware for matrix multiply-accumulate, 8-16x faster than CUDA cores for ML workloads |
| **HBM (High Bandwidth Memory)** | Stacked DRAM used as GPU VRAM, provides 2-3 TB/s bandwidth |
| **SM (Streaming Multiprocessor)** | GPU building block containing CUDA cores, Tensor Cores, shared memory, and schedulers |
| **VRAM** | Video RAM -- the GPU's local memory (HBM2e on A100, HBM3 on H100) |
| **NVLink** | NVIDIA's high-bandwidth GPU-to-GPU interconnect (600 GB/s on 4th gen) |
| **PCIe** | Standard CPU-to-GPU and GPU-to-GPU interconnect (64 GB/s on Gen4 x16) |
| **FLOPS** | Floating-point operations per second -- primary measure of GPU compute throughput |
| **Thermal Throttling** | Automatic clock reduction when GPU temperature exceeds safe limits (~83 C) |

---

## Common Mistakes

| Mistake | Symptom | Fix |
|---|---|---|
| Requesting more VRAM than available | Pod stuck in `Pending` state | Check GPU memory with `nvidia-smi` and size workloads accordingly |
| Ignoring NVLink topology | Multi-GPU training is unexpectedly slow | Use `nvidia-smi topo -m` and schedule pods on NVLink-connected GPUs |
| Using FP32 when FP16/BF16 works | 2x memory waste, slower training | Enable mixed-precision with `torch.cuda.amp` |
| Not monitoring GPU utilization | Paying for idle GPUs | Set up DCGM exporter + Prometheus (Module 08) |
| Assuming all GPUs are equal | Performance varies wildly | Match GPU type to workload: T4 for inference, A100/H100 for training |

---

## Self-Check Questions

1. Why do GPUs have thousands of cores while CPUs have only dozens? What trade-off does this represent?
2. What is the difference between CUDA cores and Tensor Cores? When does each matter?
3. An A100-80GB shows 45 GB memory used and 35% GPU utilization. Can you add another 30 GB workload?
4. Why is NVLink important for distributed training but irrelevant for single-GPU inference?
5. You need to train a 13B parameter model with AdamW in FP16. Estimate the VRAM needed. Which GPU(s) would you choose?

---

## You Know You Have Completed This Module When...

- [ ] You can explain CPU vs. GPU architecture trade-offs without notes
- [ ] You can read nvidia-smi output and identify utilization, memory, temperature, and ECC status
- [ ] You can estimate VRAM requirements for training and inference workloads
- [ ] You understand the memory hierarchy (registers, L1, L2, HBM, PCIe, NVLink) and why it matters
- [ ] You can map workloads to the right NVIDIA GPU (T4, A10G, A100, H100)
- [ ] Self-check questions answered confidently

---

## Troubleshooting

### Common Issues

**Issue: `nvidia-smi` command not found**
- Install NVIDIA drivers: [NVIDIA Driver Downloads](https://www.nvidia.com/Download/index.aspx)
- On cloud VMs, drivers are usually pre-installed. Check with `lsmod | grep nvidia`.

**Issue: CUDA version mismatch**
- The CUDA version shown by `nvidia-smi` is the maximum supported version.
- Your PyTorch/TensorFlow CUDA version must be equal to or lower than the driver's CUDA version.

**Issue: GPU not detected in Docker container**
```bash
# Install NVIDIA Container Toolkit
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker

# Test GPU access in container
docker run --rm --gpus all nvidia/cuda:12.3.1-base-ubuntu22.04 nvidia-smi
```

---

**Next: [Module 02 - NVIDIA GPU Operator on Kubernetes](../02-nvidia-gpu-operator/)**
