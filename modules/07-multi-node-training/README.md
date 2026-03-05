# Module 07: GPU Monitoring (nvidia-smi, DCGM, Prometheus)

| | |
|---|---|
| **Time** | 3-5 hours |
| **Difficulty** | Advanced |
| **Prerequisites** | Module 06 completed |

---

## Learning Objectives

By the end of this module, you will be able to:

- Monitor GPU utilization, memory, temperature, and power using nvidia-smi and pynvml
- Deploy DCGM Exporter to expose GPU metrics as Prometheus endpoints
- Build Grafana dashboards for GPU cluster observability
- Set up alerting rules for GPU health issues (thermal throttling, ECC errors, low utilization)
- Use custom Python-based GPU monitoring with the pynvml library
- Correlate GPU metrics with training performance (throughput, loss curves)

---

## Concepts

### The GPU Monitoring Stack

```
┌─────────────────────────────────────────────────────────────┐
│                    GPU Monitoring Stack                       │
│                                                              │
│  ┌───────────┐     ┌──────────────┐     ┌───────────────┐  │
│  │ nvidia-smi│     │ DCGM Exporter│     │ Custom pynvml │  │
│  │ (CLI tool)│     │ (DaemonSet)  │     │ (Python app)  │  │
│  └─────┬─────┘     └──────┬───────┘     └──────┬────────┘  │
│        │                   │                     │           │
│        │ manual             │ :9400/metrics       │ :9400     │
│        │                   │                     │           │
│        │              ┌────▼─────────────────────▼────┐     │
│        │              │         Prometheus              │     │
│        │              │    (scrape + store + query)     │     │
│        │              └────────────┬───────────────────┘     │
│        │                           │                         │
│        │              ┌────────────▼───────────────────┐     │
│        │              │          Grafana                │     │
│        │              │   (dashboards + alerts)         │     │
│        │              └────────────────────────────────┘     │
│        │                                                     │
│        ▼                                                     │
│   Interactive debugging     Production observability         │
└─────────────────────────────────────────────────────────────┘
```

### nvidia-smi Deep Dive

Beyond the basic dashboard, nvidia-smi offers powerful monitoring commands:

```bash
# Continuous monitoring (1-second interval)
nvidia-smi dmon -s pucvmet -d 1
# p=power, u=utilization, c=clocks, v=voltage, m=memory, e=ecc, t=temp

# Query specific metrics as CSV
nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw,clocks.sm,clocks.mem --format=csv -l 5

# Per-process GPU usage
nvidia-smi pmon -s um -d 5
# Shows which PID is using how much GPU compute and memory

# GPU topology (critical for multi-GPU training)
nvidia-smi topo -m

# Check for ECC errors (hardware health)
nvidia-smi --query-gpu=ecc.errors.corrected.volatile.total,ecc.errors.uncorrected.volatile.total --format=csv

# MIG device information
nvidia-smi mig -lgip    # List GPU Instance Profiles
nvidia-smi mig -lgi     # List active GPU Instances
```

### DCGM (Data Center GPU Manager)

DCGM is NVIDIA's enterprise GPU monitoring framework. The DCGM Exporter runs as a DaemonSet (deployed by GPU Operator) and exposes metrics in Prometheus format.

**Key DCGM metrics:**

| Metric | Prometheus Name | Description | Alert Threshold |
|---|---|---|---|
| GPU Utilization | `DCGM_FI_DEV_GPU_UTIL` | % of time SMs are active | < 20% = underutilized |
| Memory Utilization | `DCGM_FI_DEV_MEM_COPY_UTIL` | % of memory bandwidth used | > 90% = memory pressure |
| Memory Used | `DCGM_FI_DEV_FB_USED` | Framebuffer memory in MB | > 95% of total = OOM risk |
| Temperature | `DCGM_FI_DEV_GPU_TEMP` | GPU junction temperature | > 83 C = thermal throttling |
| Power Draw | `DCGM_FI_DEV_POWER_USAGE` | Current power in watts | > 95% of limit = thermal risk |
| SM Clock | `DCGM_FI_DEV_SM_CLOCK` | SM frequency in MHz | Drop = thermal/power throttling |
| ECC SBE | `DCGM_FI_DEV_ECC_SBE_VOL_TOTAL` | Single-bit ECC errors | > 0 = monitor trend |
| ECC DBE | `DCGM_FI_DEV_ECC_DBE_VOL_TOTAL` | Double-bit ECC errors | > 0 = replace GPU |
| PCIe TX/RX | `DCGM_FI_DEV_PCIE_TX_THROUGHPUT` | PCIe bandwidth in KB/s | Correlate with training speed |
| NVLink TX/RX | `DCGM_FI_DEV_NVLINK_BANDWIDTH_TOTAL` | NVLink bandwidth | Low = communication bottleneck |
| XID Errors | `DCGM_FI_DEV_XID_ERRORS` | GPU error code | Any XID = investigate |

### Custom pynvml Monitoring

For custom metrics or integration with your own services, use the pynvml Python library directly. See `src/monitoring/gpu_metrics.py` for a complete implementation.

```python
import pynvml

pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)

# Utilization
util = pynvml.nvmlDeviceGetUtilizationRates(handle)
print(f"GPU: {util.gpu}%, Memory: {util.memory}%")

# Memory
mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
print(f"Used: {mem.used / 1e9:.1f} GB / {mem.total / 1e9:.1f} GB")

# Temperature
temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
print(f"Temp: {temp} C")

# Power
power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000  # mW -> W
print(f"Power: {power:.0f} W")

pynvml.nvmlShutdown()
```

### Prometheus Alerting Rules

```yaml
groups:
  - name: gpu-alerts
    rules:
      # GPU is idle but allocated (wasting money)
      - alert: GPUIdleButAllocated
        expr: DCGM_FI_DEV_GPU_UTIL < 10
        for: 15m
        labels:
          severity: warning
        annotations:
          summary: "GPU {{ $labels.gpu }} is idle ({{ $value }}% utilization)"

      # Thermal throttling imminent
      - alert: GPUThermalThrottling
        expr: DCGM_FI_DEV_GPU_TEMP > 83
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "GPU {{ $labels.gpu }} temperature {{ $value }}C (throttling threshold: 83C)"

      # Memory nearly full (OOM risk)
      - alert: GPUMemoryPressure
        expr: (DCGM_FI_DEV_FB_USED / DCGM_FI_DEV_FB_FREE) * 100 > 90
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "GPU {{ $labels.gpu }} memory at {{ $value }}%"

      # ECC double-bit errors (hardware failure)
      - alert: GPUECCDoubleBitError
        expr: DCGM_FI_DEV_ECC_DBE_VOL_TOTAL > 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "GPU {{ $labels.gpu }} has double-bit ECC errors -- schedule replacement"

      # XID errors (various GPU faults)
      - alert: GPUXIDError
        expr: DCGM_FI_DEV_XID_ERRORS > 0
        for: 1m
        labels:
          severity: warning
        annotations:
          summary: "GPU {{ $labels.gpu }} XID error {{ $value }}"
```

### Grafana Dashboard Panels

A production GPU dashboard should include these panels:

| Panel | Query | Visualization |
|---|---|---|
| GPU Utilization (all GPUs) | `DCGM_FI_DEV_GPU_UTIL` | Time series, stacked |
| Memory Usage (per GPU) | `DCGM_FI_DEV_FB_USED / DCGM_FI_DEV_FB_FREE * 100` | Gauge (0-100%) |
| Temperature (per GPU) | `DCGM_FI_DEV_GPU_TEMP` | Time series with threshold at 83 C |
| Power Draw (per GPU) | `DCGM_FI_DEV_POWER_USAGE` | Time series |
| SM Clock Speed | `DCGM_FI_DEV_SM_CLOCK` | Time series (drops indicate throttling) |
| ECC Error Rate | `rate(DCGM_FI_DEV_ECC_SBE_VOL_TOTAL[5m])` | Counter |
| PCIe Throughput | `DCGM_FI_DEV_PCIE_TX_THROUGHPUT` | Time series |
| GPU Allocation | `kube_pod_resource_limit{resource="nvidia_com_gpu"}` | Table |

---

## Hands-On Lab

### Prerequisites Check

```bash
# Verify DCGM Exporter is running
kubectl get pods -n gpu-operator -l app=nvidia-dcgm-exporter

# Check Prometheus can scrape GPU metrics
kubectl port-forward -n gpu-operator svc/nvidia-dcgm-exporter 9400:9400 &
curl -s http://localhost:9400/metrics | head -20
```

### Exercise 1: Explore nvidia-smi Monitoring

**Goal:** Master nvidia-smi for interactive GPU debugging.

```bash
# Start a GPU workload to generate utilization
kubectl apply -f - <<EOF
apiVersion: v1
kind: Pod
metadata:
  name: gpu-stress
spec:
  restartPolicy: Never
  containers:
    - name: stress
      image: nvcr.io/nvidia/cuda:12.3.1-base-ubuntu22.04
      command: ["bash", "-c", "apt-get update && apt-get install -y gpu-burn && gpu_burn 120"]
      resources:
        limits:
          nvidia.com/gpu: 1
EOF

# Monitor in real-time
nvidia-smi dmon -s pucvmet -d 1

# CSV output for analysis
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,temperature.gpu,power.draw --format=csv -l 2 > gpu_metrics.csv
```

### Exercise 2: Run the Custom GPU Metrics Exporter

**Goal:** Deploy the Python-based GPU metrics collector from `src/monitoring/gpu_metrics.py`.

```bash
# Run locally (if NVIDIA GPU is available)
python src/monitoring/gpu_metrics.py

# Or deploy on Kubernetes
kubectl apply -f - <<EOF
apiVersion: v1
kind: Pod
metadata:
  name: gpu-metrics-exporter
  labels:
    app: gpu-metrics
spec:
  containers:
    - name: exporter
      image: python:3.11-slim
      command: ["python", "/app/gpu_metrics.py"]
      ports:
        - containerPort: 9400
      env:
        - name: GPU_METRICS_PORT
          value: "9400"
        - name: GPU_METRICS_INTERVAL
          value: "15"
      resources:
        limits:
          nvidia.com/gpu: 1
EOF
```

### Exercise 3: Set Up Prometheus + Grafana GPU Dashboard

**Goal:** Create a complete GPU monitoring pipeline.

```bash
# Start the monitoring stack
docker compose up -d prometheus grafana dcgm-exporter

# Access Grafana at http://localhost:3000 (admin/admin)
# Add Prometheus data source: http://prometheus:9090
# Import NVIDIA DCGM dashboard: ID 12239
```

---

## Key Terminology

| Term | Definition |
|---|---|
| **DCGM** | Data Center GPU Manager -- NVIDIA's GPU monitoring and management framework |
| **DCGM Exporter** | DaemonSet that exposes DCGM metrics in Prometheus format on port 9400 |
| **pynvml** | Python bindings for NVML (NVIDIA Management Library) for programmatic GPU access |
| **XID Error** | NVIDIA GPU error code indicating a hardware or driver fault |
| **Thermal Throttling** | Automatic reduction of GPU clock speeds when temperature exceeds safe limits |
| **ECC** | Error-Correcting Code -- detects and corrects memory bit errors |
| **ServiceMonitor** | Prometheus Operator CRD that auto-discovers endpoints to scrape |

---

## Common Mistakes

| Mistake | Symptom | Fix |
|---|---|---|
| Not deploying DCGM Exporter | No GPU metrics in Prometheus | Ensure `dcgmExporter.enabled=true` in ClusterPolicy |
| Missing ServiceMonitor | Prometheus doesn't scrape GPU metrics | Set `dcgmExporter.serviceMonitor.enabled=true` or create manually |
| Alerting on instantaneous spikes | Alert fatigue from short GPU bursts | Set appropriate `for:` duration (5-15 minutes) |
| Ignoring ECC errors | GPU failure during training | Alert on any double-bit ECC error immediately |
| Not correlating GPU metrics with job metrics | Can't diagnose training slowdowns | Export training throughput as Prometheus metric |

---

## Self-Check Questions

1. What is the difference between GPU utilization and memory utilization? Which is more important for diagnosing slow training?
2. At what temperature does an NVIDIA GPU start thermal throttling? What are the consequences?
3. Explain the monitoring stack: GPU hardware -> NVML -> DCGM -> Prometheus -> Grafana.
4. What does a double-bit ECC error indicate? What action should you take?
5. Your GPU shows 95% utilization but training is slow. What other metrics would you check?

---

## You Know You Have Completed This Module When...

- [ ] You can use `nvidia-smi dmon` for real-time GPU monitoring
- [ ] DCGM Exporter is deployed and Prometheus is scraping GPU metrics
- [ ] You have a Grafana dashboard showing GPU utilization, memory, temperature, and power
- [ ] Alerting rules are configured for thermal throttling, ECC errors, and idle GPUs
- [ ] You understand the custom pynvml exporter code in `src/monitoring/gpu_metrics.py`
- [ ] Self-check questions answered confidently

---

## Troubleshooting

### Common Issues

**Issue: DCGM Exporter returns empty metrics**
```bash
kubectl logs -n gpu-operator -l app=nvidia-dcgm-exporter --tail=30
# Common cause: DCGM cannot connect to the GPU driver
# Fix: Ensure the NVIDIA driver is loaded and functional
nvidia-smi  # On the host, verify driver works
```

**Issue: Prometheus not scraping DCGM metrics**
```bash
# Check ServiceMonitor exists
kubectl get servicemonitor -n gpu-operator

# Verify Prometheus targets
curl http://localhost:9090/api/v1/targets | jq '.data.activeTargets[] | select(.labels.job == "dcgm-exporter")'
```

**Issue: Grafana dashboard shows "No Data"**
- Verify Prometheus data source is configured correctly in Grafana
- Check time range (GPU metrics may not have been collected yet)
- Run `curl http://localhost:9090/api/v1/query?query=DCGM_FI_DEV_GPU_UTIL` to verify data exists

---

**Next: [Module 08 - Cost Optimization](../08-monitoring-gpu-usage/)**
