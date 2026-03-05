"""
GPU Monitoring with pynvml and Prometheus

Collects real-time GPU metrics (utilization, memory, temperature, power,
clock speeds) using NVIDIA's pynvml library and exposes them as Prometheus
metrics for scraping by monitoring stacks (Prometheus + Grafana).

Usage:
    # Run as a standalone exporter on port 9400
    python src/monitoring/gpu_metrics.py

    # Import for programmatic access
    from src.monitoring.gpu_metrics import GPUMetricsCollector
    collector = GPUMetricsCollector()
    metrics = collector.collect_all()

Requirements:
    pip install pynvml prometheus-client python-dotenv
"""

import os
import sys
import time
import signal
import logging
from dataclasses import dataclass, field
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# pynvml import with graceful fallback for environments without NVIDIA GPUs
# ---------------------------------------------------------------------------
try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False

from prometheus_client import (
    Gauge,
    Counter,
    Histogram,
    Info,
    start_http_server,
    REGISTRY,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("gpu_metrics")

# ---------------------------------------------------------------------------
# Prometheus metric definitions
# ---------------------------------------------------------------------------
GPU_UTILIZATION = Gauge(
    "gpu_utilization_percent",
    "GPU core utilization percentage (0-100)",
    ["gpu_index", "gpu_name", "gpu_uuid"],
)
GPU_MEMORY_USED = Gauge(
    "gpu_memory_used_bytes",
    "GPU memory currently in use (bytes)",
    ["gpu_index", "gpu_name", "gpu_uuid"],
)
GPU_MEMORY_TOTAL = Gauge(
    "gpu_memory_total_bytes",
    "Total GPU memory (bytes)",
    ["gpu_index", "gpu_name", "gpu_uuid"],
)
GPU_MEMORY_UTILIZATION = Gauge(
    "gpu_memory_utilization_percent",
    "GPU memory utilization percentage (0-100)",
    ["gpu_index", "gpu_name", "gpu_uuid"],
)
GPU_TEMPERATURE = Gauge(
    "gpu_temperature_celsius",
    "GPU temperature in Celsius",
    ["gpu_index", "gpu_name", "gpu_uuid"],
)
GPU_POWER_DRAW = Gauge(
    "gpu_power_draw_watts",
    "Current GPU power draw in watts",
    ["gpu_index", "gpu_name", "gpu_uuid"],
)
GPU_POWER_LIMIT = Gauge(
    "gpu_power_limit_watts",
    "GPU power limit in watts",
    ["gpu_index", "gpu_name", "gpu_uuid"],
)
GPU_CLOCK_SM = Gauge(
    "gpu_clock_sm_mhz",
    "Current SM (streaming multiprocessor) clock speed in MHz",
    ["gpu_index", "gpu_name", "gpu_uuid"],
)
GPU_CLOCK_MEMORY = Gauge(
    "gpu_clock_memory_mhz",
    "Current memory clock speed in MHz",
    ["gpu_index", "gpu_name", "gpu_uuid"],
)
GPU_FAN_SPEED = Gauge(
    "gpu_fan_speed_percent",
    "GPU fan speed percentage",
    ["gpu_index", "gpu_name", "gpu_uuid"],
)
GPU_ECC_ERRORS = Counter(
    "gpu_ecc_errors_total",
    "Total ECC memory errors",
    ["gpu_index", "gpu_name", "gpu_uuid", "error_type"],
)
GPU_PROCESS_COUNT = Gauge(
    "gpu_running_processes",
    "Number of compute processes running on the GPU",
    ["gpu_index", "gpu_name", "gpu_uuid"],
)
GPU_INFO = Info(
    "gpu",
    "Static GPU information",
)
SCRAPE_DURATION = Histogram(
    "gpu_metrics_scrape_duration_seconds",
    "Time spent collecting GPU metrics",
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25],
)
SCRAPE_ERRORS = Counter(
    "gpu_metrics_scrape_errors_total",
    "Total number of GPU metrics collection errors",
)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class GPUDeviceMetrics:
    """Metrics snapshot for a single GPU device."""
    index: int
    name: str
    uuid: str
    utilization_gpu: float          # percent 0-100
    utilization_memory: float       # percent 0-100
    memory_used_bytes: int
    memory_total_bytes: int
    temperature_celsius: float
    power_draw_watts: float
    power_limit_watts: float
    clock_sm_mhz: int
    clock_memory_mhz: int
    fan_speed_percent: float
    process_count: int
    ecc_single_bit: int = 0
    ecc_double_bit: int = 0


# ---------------------------------------------------------------------------
# Collector
# ---------------------------------------------------------------------------
class GPUMetricsCollector:
    """
    Collects GPU metrics via pynvml and updates Prometheus gauges.

    Lifecycle:
        collector = GPUMetricsCollector()
        collector.initialize()          # Call once at startup
        metrics = collector.collect_all()  # Call on each scrape
        collector.shutdown()            # Call on exit
    """

    def __init__(self):
        self._initialized = False
        self._device_count = 0
        self._handles = []

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def initialize(self) -> bool:
        """Initialize pynvml and enumerate GPU devices."""
        if not PYNVML_AVAILABLE:
            logger.warning("pynvml is not installed. GPU metrics will be unavailable.")
            return False

        try:
            pynvml.nvmlInit()
            self._device_count = pynvml.nvmlDeviceGetCount()
            self._handles = [
                pynvml.nvmlDeviceGetHandleByIndex(i)
                for i in range(self._device_count)
            ]
            self._initialized = True

            driver = pynvml.nvmlSystemGetDriverVersion()
            cuda = pynvml.nvmlSystemGetCudaDriverVersion_v2()
            logger.info(
                "pynvml initialized: %d GPU(s), driver %s, CUDA %d.%d",
                self._device_count,
                driver,
                cuda // 1000,
                (cuda % 1000) // 10,
            )

            # Publish static GPU info
            if self._device_count > 0:
                name = pynvml.nvmlDeviceGetName(self._handles[0])
                if isinstance(name, bytes):
                    name = name.decode("utf-8")
                GPU_INFO.info({
                    "driver_version": driver if isinstance(driver, str) else driver.decode("utf-8"),
                    "device_count": str(self._device_count),
                    "gpu_0_name": name,
                })

            return True

        except pynvml.NVMLError as exc:
            logger.error("Failed to initialize pynvml: %s", exc)
            return False

    def shutdown(self):
        """Release pynvml resources."""
        if self._initialized:
            try:
                pynvml.nvmlShutdown()
                logger.info("pynvml shut down cleanly.")
            except pynvml.NVMLError as exc:
                logger.warning("Error during pynvml shutdown: %s", exc)
            self._initialized = False

    # ------------------------------------------------------------------
    # Collection
    # ------------------------------------------------------------------
    def _collect_device(self, index: int) -> Optional[GPUDeviceMetrics]:
        """Collect metrics for a single GPU."""
        handle = self._handles[index]
        try:
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode("utf-8")

            uuid = pynvml.nvmlDeviceGetUUID(handle)
            if isinstance(uuid, bytes):
                uuid = uuid.decode("utf-8")

            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            temp = pynvml.nvmlDeviceGetTemperature(
                handle, pynvml.NVML_TEMPERATURE_GPU
            )
            power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # mW -> W
            power_limit = pynvml.nvmlDeviceGetPowerManagementLimit(handle) / 1000.0
            clock_sm = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM)
            clock_mem = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM)

            # Fan speed (may not be available on all GPUs)
            try:
                fan = pynvml.nvmlDeviceGetFanSpeed(handle)
            except pynvml.NVMLError:
                fan = 0

            # Process count
            try:
                procs = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
                proc_count = len(procs)
            except pynvml.NVMLError:
                proc_count = 0

            # ECC errors (only on supported GPUs)
            ecc_single = 0
            ecc_double = 0
            try:
                ecc_single = pynvml.nvmlDeviceGetTotalEccErrors(
                    handle,
                    pynvml.NVML_SINGLE_BIT_ECC,
                    pynvml.NVML_VOLATILE_ECC,
                )
                ecc_double = pynvml.nvmlDeviceGetTotalEccErrors(
                    handle,
                    pynvml.NVML_DOUBLE_BIT_ECC,
                    pynvml.NVML_VOLATILE_ECC,
                )
            except pynvml.NVMLError:
                pass

            return GPUDeviceMetrics(
                index=index,
                name=name,
                uuid=uuid,
                utilization_gpu=util.gpu,
                utilization_memory=util.memory,
                memory_used_bytes=mem_info.used,
                memory_total_bytes=mem_info.total,
                temperature_celsius=temp,
                power_draw_watts=power,
                power_limit_watts=power_limit,
                clock_sm_mhz=clock_sm,
                clock_memory_mhz=clock_mem,
                fan_speed_percent=fan,
                process_count=proc_count,
                ecc_single_bit=ecc_single,
                ecc_double_bit=ecc_double,
            )

        except pynvml.NVMLError as exc:
            logger.error("Error collecting metrics for GPU %d: %s", index, exc)
            SCRAPE_ERRORS.inc()
            return None

    @SCRAPE_DURATION.time()
    def collect_all(self) -> list[GPUDeviceMetrics]:
        """Collect metrics for all GPUs and update Prometheus gauges."""
        if not self._initialized:
            logger.warning("Collector not initialized. Call initialize() first.")
            return []

        results = []
        for i in range(self._device_count):
            metrics = self._collect_device(i)
            if metrics is None:
                continue

            labels = {
                "gpu_index": str(metrics.index),
                "gpu_name": metrics.name,
                "gpu_uuid": metrics.uuid,
            }

            GPU_UTILIZATION.labels(**labels).set(metrics.utilization_gpu)
            GPU_MEMORY_USED.labels(**labels).set(metrics.memory_used_bytes)
            GPU_MEMORY_TOTAL.labels(**labels).set(metrics.memory_total_bytes)
            GPU_MEMORY_UTILIZATION.labels(**labels).set(metrics.utilization_memory)
            GPU_TEMPERATURE.labels(**labels).set(metrics.temperature_celsius)
            GPU_POWER_DRAW.labels(**labels).set(metrics.power_draw_watts)
            GPU_POWER_LIMIT.labels(**labels).set(metrics.power_limit_watts)
            GPU_CLOCK_SM.labels(**labels).set(metrics.clock_sm_mhz)
            GPU_CLOCK_MEMORY.labels(**labels).set(metrics.clock_memory_mhz)
            GPU_FAN_SPEED.labels(**labels).set(metrics.fan_speed_percent)
            GPU_PROCESS_COUNT.labels(**labels).set(metrics.process_count)

            results.append(metrics)

        return results

    def print_summary(self, metrics: list[GPUDeviceMetrics]):
        """Print a human-readable summary to stdout."""
        if not metrics:
            print("No GPU metrics available.")
            return

        print(f"\n{'='*70}")
        print(f" GPU Metrics Summary  ({len(metrics)} device(s))")
        print(f"{'='*70}")

        for m in metrics:
            mem_used_gb = m.memory_used_bytes / (1024 ** 3)
            mem_total_gb = m.memory_total_bytes / (1024 ** 3)
            print(f"\n  GPU {m.index}: {m.name}")
            print(f"  {'─'*50}")
            print(f"  Utilization:  {m.utilization_gpu:5.1f}%  (memory: {m.utilization_memory:.1f}%)")
            print(f"  Memory:       {mem_used_gb:.1f} / {mem_total_gb:.1f} GB")
            print(f"  Temperature:  {m.temperature_celsius:.0f} C")
            print(f"  Power:        {m.power_draw_watts:.0f} / {m.power_limit_watts:.0f} W")
            print(f"  Clocks:       SM {m.clock_sm_mhz} MHz, Mem {m.clock_memory_mhz} MHz")
            print(f"  Fan:          {m.fan_speed_percent:.0f}%")
            print(f"  Processes:    {m.process_count}")
            if m.ecc_single_bit or m.ecc_double_bit:
                print(f"  ECC Errors:   single={m.ecc_single_bit}, double={m.ecc_double_bit}")

        print(f"\n{'='*70}\n")


# ---------------------------------------------------------------------------
# Standalone exporter entry point
# ---------------------------------------------------------------------------
def main():
    """Run as a standalone Prometheus exporter."""
    port = int(os.getenv("GPU_METRICS_PORT", "9400"))
    interval = int(os.getenv("GPU_METRICS_INTERVAL", "15"))

    collector = GPUMetricsCollector()
    if not collector.initialize():
        logger.error(
            "Could not initialize GPU metrics collector. "
            "Make sure NVIDIA drivers and pynvml are installed."
        )
        sys.exit(1)

    # Graceful shutdown on SIGTERM / SIGINT
    running = True

    def _shutdown(signum, frame):
        nonlocal running
        logger.info("Received signal %s, shutting down...", signum)
        running = False

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    # Start Prometheus HTTP server
    start_http_server(port)
    logger.info("GPU metrics exporter listening on :%d (interval=%ds)", port, interval)

    while running:
        try:
            metrics = collector.collect_all()
            collector.print_summary(metrics)
        except Exception as exc:
            logger.error("Collection cycle failed: %s", exc)
            SCRAPE_ERRORS.inc()

        time.sleep(interval)

    collector.shutdown()
    logger.info("Exporter stopped.")


if __name__ == "__main__":
    main()
