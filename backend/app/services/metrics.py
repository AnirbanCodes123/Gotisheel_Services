"""CPU / GPU / RAM utilization metrics."""

from __future__ import annotations

from typing import Any

import psutil


def _gpu_stats() -> list[dict[str, Any]]:
    try:
        import pynvml

        pynvml.nvmlInit()
        count = pynvml.nvmlDeviceGetCount()
        gpus = []
        for index in range(count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode("utf-8", errors="ignore")
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            gpus.append(
                {
                    "index": index,
                    "name": name,
                    "util_percent": util.gpu,
                    "mem_used_mb": round(mem.used / (1024 * 1024), 1),
                    "mem_total_mb": round(mem.total / (1024 * 1024), 1),
                    "mem_percent": round(100.0 * mem.used / max(mem.total, 1), 1),
                }
            )
        return gpus
    except Exception as exc:
        return [{"available": False, "detail": str(exc)}]


def collect_metrics(extra: dict[str, Any] | None = None) -> dict[str, Any]:
    try:
        vm = psutil.virtual_memory()
        payload = {
            "cpu_percent": psutil.cpu_percent(interval=0.0),
            "cpu_count": psutil.cpu_count() or 0,
            "ram_percent": vm.percent,
            "ram_used_gb": round(vm.used / (1024**3), 2),
            "ram_total_gb": round(vm.total / (1024**3), 2),
            "gpus": _gpu_stats(),
            "process": {
                "pid": psutil.Process().pid,
                "rss_mb": round(psutil.Process().memory_info().rss / (1024 * 1024), 1),
                "threads": psutil.Process().num_threads(),
            },
        }
    except Exception as exc:
        payload = {
            "cpu_percent": 0,
            "cpu_count": 0,
            "ram_percent": 0,
            "ram_used_gb": 0,
            "ram_total_gb": 0,
            "gpus": [],
            "process": {},
            "metrics_error": str(exc),
        }
    if extra:
        payload.update(extra)
    return payload
