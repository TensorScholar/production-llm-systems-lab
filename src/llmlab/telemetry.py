"""Server metrics (/metrics) and GPU (NVML) sampling."""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from typing import Any

import aiohttp
import pandas as pd
from prometheus_client.parser import text_string_to_metric_families

from llmlab.schemas import MetricsMap

logger = logging.getLogger(__name__)


def parse_metrics_text(text: str, metrics_map: MetricsMap) -> dict[str, float]:
    """Parse Prometheus text; map version-specific names to logical names.

    Formula: first matching candidate name wins. Unmapped names ignored.
    """
    by_name: dict[str, float] = {}
    for family in text_string_to_metric_families(text):
        for sample in family.samples:
            try:
                by_name[sample.name] = float(sample.value)
            except (TypeError, ValueError):
                continue
    out: dict[str, float] = {}
    mapping = metrics_map.model_dump()
    for logical in (
        "kv_usage",
        "running",
        "waiting",
        "preemptions",
        "prompt_tokens",
        "generation_tokens",
    ):
        candidates = mapping.get(logical, [])
        for cand in candidates:
            if cand in by_name:
                out[logical] = by_name[cand]
                break
    return out


def parse_cache_config_info(text: str) -> dict[str, str]:
    """Extract vllm:cache_config_info labels, if exposed (version-specific)."""
    labels: dict[str, str] = {}
    for family in text_string_to_metric_families(text):
        for sample in family.samples:
            if sample.name in ("vllm:cache_config_info", "vllm_cache_config_info"):
                labels = {k: str(v) for k, v in sample.labels.items()}
    return labels


class PrometheusScraper:
    """Poll /metrics at interval_s; return long-form samples."""

    def __init__(self, url: str, interval_s: float, metrics_map: MetricsMap) -> None:
        self.url = url
        self.interval_s = interval_s
        self.metrics_map = metrics_map

    async def snapshot_raw(self) -> str:
        async with aiohttp.ClientSession() as session:
            async with session.get(self.url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                resp.raise_for_status()
                return await resp.text()

    async def run(self, stop: asyncio.Event) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        async with aiohttp.ClientSession() as session:
            while not stop.is_set():
                t_ns = time.perf_counter_ns()
                try:
                    async with session.get(
                        self.url, timeout=aiohttp.ClientTimeout(total=10)
                    ) as resp:
                        text = await resp.text()
                    parsed = parse_metrics_text(text, self.metrics_map)
                    for name, value in parsed.items():
                        rows.append({"t_ns": t_ns, "name": name, "value": value})
                except Exception as e:
                    logger.debug("scrape failed: %s", e)
                try:
                    await asyncio.wait_for(stop.wait(), timeout=self.interval_s)
                except TimeoutError:
                    continue
        return pd.DataFrame(rows, columns=["t_ns", "name", "value"])


class NvmlSampler:
    """NVML GPU sampler (5 Hz default). Marks unavailable if nvidia-ml-py missing."""

    def __init__(self, device_index: int = 0, interval_s: float = 0.2) -> None:
        self.device_index = device_index
        self.interval_s = interval_s

    def available(self) -> bool:
        import importlib.util
        import sys

        if "pynvml" in sys.modules:
            return True
        return importlib.util.find_spec("pynvml") is not None

    def device_info(self) -> dict[str, Any]:
        try:
            import pynvml  # type: ignore[import-not-found]

            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_index)
            name = pynvml.nvmlDeviceGetName(handle)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            driver = pynvml.nvmlSystemGetDriverVersion()
            return {
                "name": name if isinstance(name, str) else name.decode(),
                "total_bytes": int(mem.total),
                "driver": driver if isinstance(driver, str) else driver.decode(),
            }
        except Exception as e:
            return {"error": f"{type(e).__name__}: {e}"}

    def run_in_thread(self, stop: threading.Event) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        try:
            import pynvml

            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_index)
        except Exception as e:
            logger.warning("NVML unavailable: %s", e)
            return pd.DataFrame()
        while not stop.is_set():
            try:
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
                temp = pynvml.nvmlDeviceGetTemperature(handle, 0)
                clock = pynvml.nvmlDeviceGetClockInfo(handle, 0)
                mem_clock = pynvml.nvmlDeviceGetClockInfo(handle, 2)
                throttle = pynvml.nvmlDeviceGetCurrentClocksThrottleReasons(handle)
                rows.append(
                    {
                        "t_ns": time.perf_counter_ns(),
                        "mem_used_bytes": int(mem.used),
                        "mem_total_bytes": int(mem.total),
                        "gpu_util": float(util.gpu),
                        "sm_clock_mhz": float(clock),
                        "mem_clock_mhz": float(mem_clock),
                        "power_w": float(power),
                        "temp_c": float(temp),
                        "throttle_reasons": int(throttle),
                    }
                )
            except Exception as e:
                logger.debug("nvml sample failed: %s", e)
            stop.wait(self.interval_s)
        return pd.DataFrame(rows)
