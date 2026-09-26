"""vLLM engine lifecycle and self-reports (external engine; never imported as a lib).

vLLM runs in its own environment (Docker digest or locked venv). This module
owns startup, readiness, log parsing, and shutdown. With
engine.name == "mock" it launches the in-process mock server instead (tests).
Exact log/metric text is version-specific and verified in T13.
"""

from __future__ import annotations

import logging
import re
import signal
import subprocess
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import requests
import yaml

from llmlab.schemas import EngineConfig, ModelRef, StartupReport

logger = logging.getLogger(__name__)


class EngineStartupError(Exception):
    """Engine failed to start or become ready."""


class ParseError(Exception):
    """Engine self-report could not be parsed."""


_PATTERNS: dict[str, dict[str, re.Pattern[str]]] = {
    "vllm-startup-v1": {
        "kv_tokens": re.compile(r"GPU KV cache size:\s*([\d,]+)\s*tokens"),
        "max_conc": re.compile(
            r"Maximum concurrency for\s*([\d,]+)\s*tokens per request:\s*([\d.]+)x"
        ),
        "model_load": re.compile(r"Model loading took\s*([\d.]+)\s*(GiB|GB)"),
        "version": re.compile(r"vLLM version:\s*([^\s]+)"),
    }
}


def build_cli_args(model: ModelRef, args: Mapping[str, Any]) -> list[str]:
    """Build vLLM serve args: snake_case → --kebab-case; True → --flag.

    Formula: --revision always appended from the pinned ModelRef.
    """
    out: list[str] = []
    for key, value in args.items():
        flag = "--" + key.replace("_", "-")
        if value is True:
            out.append(flag)
        elif value is False:
            out.append("--no-" + key.replace("_", "-"))
        else:
            out.extend([flag, str(value)])
    out.extend(["--revision", model.revision])
    return out


def _to_int_commas(text: str) -> int:
    return int(text.replace(",", "").replace("_", ""))


def parse_startup_log(text: str, parser_version: str) -> StartupReport:
    """Parse vLLM startup log into a StartupReport (version-keyed regexes)."""
    if parser_version not in _PATTERNS:
        raise ParseError(f"unknown parser version: {parser_version}")
    pats = _PATTERNS[parser_version]
    lines = text.splitlines()
    kv_tokens: int | None = None
    num_blocks: int | None = None
    block_size: int | None = None
    max_conc: float | None = None
    max_conc_tokens: int | None = None
    model_load: int | None = None
    m = pats["kv_tokens"].search(text)
    if m:
        kv_tokens = _to_int_commas(m.group(1))
    m = pats["max_conc"].search(text)
    if m:
        max_conc_tokens = _to_int_commas(m.group(1))
        max_conc = float(m.group(2))
    m = pats["model_load"].search(text)
    if m:
        val = float(m.group(1))
        mult = 1024**3 if m.group(2) == "GiB" else 10**9
        model_load = int(val * mult)
    # Block count from "Block size: N" / "Num blocks: M" style lines if present.
    mb = re.search(r"[Nn]um(?:ber of)?\s*(?:GPU\s*)?blocks:\s*([\d,]+)", text)
    if mb:
        num_blocks = _to_int_commas(mb.group(1))
    bb = re.search(r"[Bb]lock size:\s*(\d+)", text)
    if bb:
        block_size = int(bb.group(1))
    return StartupReport(
        kv_cache_tokens=kv_tokens,
        num_gpu_blocks=num_blocks,
        block_size=block_size,
        max_concurrency=max_conc,
        max_concurrency_tokens=max_conc_tokens,
        model_load_bytes=model_load,
        raw_lines=lines[-50:],
        parser_version=parser_version,
    )


class VLLMServer:
    """Lifecycle: start → wait_ready → startup_report → stop."""

    def __init__(
        self,
        engine: EngineConfig,
        model: ModelRef,
        args: Mapping[str, Any],
        log_path: Path,
        metrics_map_path: Path | None = None,
    ) -> None:
        self.engine = engine
        self.model = model
        self.args = dict(args)
        self.log_path = log_path
        self.metrics_map_path = metrics_map_path
        self.proc: subprocess.Popen[bytes] | None = None
        self._log_fh: Any = None

    @property
    def base_url(self) -> str:
        return f"http://{self.engine.host}:{self.engine.port}"

    def _command(self) -> list[str]:
        if self.engine.name == "mock":
            cmd = self.engine.command
            if cmd == ["vllm", "serve"]:
                cmd = ["python", "-m", "llmlab.testing.mock_server"]
            extra: list[str] = []
            mock_args = self.args.get("mock", {})
            if isinstance(mock_args, dict):
                if "capacity_tokens" in mock_args:
                    extra += ["--capacity-tokens", str(mock_args["capacity_tokens"])]
                if "ttft_ms" in mock_args:
                    extra += ["--ttft-ms", str(mock_args["ttft_ms"])]
                if "itl_ms" in mock_args:
                    extra += ["--itl-ms", str(mock_args["itl_ms"])]
                if "error_rate" in mock_args:
                    extra += ["--error-rate", str(mock_args["error_rate"])]
            return [*cmd, "--port", str(self.engine.port), *extra]
        cmd = list(self.engine.command)
        # vllm serve <model> <args...>
        cmd += [self.model.hf_repo, *build_cli_args(self.model, self.args)]
        cmd += ["--host", self.engine.host, "--port", str(self.engine.port)]
        return cmd

    def start(self) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_fh = open(self.log_path, "ab")
        cmd = self._command()
        logger.info("starting engine: %s", " ".join(cmd))
        self.proc = subprocess.Popen(cmd, stdout=self._log_fh, stderr=subprocess.STDOUT)

    def wait_ready(self, timeout_s: float | None = None) -> None:
        timeout = timeout_s if timeout_s is not None else self.engine.startup_timeout_s
        deadline = time.monotonic() + timeout
        url = f"{self.base_url}/health"
        last: str = ""
        while time.monotonic() < deadline:
            if self.proc is not None and self.proc.poll() is not None:
                raise EngineStartupError(f"engine exited during startup: {self.log_path}")
            try:
                r = requests.get(url, timeout=5)
                if r.status_code == 200:
                    # Then check /v1/models lists the model.
                    try:
                        m = requests.get(f"{self.base_url}/v1/models", timeout=5).json()
                        ids = [d.get("id") for d in m.get("data", [])]
                        if self.engine.name == "mock" or self.model.hf_repo in ids or ids:
                            return
                    except Exception as e:
                        last = str(e)
                else:
                    last = f"health={r.status_code}"
            except Exception as e:
                last = str(e)
            time.sleep(2.0)
        raise EngineStartupError(f"engine not ready in {timeout}s (last: {last})")

    def startup_report(self) -> StartupReport:
        text = self.log_path.read_text(encoding="utf-8", errors="replace")
        report = parse_startup_log(text, self.engine.log_parser)
        # Cross-check cache_config_info when available (metrics endpoint).
        try:
            raw = requests.get(f"{self.base_url}/metrics", timeout=10).text
            from llmlab.telemetry import parse_cache_config_info

            labels = parse_cache_config_info(raw)
            if labels:
                nblocks = labels.get("num_gpu_blocks") or labels.get("num_blocks")
                bs = labels.get("block_size")
                if nblocks and report.num_gpu_blocks is not None:
                    if int(nblocks) != report.num_gpu_blocks:
                        msg = "log vs cache_config_info block-count mismatch"
                        raise ParseError(msg)
                if bs and report.block_size is not None and int(bs) != report.block_size:
                    raise ParseError("log vs cache_config_info block-size mismatch")
                if nblocks and report.num_gpu_blocks is None:
                    report = report.model_copy(update={"num_gpu_blocks": int(nblocks)})
                if bs and report.block_size is None:
                    report = report.model_copy(update={"block_size": int(bs)})
        except ParseError:
            raise
        except Exception as e:
            logger.debug("cache_config_info cross-check skipped: %s", e)
        return report

    def version(self) -> str:
        try:
            r = requests.get(f"{self.base_url}/version", timeout=10)
            if r.status_code == 200:
                v = r.json().get("version")
                if v:
                    return str(v)
        except Exception as e:
            logger.debug("/version lookup failed: %s", e)
        text = self.log_path.read_text(encoding="utf-8", errors="replace")
        m = _PATTERNS.get(self.engine.log_parser, {}).get("version")
        if m is not None:
            found = m.search(text)
            if found:
                return found.group(1)
        return "unknown"

    def stop(self, timeout_s: float = 60) -> None:
        if self.proc is None:
            return
        try:
            self.proc.send_signal(signal.SIGINT)
            self.proc.wait(timeout=timeout_s / 3)
        except subprocess.TimeoutExpired:
            try:
                self.proc.terminate()
                self.proc.wait(timeout=timeout_s / 3)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=timeout_s / 3)
        finally:
            if self._log_fh is not None:
                self._log_fh.close()
                self._log_fh = None

    def metrics_map(self) -> Any:
        from llmlab.schemas import MetricsMap

        path = self.metrics_map_path or Path(self.engine.metrics_map)
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        return MetricsMap(**raw)

    def __enter__(self) -> VLLMServer:
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()
