"""Explicit GPU / ROCm device check for the neural slice (S05/T01).

The project's single highest environment risk is whether the RX 9070 XT
(``gfx1201``) is actually visible to a ROCm container, and whether torch
resolves a GPU at all. This module answers that question with a **recorded
artifact** instead of an assumption: it probes the runtime, resolves the torch
device, and writes a human-readable report (``reports/gpu_check.md``) plus an
optional machine-readable JSON sidecar.

What is probed
--------------
* **torch** — imported lazily, so the module (and its tests) work on a CPU-only
  host. The probe records the torch/HIP versions, ``torch.cuda.is_available()``,
  the device count, the device name, the ``gcnArchName`` (the architecture
  string torch exposes, e.g. ``gfx1201``), and the device torch resolves:
  ``cuda:0`` when a GPU is usable, ``cpu`` otherwise.
* **ROCm tooling** — ``rocminfo``, ``amd-smi static``, and ``rocm-smi`` are run
  if present, each under a hard timeout, and their output is scanned for
  ``gfx`` architecture tokens. This corroborates the torch finding from the
  runtime side.
* **ROCm version** — read from ``/opt/rocm/.info/version`` when available.

Fail-soft, never silent
-----------------------
Every external dependency can fail without crashing the check: a missing torch
import, a missing ROCm binary, a hung subprocess, and an unreadable version
file are each recorded as a probe result (or a warning) and the check still
produces a report. A GPU that is absent is a **documented fact**, not an error:
:func:`main` exits ``0`` so downstream slices can fall back to CPU with a
logged warning. Genuine misuse (an empty target architecture, a non-result
handed to the writer) raises a named :class:`DeviceCheckError` subclass.

Observability
-------------
:func:`run_device_check` logs the resolved device, the target-visibility
verdict, and each warning at ``INFO``/``WARNING``. :func:`render_device_check_report`
renders the markdown artifact, :func:`describe_device_check` renders the
one-block CLI summary, and every :class:`DeviceCheckResult` is available as a
JSON-serialisable dict via :meth:`DeviceCheckResult.to_dict`.
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Sequence

from heart.config import REPORTS_DIR

logger = logging.getLogger(__name__)

__all__ = [
    "DEVICE_CHECK_SCHEMA_VERSION",
    "TARGET_GPU_ARCH",
    "TARGET_GPU_MARKETING_NAME",
    "DEVICE_CUDA",
    "DEVICE_CPU",
    "DEFAULT_REPORT_FILENAME",
    "DEFAULT_JSON_FILENAME",
    "DEFAULT_REPORT_PATH",
    "DEFAULT_JSON_PATH",
    "DEFAULT_PROBE_TIMEOUT_SECONDS",
    "PROBE_FOUND",
    "PROBE_MISSING",
    "PROBE_ERROR",
    "PROBE_SKIPPED",
    "GPU_TOOLING_COMMANDS",
    "DeviceCheckError",
    "DeviceCheckConfigError",
    "DeviceCheckReportError",
    "CommandResult",
    "DeviceProbe",
    "TorchProbe",
    "DeviceCheckResult",
    "CommandRunner",
    "default_command_runner",
    "read_rocm_version",
    "in_container",
    "probe_torch",
    "probe_rocm_tooling",
    "run_device_check",
    "render_device_check_report",
    "write_device_check_report",
    "write_device_check_json",
    "run_and_write_device_check",
    "describe_device_check",
    "build_parser",
    "main",
]

# ---------------------------------------------------------------------------
# Declared constants
# ---------------------------------------------------------------------------

#: Machine-readable schema version of the JSON sidecar.
DEVICE_CHECK_SCHEMA_VERSION: str = "1"

#: The GPU architecture the RX 9070 XT (Navi 48) exposes to ROCm.
TARGET_GPU_ARCH: str = "gfx1201"

#: Human-readable name of the target GPU.
TARGET_GPU_MARKETING_NAME: str = "AMD Radeon RX 9070 XT"

#: The torch device string used when a GPU is usable.
DEVICE_CUDA: str = "cuda:0"

#: The torch device string used when no GPU is usable.
DEVICE_CPU: str = "cpu"

#: Default markdown report filename (under ``reports/``).
DEFAULT_REPORT_FILENAME: str = "gpu_check.md"

#: Default JSON sidecar filename (under ``reports/``).
DEFAULT_JSON_FILENAME: str = "gpu_check.json"

#: Default markdown report location.
DEFAULT_REPORT_PATH: Path = Path(REPORTS_DIR) / DEFAULT_REPORT_FILENAME

#: Default JSON sidecar location.
DEFAULT_JSON_PATH: Path = Path(REPORTS_DIR) / DEFAULT_JSON_FILENAME

#: Hard per-command timeout, so a wedged ROCm tool cannot hang the check.
DEFAULT_PROBE_TIMEOUT_SECONDS: float = 20.0

#: A probe that produced evidence of its subject.
PROBE_FOUND: str = "found"

#: A probe whose subject is not installed / not on PATH.
PROBE_MISSING: str = "missing"

#: A probe that ran but could not produce evidence (error / bad exit / timeout).
PROBE_ERROR: str = "error"

#: A probe that was deliberately not attempted.
PROBE_SKIPPED: str = "skipped"

#: ROCm tooling probed, in order, with the argument vector to run.
GPU_TOOLING_COMMANDS: tuple[tuple[str, ...], ...] = (
    ("rocminfo",),
    ("amd-smi", "static"),
    ("rocm-smi",),
)

#: Matches ROCm architecture tokens such as ``gfx1201`` or ``gfx90a``.
_GFX_TOKEN_RE = re.compile(r"gfx[0-9a-f]{2,}", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Named exceptions
# ---------------------------------------------------------------------------


class DeviceCheckError(Exception):
    """Base class for every device-check failure."""


class DeviceCheckConfigError(DeviceCheckError):
    """The requested device-check configuration is invalid."""


class DeviceCheckReportError(DeviceCheckError):
    """The device-check report could not be written."""


# ---------------------------------------------------------------------------
# Subprocess probing
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CommandResult:
    """The outcome of one probed subprocess, including its failure mode."""

    argv: tuple[str, ...]
    returncode: int | None
    stdout: str = ""
    stderr: str = ""
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and self.error is None

    def to_dict(self) -> dict[str, object]:
        return {
            "argv": list(self.argv),
            "returncode": self.returncode,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "error": self.error,
            "ok": self.ok,
        }


#: Injection seam: ``(argv, timeout_seconds) -> CommandResult``.
CommandRunner = Callable[[Sequence[str], float], CommandResult]


def default_command_runner(argv: Sequence[str], timeout_seconds: float) -> CommandResult:
    """Run ``argv`` capturing output, converting every failure into a result.

    A binary that is not on ``PATH``, a timeout, and an OS-level spawn error are
    each reported through :attr:`CommandResult.error` rather than raised, so the
    device check can record a missing ROCm tool instead of crashing.
    """
    tokens = tuple(str(token) for token in argv)
    if not tokens:
        return CommandResult(tokens, None, error="empty command")
    if shutil.which(tokens[0]) is None:
        return CommandResult(tokens, None, error=f"{tokens[0]!r} not found on PATH")
    try:
        proc = subprocess.run(  # noqa: S603 - argv is a fixed declared tuple
            list(tokens),
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return CommandResult(
            tokens,
            None,
            stdout=_decode(exc.stdout),
            stderr=_decode(exc.stderr),
            error=f"timed out after {timeout_seconds:.0f}s",
        )
    except OSError as exc:
        return CommandResult(tokens, None, error=f"{type(exc).__name__}: {exc}")
    return CommandResult(tokens, proc.returncode, proc.stdout, proc.stderr, None)


def _decode(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


# ---------------------------------------------------------------------------
# Probe value objects
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DeviceProbe:
    """One named probe and what it observed."""

    name: str
    status: str
    detail: str

    def to_dict(self) -> dict[str, object]:
        return {"name": self.name, "status": self.status, "detail": self.detail}


@dataclass(frozen=True)
class TorchProbe:
    """What the torch runtime reported about compute devices."""

    torch_available: bool
    resolves_to: str
    torch_version: str | None = None
    hip_version: str | None = None
    cuda_available: bool = False
    device_count: int = 0
    device_name: str | None = None
    gcn_arch_name: str | None = None
    error: str | None = None

    @property
    def gpu_available(self) -> bool:
        return self.cuda_available and self.device_count > 0 and self.error is None

    def to_dict(self) -> dict[str, object]:
        return {
            "torch_available": self.torch_available,
            "resolves_to": self.resolves_to,
            "torch_version": self.torch_version,
            "hip_version": self.hip_version,
            "cuda_available": self.cuda_available,
            "device_count": int(self.device_count),
            "device_name": self.device_name,
            "gcn_arch_name": self.gcn_arch_name,
            "error": self.error,
            "gpu_available": self.gpu_available,
        }


@dataclass(frozen=True)
class DeviceCheckResult:
    """Everything the device check established, in one recordable object."""

    generated_at: str
    hostname: str
    platform_name: str
    in_container: bool
    gfx_target: str
    target_marketing_name: str
    gfx_target_visible: bool
    torch_device: str
    torch: TorchProbe
    probes: tuple[DeviceProbe, ...]
    rocm_version: str | None
    warnings: tuple[str, ...]
    schema_version: str = DEVICE_CHECK_SCHEMA_VERSION

    @property
    def gpu_available(self) -> bool:
        return self.torch_device != DEVICE_CPU

    @property
    def cpu_fallback(self) -> bool:
        return not self.gpu_available

    def record_for(self, name: str) -> DeviceProbe | None:
        for probe in self.probes:
            if probe.name == name:
                return probe
        return None

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "generated_at": self.generated_at,
            "hostname": self.hostname,
            "platform": self.platform_name,
            "in_container": self.in_container,
            "gfx_target": self.gfx_target,
            "target_marketing_name": self.target_marketing_name,
            "gfx_target_visible": self.gfx_target_visible,
            "torch_device": self.torch_device,
            "gpu_available": self.gpu_available,
            "cpu_fallback": self.cpu_fallback,
            "rocm_version": self.rocm_version,
            "torch": self.torch.to_dict(),
            "probes": [probe.to_dict() for probe in self.probes],
            "warnings": list(self.warnings),
        }


# ---------------------------------------------------------------------------
# Environment probes
# ---------------------------------------------------------------------------


def in_container() -> bool:
    """Return whether the current process appears to run inside a container."""
    if os.environ.get("container"):
        return True
    return any(
        Path(marker).exists()
        for marker in ("/run/.containerenv", "/.dockerenv")
    )


def read_rocm_version(path: str | Path = "/opt/rocm/.info/version") -> str | None:
    """Read the ROCm version string, returning ``None`` when unavailable."""
    try:
        text = Path(path).read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return text or None


def probe_torch(
    torch_module: object | None = None,
    *,
    importer: Callable[[str], object] = importlib.import_module,
) -> TorchProbe:
    """Probe torch for compute devices, resolving ``cuda:0`` or ``cpu``.

    Parameters
    ----------
    torch_module:
        A pre-imported torch module (used by tests to simulate a GPU, a CPU-only
        runtime, or a probe failure). When ``None``, torch is imported lazily.
    importer:
        Import seam used only when ``torch_module`` is ``None``; tests can pass
        an importer that raises :class:`ImportError` to simulate the missing
        dependency.
    """
    module = torch_module
    if module is None:
        try:
            module = importer("torch")
        except ImportError as exc:
            detail = f"torch is not importable: {exc}"
            logger.warning("%s; neural runs will use the CPU fallback.", detail)
            return TorchProbe(
                torch_available=False,
                resolves_to=DEVICE_CPU,
                error=detail,
            )

    torch_version = _string_or_none(getattr(module, "__version__", None))
    version_attr = getattr(module, "version", None)
    hip_version = _string_or_none(getattr(version_attr, "hip", None))

    try:
        cuda = module.cuda  # type: ignore[attr-defined]
        cuda_available = bool(cuda.is_available())
        device_count = int(cuda.device_count()) if cuda_available else 0
        device_name: str | None = None
        gcn_arch: str | None = None
        if cuda_available and device_count > 0:
            properties = cuda.get_device_properties(0)
            device_name = _string_or_none(getattr(properties, "name", None))
            gcn_arch = _string_or_none(
                getattr(properties, "gcnArchName", None)
            ) or _string_or_none(getattr(properties, "gcn_arch_name", None))
        resolves_to = DEVICE_CUDA if device_count > 0 else DEVICE_CPU
        return TorchProbe(
            torch_available=True,
            resolves_to=resolves_to,
            torch_version=torch_version,
            hip_version=hip_version,
            cuda_available=cuda_available,
            device_count=device_count,
            device_name=device_name,
            gcn_arch_name=gcn_arch,
        )
    except Exception as exc:  # noqa: BLE001 - a probe failure must not crash the check
        detail = f"{type(exc).__name__}: {exc}"
        logger.warning(
            "torch device probe failed (%s); resolving to CPU fallback.", detail
        )
        return TorchProbe(
            torch_available=True,
            resolves_to=DEVICE_CPU,
            torch_version=torch_version,
            hip_version=hip_version,
            error=detail,
        )


def _string_or_none(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def probe_rocm_tooling(
    *,
    command_runner: CommandRunner | None = None,
    timeout_seconds: float = DEFAULT_PROBE_TIMEOUT_SECONDS,
    gfx_target: str = TARGET_GPU_ARCH,
) -> tuple[tuple[DeviceProbe, ...], tuple[str, ...]]:
    """Run the ROCm tooling and scan its output for ``gfx`` architecture tokens.

    Returns the probe records and the deduplicated architecture tokens found.
    A tool that is missing, times out, or exits non-zero is recorded, never
    raised.
    """
    runner = command_runner or default_command_runner
    probes: list[DeviceProbe] = []
    architectures: list[str] = []

    for argv in GPU_TOOLING_COMMANDS:
        result = runner(argv, timeout_seconds)
        tool = " ".join(argv)
        if result.error is not None:
            status = PROBE_MISSING if "not found on PATH" in result.error else PROBE_ERROR
            detail = result.error
        elif not result.ok:
            status = PROBE_ERROR
            detail = f"exit code {result.returncode}"
        else:
            found = _extract_gfx_tokens(result.stdout, result.stderr)
            for arch in found:
                if arch not in architectures:
                    architectures.append(arch)
            status = PROBE_FOUND if found else PROBE_SKIPPED
            detail = (
                f"architectures: {', '.join(found)}" if found else "no gfx token in output"
            )
        probes.append(DeviceProbe(name=tool, status=status, detail=detail))

    target_visible = any(
        arch.lower() == gfx_target.lower() for arch in architectures
    )
    probes.append(
        DeviceProbe(
            name="rocm-architectures",
            status=PROBE_FOUND if target_visible else PROBE_MISSING,
            detail=(
                f"target {gfx_target} "
                + ("visible in ROCm tooling" if target_visible else "not seen in ROCm tooling")
                + (f"; all: {', '.join(architectures)}" if architectures else "")
            ),
        )
    )
    return tuple(probes), tuple(architectures)


def _extract_gfx_tokens(*chunks: str) -> list[str]:
    tokens: list[str] = []
    for chunk in chunks:
        for match in _GFX_TOKEN_RE.findall(chunk or ""):
            token = match.lower()
            if token not in tokens:
                tokens.append(token)
    return tokens


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def _validate_gfx_target(gfx_target: object) -> str:
    if not isinstance(gfx_target, str) or not gfx_target.strip():
        raise DeviceCheckConfigError(
            f"gfx_target must be a non-empty string, got {gfx_target!r}."
        )
    return gfx_target.strip()


def run_device_check(
    *,
    torch_module: object | None = None,
    torch_importer: Callable[[str], object] = importlib.import_module,
    command_runner: CommandRunner | None = None,
    gfx_target: str = TARGET_GPU_ARCH,
    target_marketing_name: str = TARGET_GPU_MARKETING_NAME,
    timeout_seconds: float = DEFAULT_PROBE_TIMEOUT_SECONDS,
    rocm_version_path: str | Path = "/opt/rocm/.info/version",
) -> DeviceCheckResult:
    """Probe the runtime and return the recorded device-check result.

    All external dependencies are injectable so the no-GPU and GPU branches can
    be exercised without a GPU, torch, or ROCm tooling present.
    """
    target = _validate_gfx_target(gfx_target)
    if isinstance(timeout_seconds, bool) or not isinstance(
        timeout_seconds, (int, float)
    ):
        raise DeviceCheckConfigError(
            f"timeout_seconds must be a number, got {timeout_seconds!r}."
        )
    if timeout_seconds <= 0:
        raise DeviceCheckConfigError(
            f"timeout_seconds must be > 0, got {timeout_seconds!r}."
        )

    torch_probe = probe_torch(torch_module, importer=torch_importer)
    probes, architectures = probe_rocm_tooling(
        command_runner=command_runner,
        timeout_seconds=float(timeout_seconds),
        gfx_target=target,
    )
    rocm_version = read_rocm_version(rocm_version_path)

    torch_arch = (torch_probe.gcn_arch_name or "").lower()
    gfx_target_visible = (
        any(arch == target.lower() for arch in architectures)
        or torch_arch == target.lower()
    )

    warnings: list[str] = []
    if not torch_probe.torch_available:
        warnings.append(
            "torch is not available in this environment; the neural slice must "
            f"use the documented CPU fallback ({torch_probe.error})."
        )
    elif torch_probe.error is not None:
        warnings.append(
            f"torch device probe raised ({torch_probe.error}); the check "
            "resolved to the CPU fallback rather than trusting a partial probe."
        )
    if torch_probe.resolves_to == DEVICE_CPU:
        warnings.append(
            f"GPU unavailable; torch resolved to {DEVICE_CPU}. Neural runs will "
            "run on CPU with an explicit WARNING (CPU fallback)."
        )
    if gfx_target_visible and torch_probe.resolves_to == DEVICE_CPU:
        warnings.append(
            f"{target} is visible to ROCm tooling but torch resolved to "
            f"{DEVICE_CPU}; the GPU is not usable by torch."
        )
    if (
        torch_probe.resolves_to != DEVICE_CPU
        and torch_probe.gcn_arch_name
        and torch_arch != target.lower()
    ):
        warnings.append(
            f"torch resolved {torch_probe.resolves_to} but reports architecture "
            f"{torch_probe.gcn_arch_name!r}, not the target {target!r}."
        )

    result = DeviceCheckResult(
        generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        hostname=socket.gethostname(),
        platform_name=f"{platform.system()} {platform.machine()}",
        in_container=in_container(),
        gfx_target=target,
        target_marketing_name=target_marketing_name,
        gfx_target_visible=gfx_target_visible,
        torch_device=torch_probe.resolves_to,
        torch=torch_probe,
        probes=probes,
        rocm_version=rocm_version,
        warnings=tuple(warnings),
    )
    logger.info(
        "Device check: target=%s visible=%s torch_device=%s torch=%s "
        "container=%s",
        target,
        "yes" if gfx_target_visible else "no",
        result.torch_device,
        torch_probe.torch_version or "unavailable",
        result.in_container,
    )
    for warning in warnings:
        logger.warning("%s", warning)
    return result


# ---------------------------------------------------------------------------
# Report rendering / writing
# ---------------------------------------------------------------------------


def _visibility_word(visible: bool) -> str:
    return "YES" if visible else "NO"


def render_device_check_report(result: DeviceCheckResult) -> str:
    """Render the device-check result as the markdown artifact.

    The first two bold lines are the contract: they state explicitly whether the
    target architecture is visible and which torch device was resolved.
    """
    if not isinstance(result, DeviceCheckResult):
        raise DeviceCheckReportError(
            f"result must be a DeviceCheckResult, got {type(result).__name__}."
        )
    torch = result.torch
    lines: list[str] = [
        "# GPU / ROCm device check",
        "",
        "_Generated by `heart.gpu.device_check` from live runtime probes. "
        "Do not edit by hand._",
        "",
        f"**{result.gfx_target} visible: {_visibility_word(result.gfx_target_visible)}**",
        f"**torch device resolved: {result.torch_device}**",
        "",
        f"Target GPU: `{result.gfx_target}` ({result.target_marketing_name})",
        "",
        "## Runtime",
        "",
        f"- generated at: {result.generated_at}",
        f"- hostname: {result.hostname}",
        f"- platform: {result.platform_name}",
        f"- running in container: {'yes' if result.in_container else 'no'}",
        f"- ROCm version: {result.rocm_version or 'unknown'}",
        f"- torch version: {torch.torch_version or 'unavailable'}",
        f"- HIP version: {torch.hip_version or 'unknown'}",
        f"- torch.cuda.is_available(): {torch.cuda_available}",
        f"- torch device count: {torch.device_count}",
        f"- torch device name: {torch.device_name or 'none'}",
        f"- torch gcnArchName: {torch.gcn_arch_name or 'none'}",
        f"- GPU usable by torch: {'yes' if result.gpu_available else 'no'}",
        "",
        "## Probes",
        "",
        "| probe | status | detail |",
        "| --- | --- | --- |",
    ]
    for probe in result.probes:
        detail = probe.detail.replace("|", "\\|")
        lines.append(f"| `{probe.name}` | {probe.status} | {detail} |")
    lines.append("")

    if result.cpu_fallback:
        lines.extend(
            [
                "## Interpretation",
                "",
                f"The target architecture is not usable by torch in this run, so "
                f"neural training resolves to `{result.torch_device}`. This is a "
                "documented CPU fallback, not a silent failure: the MLP sweep "
                "will log a WARNING and still complete.",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "## Interpretation",
                "",
                f"torch resolved `{result.torch_device}`"
                + (
                    f" ({torch.device_name})"
                    if torch.device_name
                    else ""
                )
                + f", and the target architecture `{result.gfx_target}` is "
                + ("visible" if result.gfx_target_visible else "not confirmed")
                + ". Neural training uses the GPU path.",
                "",
            ]
        )

    if result.warnings:
        lines.append("## Warnings")
        lines.append("")
        for warning in result.warnings:
            lines.append(f"- {warning}")
        lines.append("")

    lines.extend(
        [
            "## Reproduce",
            "",
            "```bash",
            "containers/run-rocm.sh",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def _atomic_write_text(path: str | Path, content: str) -> Path:
    destination = Path(path)
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        tmp = destination.with_name(destination.name + ".part")
        tmp.write_text(content, encoding="utf-8")
        os.replace(tmp, destination)
    except OSError as exc:
        raise DeviceCheckReportError(
            f"Could not write the device-check report to {destination}: {exc}"
        ) from exc
    return destination


def write_device_check_report(result: DeviceCheckResult, path: str | Path) -> Path:
    """Atomically write the markdown device-check report for ``result``."""
    if not isinstance(result, DeviceCheckResult):
        raise DeviceCheckReportError(
            f"result must be a DeviceCheckResult, got {type(result).__name__}."
        )
    destination = _atomic_write_text(path, render_device_check_report(result))
    logger.info("Wrote GPU device-check report to %s", destination)
    return destination


def write_device_check_json(result: DeviceCheckResult, path: str | Path) -> Path:
    """Atomically write the machine-readable JSON sidecar for ``result``."""
    if not isinstance(result, DeviceCheckResult):
        raise DeviceCheckReportError(
            f"result must be a DeviceCheckResult, got {type(result).__name__}."
        )
    try:
        payload = json.dumps(result.to_dict(), indent=2, sort_keys=True)
    except (TypeError, ValueError) as exc:
        raise DeviceCheckReportError(
            f"Could not serialise the device-check result: {exc}"
        ) from exc
    destination = _atomic_write_text(path, payload + "\n")
    logger.info("Wrote GPU device-check JSON to %s", destination)
    return destination


def run_and_write_device_check(
    *,
    report_path: str | Path = DEFAULT_REPORT_PATH,
    json_path: str | Path | None = None,
    torch_module: object | None = None,
    torch_importer: Callable[[str], object] = importlib.import_module,
    command_runner: CommandRunner | None = None,
    gfx_target: str = TARGET_GPU_ARCH,
    timeout_seconds: float = DEFAULT_PROBE_TIMEOUT_SECONDS,
) -> tuple[DeviceCheckResult, Path, Path | None]:
    """Run the check and persist its report (and optional JSON sidecar).

    Returns the result, the markdown path, and the JSON path (or ``None``).
    """
    result = run_device_check(
        torch_module=torch_module,
        torch_importer=torch_importer,
        command_runner=command_runner,
        gfx_target=gfx_target,
        timeout_seconds=timeout_seconds,
    )
    report = write_device_check_report(result, report_path)
    json_written = (
        write_device_check_json(result, json_path) if json_path is not None else None
    )
    return result, report, json_written


def describe_device_check(result: DeviceCheckResult) -> str:
    """Render the one-block CLI summary of a device-check result."""
    if not isinstance(result, DeviceCheckResult):
        raise DeviceCheckReportError(
            f"result must be a DeviceCheckResult, got {type(result).__name__}."
        )
    return "\n".join(
        [
            f"target {result.gfx_target} ({result.target_marketing_name}): "
            f"{'VISIBLE' if result.gfx_target_visible else 'NOT VISIBLE'}",
            f"torch device resolved: {result.torch_device}",
            f"torch: {result.torch.torch_version or 'unavailable'}"
            + (f" (HIP {result.torch.hip_version})" if result.torch.hip_version else ""),
            f"gpu name: {result.torch.device_name or 'none'}",
            f"gcnArchName: {result.torch.gcn_arch_name or 'none'}",
            f"ROCm: {result.rocm_version or 'unknown'}"
            f" | container: {'yes' if result.in_container else 'no'}",
            f"cpu fallback in effect: {'yes' if result.cpu_fallback else 'no'}",
        ]
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m heart.gpu.device_check",
        description=(
            "Probe the runtime for the target GPU architecture and the torch "
            "device torch resolves, and record the finding as a report artifact."
        ),
    )
    parser.add_argument(
        "--report-path",
        default=str(DEFAULT_REPORT_PATH),
        help=f"Markdown report destination (default: {DEFAULT_REPORT_PATH}).",
    )
    parser.add_argument(
        "--json-path",
        default=None,
        help=(
            "Optional machine-readable JSON sidecar destination "
            f"(suggested: {DEFAULT_JSON_PATH})."
        ),
    )
    parser.add_argument(
        "--gfx-target",
        default=TARGET_GPU_ARCH,
        help=f"Target GPU architecture to look for (default: {TARGET_GPU_ARCH}).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_PROBE_TIMEOUT_SECONDS,
        help=(
            "Per-command probe timeout in seconds "
            f"(default: {DEFAULT_PROBE_TIMEOUT_SECONDS:.0f})."
        ),
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the full result as JSON instead of the summary block.",
    )
    return parser


def main(
    argv: list[str] | None = None,
    *,
    torch_module: object | None = None,
    torch_importer: Callable[[str], object] = importlib.import_module,
    command_runner: CommandRunner | None = None,
) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    try:
        result, report, sidecar = run_and_write_device_check(
            report_path=args.report_path,
            json_path=args.json_path,
            torch_module=torch_module,
            torch_importer=torch_importer,
            command_runner=command_runner,
            gfx_target=args.gfx_target,
            timeout_seconds=args.timeout,
        )
    except DeviceCheckError as exc:
        print(f"device-check error: {exc}")
        return 1

    if args.json:
        print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    else:
        print(describe_device_check(result))
        print(f"report: {report}")
        if sidecar is not None:
            print(f"json:   {sidecar}")
    # A missing GPU is a documented fact, not a failure: exit 0 either way.
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised via subprocess/CLI
    sys.exit(main())
