"""Tests for the explicit GPU / ROCm device check (S05/T01).

Contracts under test:

1. **torch probing** — :func:`heart.gpu.device_check.probe_torch` resolves
   ``cuda:0`` when a GPU is usable and ``cpu`` otherwise, and never raises: a
   missing torch import and a torch probe that throws are both recorded.
2. **ROCm tooling probing** — missing binaries, non-zero exits, timeouts, and
   malformed output are recorded as probe results; ``gfx`` tokens are parsed
   from stdout/stderr.
3. **Orchestration** — :func:`run_device_check` reports the target-visible
   verdict and the resolved device, and emits the CPU-fallback warning when no
   GPU is usable. Both the GPU and the no-GPU branch are exercised by
   simulation (no GPU, torch, or ROCm tooling required).
4. **Artifact** — :func:`render_device_check_report` states explicitly whether
   ``gfx1201`` is visible and which torch device resolved;
   :func:`write_device_check_report` writes it atomically, creating parent
   directories, and the JSON sidecar round-trips.
5. **Negative paths** — empty target, non-positive timeout, a non-result
   passed to the writer, and a missing report directory all raise the named
   ``DeviceCheckError`` subclasses rather than leaking an unhandled error.

No test requires a GPU, torch, or ROCm tooling: the probes accept injected
fakes, which is exactly what lets the no-GPU branch be asserted directly.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from heart.gpu.device_check import (
    DEVICE_CHECK_SCHEMA_VERSION,
    DEVICE_CPU,
    DEVICE_CUDA,
    PROBE_ERROR,
    PROBE_FOUND,
    PROBE_MISSING,
    PROBE_SKIPPED,
    TARGET_GPU_ARCH,
    CommandResult,
    DeviceCheckConfigError,
    DeviceCheckReportError,
    build_parser,
    default_command_runner,
    describe_device_check,
    in_container,
    main,
    probe_rocm_tooling,
    probe_torch,
    read_rocm_version,
    render_device_check_report,
    run_and_write_device_check,
    run_device_check,
    write_device_check_json,
    write_device_check_report,
)


# ---------------------------------------------------------------------------
# Fakes / helpers
# ---------------------------------------------------------------------------


def _fake_torch(
    *,
    available: bool,
    count: int = 0,
    name: str | None = None,
    gcn: str | None = None,
    hip: str | None = None,
    version: str = "2.10.0+rocm7.2.4",
    raises: bool = False,
) -> object:
    """Build a duck-typed torch stand-in for the probe."""

    class _Cuda:
        def is_available(self) -> bool:
            if raises:
                raise RuntimeError("HIP error: no device")
            return available

        def device_count(self) -> int:
            if raises:
                raise RuntimeError("HIP error: no device")
            return count

        def get_device_properties(self, index: int) -> object:
            return SimpleNamespace(name=name, gcnArchName=gcn)

    return SimpleNamespace(cuda=_Cuda(), version=SimpleNamespace(hip=hip), __version__=version)


def _cpu_torch() -> object:
    return _fake_torch(available=False)


def _gpu_torch() -> object:
    return _fake_torch(
        available=True,
        count=1,
        name="AMD Radeon RX 9070 XT",
        gcn=TARGET_GPU_ARCH,
        hip="7.2.53211",
    )


def _runner(outputs: dict[str, str] | None = None, *, missing: tuple[str, ...] = ()):
    """A command runner returning canned stdout, or a not-found error."""
    mapping = outputs or {}

    def run(argv, timeout):
        tool = " ".join(argv)
        if tool in missing:
            return CommandResult(tuple(argv), None, error=f"{argv[0]!r} not found on PATH")
        return CommandResult(tuple(argv), 0, stdout=mapping.get(tool, ""), stderr="")

    return run


def _no_gpu_runner():
    return _runner(missing=("rocminfo", "amd-smi static", "rocm-smi"))


# ---------------------------------------------------------------------------
# probe_torch
# ---------------------------------------------------------------------------


def test_probe_torch_missing_dependency_is_recorded() -> None:
    def failing_importer(name: str) -> object:
        raise ImportError("No module named 'torch'")

    probe = probe_torch(None, importer=failing_importer)

    assert probe.torch_available is False
    assert probe.resolves_to == DEVICE_CPU
    assert probe.error is not None and "torch" in probe.error


def test_probe_torch_cpu_fallback_when_cuda_unavailable() -> None:
    probe = probe_torch(_cpu_torch())

    assert probe.torch_available is True
    assert probe.cuda_available is False
    assert probe.device_count == 0
    assert probe.resolves_to == DEVICE_CPU
    assert probe.device_name is None
    assert probe.gcn_arch_name is None


def test_probe_torch_detects_gfx1201_gpu() -> None:
    probe = probe_torch(_gpu_torch())

    assert probe.torch_available is True
    assert probe.cuda_available is True
    assert probe.device_count == 1
    assert probe.resolves_to == DEVICE_CUDA
    assert probe.device_name == "AMD Radeon RX 9070 XT"
    assert probe.gcn_arch_name == TARGET_GPU_ARCH
    assert probe.hip_version == "7.2.53211"


def test_probe_torch_exception_falls_back_to_cpu() -> None:
    probe = probe_torch(_fake_torch(available=True, count=1, raises=True))

    assert probe.torch_available is True
    assert probe.resolves_to == DEVICE_CPU
    assert probe.error is not None and "RuntimeError" in probe.error


# ---------------------------------------------------------------------------
# default_command_runner
# ---------------------------------------------------------------------------


def test_default_command_runner_missing_binary_is_recorded() -> None:
    result = default_command_runner(["heart-not-a-real-binary-xyz"], 1.0)

    assert result.ok is False
    assert result.error is not None and "not found on PATH" in result.error


def test_default_command_runner_records_timeout() -> None:
    result = default_command_runner(
        [sys.executable, "-c", "import time; time.sleep(5)"], 0.5
    )

    assert result.ok is False
    assert result.error is not None and "timed out" in result.error


def test_default_command_runner_empty_argv_is_recorded() -> None:
    result = default_command_runner([], 1.0)

    assert result.ok is False
    assert result.error == "empty command"


def test_default_command_runner_captures_success() -> None:
    result = default_command_runner([sys.executable, "-c", "print('gfx1201')"], 5.0)

    assert result.ok is True
    assert "gfx1201" in result.stdout


# ---------------------------------------------------------------------------
# probe_rocm_tooling
# ---------------------------------------------------------------------------


def test_probe_rocm_tooling_missing_tools_are_recorded() -> None:
    probes, architectures = probe_rocm_tooling(command_runner=_no_gpu_runner())

    assert architectures == ()
    by_name = {probe.name: probe for probe in probes}
    assert by_name["rocminfo"].status == PROBE_MISSING
    assert by_name["amd-smi static"].status == PROBE_MISSING
    assert by_name["rocm-architectures"].status == PROBE_MISSING


def test_probe_rocm_tooling_parses_gfx_tokens() -> None:
    runner = _runner(
        {
            "rocminfo": "  Name: gfx1201\n  Name: gfx12-generic\n",
            "amd-smi static": "TARGET_GRAPHICS_VERSION: gfx1201\n",
        },
        missing=("rocm-smi",),
    )
    probes, architectures = probe_rocm_tooling(command_runner=runner)

    assert TARGET_GPU_ARCH in architectures
    assert "gfx12" in architectures  # dedup/parse sanity, not just target
    by_name = {probe.name: probe for probe in probes}
    assert by_name["rocminfo"].status == PROBE_FOUND
    assert by_name["rocm-smi"].status == PROBE_MISSING
    assert by_name["rocm-architectures"].status == PROBE_FOUND


def test_probe_rocm_tooling_non_zero_exit_is_error() -> None:
    def runner(argv, timeout):
        return CommandResult(tuple(argv), 1, stdout="", stderr="boom")

    probes, architectures = probe_rocm_tooling(command_runner=runner)

    assert architectures == ()
    assert all(probe.status == PROBE_ERROR for probe in probes[:3])


def test_probe_rocm_tooling_no_gfx_token_is_skipped() -> None:
    runner = _runner({"rocminfo": "no architectures here\n"})
    probes, architectures = probe_rocm_tooling(command_runner=runner)

    assert architectures == ()
    by_name = {probe.name: probe for probe in probes}
    assert by_name["rocminfo"].status == PROBE_SKIPPED


# ---------------------------------------------------------------------------
# run_device_check — both branches
# ---------------------------------------------------------------------------


def test_run_device_check_no_gpu_branch_simulated(tmp_path) -> None:
    result = run_device_check(
        torch_module=_cpu_torch(),
        command_runner=_no_gpu_runner(),
        rocm_version_path=tmp_path / "missing-version",
    )

    assert result.gfx_target_visible is False
    assert result.torch_device == DEVICE_CPU
    assert result.cpu_fallback is True
    assert result.rocm_version is None
    assert any("CPU fallback" in warning for warning in result.warnings)
    assert result.record_for("rocm-architectures") is not None


def test_run_device_check_gpu_branch_simulated(tmp_path) -> None:
    version_file = tmp_path / "version"
    version_file.write_text("7.2.4\n", encoding="utf-8")
    runner = _runner({"rocminfo": "Name: gfx1201\n"})

    result = run_device_check(
        torch_module=_gpu_torch(),
        command_runner=runner,
        rocm_version_path=version_file,
    )

    assert result.gfx_target_visible is True
    assert result.torch_device == DEVICE_CUDA
    assert result.gpu_available is True
    assert result.cpu_fallback is False
    assert result.rocm_version == "7.2.4"
    assert result.warnings == ()


def test_run_device_check_gpu_without_rocm_tooling_still_visible(tmp_path) -> None:
    # torch seeing gfx1201 is sufficient, even when the tooling is absent.
    result = run_device_check(
        torch_module=_gpu_torch(),
        command_runner=_no_gpu_runner(),
        rocm_version_path=tmp_path / "missing",
    )

    assert result.gfx_target_visible is True
    assert result.torch_device == DEVICE_CUDA


def test_run_device_check_warns_on_torch_tooling_disagreement(tmp_path) -> None:
    # Tooling sees the target but torch resolves to CPU: a loud warning, not a
    # silent mixed verdict.
    runner = _runner({"amd-smi static": "TARGET_GRAPHICS_VERSION: gfx1201\n"})
    result = run_device_check(
        torch_module=_cpu_torch(),
        command_runner=runner,
        rocm_version_path=tmp_path / "missing",
    )

    assert result.gfx_target_visible is True
    assert result.torch_device == DEVICE_CPU
    assert any("not usable by torch" in warning for warning in result.warnings)


def test_run_device_check_missing_torch_is_documented() -> None:
    def failing_importer(name: str) -> object:
        raise ImportError("no torch")

    result = run_device_check(
        torch_importer=failing_importer,
        command_runner=_no_gpu_runner(),
    )

    assert result.torch.torch_available is False
    assert result.torch_device == DEVICE_CPU
    assert any("torch is not available" in warning for warning in result.warnings)


def test_run_device_check_rejects_empty_target() -> None:
    with pytest.raises(DeviceCheckConfigError):
        run_device_check(gfx_target="   ")


@pytest.mark.parametrize("timeout", [0, -1, "soon", True])
def test_run_device_check_rejects_bad_timeout(timeout) -> None:
    with pytest.raises(DeviceCheckConfigError):
        run_device_check(timeout_seconds=timeout)


def test_run_device_check_result_is_json_serialisable(tmp_path) -> None:
    result = run_device_check(
        torch_module=_gpu_torch(),
        command_runner=_runner({"rocminfo": "gfx1201\n"}),
        rocm_version_path=tmp_path / "missing",
    )

    payload = json.loads(json.dumps(result.to_dict()))
    assert payload["schema_version"] == DEVICE_CHECK_SCHEMA_VERSION
    assert payload["torch_device"] == DEVICE_CUDA
    assert payload["gfx_target_visible"] is True


# ---------------------------------------------------------------------------
# Report rendering / writing
# ---------------------------------------------------------------------------


def test_render_report_states_visibility_and_device(tmp_path) -> None:
    gpu = run_device_check(
        torch_module=_gpu_torch(),
        command_runner=_runner({"rocminfo": "gfx1201\n"}),
        rocm_version_path=tmp_path / "missing",
    )
    report = render_device_check_report(gpu)

    assert "gfx1201 visible: YES" in report
    assert "torch device resolved: cuda:0" in report
    assert "## Warnings" not in report

    cpu = run_device_check(
        torch_module=_cpu_torch(),
        command_runner=_no_gpu_runner(),
        rocm_version_path=tmp_path / "missing",
    )
    report = render_device_check_report(cpu)

    assert "gfx1201 visible: NO" in report
    assert "torch device resolved: cpu" in report
    assert "## Warnings" in report


def test_write_report_creates_parent_dirs_and_roundtrips(tmp_path) -> None:
    result, report, sidecar = run_and_write_device_check(
        report_path=tmp_path / "nested" / "gpu_check.md",
        json_path=tmp_path / "nested" / "gpu_check.json",
        torch_module=_gpu_torch(),
        command_runner=_runner({"rocminfo": "gfx1201\n"}),
    )

    assert report.is_file()
    assert sidecar is not None and sidecar.is_file()
    assert "gfx1201 visible: YES" in report.read_text(encoding="utf-8")
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    assert payload["gfx_target"] == TARGET_GPU_ARCH
    assert payload["torch_device"] == result.torch_device == DEVICE_CUDA


def test_write_json_roundtrips_for_cpu(tmp_path) -> None:
    result = run_device_check(
        torch_module=_cpu_torch(),
        command_runner=_no_gpu_runner(),
        rocm_version_path=tmp_path / "missing",
    )
    destination = write_device_check_json(result, tmp_path / "gpu_check.json")

    payload = json.loads(destination.read_text(encoding="utf-8"))
    assert payload["cpu_fallback"] is True
    assert payload["torch_device"] == DEVICE_CPU


def test_write_report_rejects_non_result(tmp_path) -> None:
    with pytest.raises(DeviceCheckReportError):
        write_device_check_report("not-a-result", tmp_path / "x.md")  # type: ignore[arg-type]
    with pytest.raises(DeviceCheckReportError):
        write_device_check_json(object(), tmp_path / "x.json")  # type: ignore[arg-type]
    with pytest.raises(DeviceCheckReportError):
        render_device_check_report(None)  # type: ignore[arg-type]


def test_write_report_reports_unwritable_path(tmp_path) -> None:
    result = run_device_check(
        torch_module=_cpu_torch(),
        command_runner=_no_gpu_runner(),
        rocm_version_path=tmp_path / "missing",
    )
    # A directory where the file should be makes the write fail loudly.
    blocked = tmp_path / "blocked.md"
    blocked.mkdir()

    with pytest.raises(DeviceCheckReportError):
        write_device_check_report(result, blocked)


# ---------------------------------------------------------------------------
# Small utility probes
# ---------------------------------------------------------------------------


def test_read_rocm_version_reads_and_misses(tmp_path) -> None:
    version_file = tmp_path / "version"
    version_file.write_text("7.2.4\n", encoding="utf-8")

    assert read_rocm_version(version_file) == "7.2.4"
    assert read_rocm_version(tmp_path / "nope") is None
    empty = tmp_path / "empty"
    empty.write_text("\n", encoding="utf-8")
    assert read_rocm_version(empty) is None


def test_in_container_detects_env_marker(monkeypatch) -> None:
    monkeypatch.setenv("container", "podman")
    assert in_container() is True


def test_describe_device_check_rejects_non_result() -> None:
    with pytest.raises(DeviceCheckReportError):
        describe_device_check(42)  # type: ignore[arg-type]


def test_build_parser_defaults_are_declared(tmp_path) -> None:
    args = build_parser().parse_args([])

    assert args.gfx_target == TARGET_GPU_ARCH
    assert args.json_path is None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_cli_writes_report_for_simulated_cpu(tmp_path, capsys) -> None:
    report = tmp_path / "gpu_check.md"
    sidecar = tmp_path / "gpu_check.json"

    exit_code = main(
        [
            "--report-path",
            str(report),
            "--json-path",
            str(sidecar),
            "--gfx-target",
            TARGET_GPU_ARCH,
        ],
        torch_module=_cpu_torch(),
        command_runner=_no_gpu_runner(),
    )

    captured = capsys.readouterr()
    assert exit_code == 0  # a missing GPU is documented, not a failure
    assert report.is_file() and sidecar.is_file()
    assert "torch device resolved: cpu" in report.read_text(encoding="utf-8")
    assert "cpu fallback in effect: yes" in captured.out


def test_cli_json_flag_emits_parseable_json(tmp_path, capsys) -> None:
    exit_code = main(
        [
            "--report-path",
            str(tmp_path / "gpu_check.md"),
            "--json",
        ],
        torch_module=_gpu_torch(),
        command_runner=_runner({"rocminfo": "gfx1201\n"}),
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    payload = json.loads(captured.out)
    assert payload["torch_device"] == DEVICE_CUDA
    assert payload["gfx_target_visible"] is True


def test_cli_config_error_returns_non_zero(tmp_path, capsys) -> None:
    exit_code = main(
        ["--report-path", str(tmp_path / "x.md"), "--gfx-target", ""],
        torch_module=_cpu_torch(),
        command_runner=_no_gpu_runner(),
    )

    assert exit_code == 1
    assert "device-check error" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Container artifacts (static contract)
# ---------------------------------------------------------------------------


def _repo_root() -> Path:
    # tests/ -> repository root
    return Path(__file__).resolve().parents[1]


def test_containerfile_pins_base_image_by_version_tag() -> None:
    containerfile = _repo_root() / "containers" / "Containerfile"
    text = containerfile.read_text(encoding="utf-8")

    from_lines = [line for line in text.splitlines() if line.startswith("FROM ")]
    assert len(from_lines) == 1
    assert (
        "docker.io/rocm/pytorch:rocm7.2.4_ubuntu24.04_py3.12_pytorch_release_2.10.0"
        in from_lines[0]
    )
    assert "latest" not in from_lines[0]
    # The audited digest is recorded next to the tag.
    assert "sha256:4449f856653602317e4101a76fce599c7f58cdccec2e539951fce5f73083179e" in text


def test_run_script_has_rocm_device_passthrough() -> None:
    script = _repo_root() / "containers" / "run-rocm.sh"
    text = script.read_text(encoding="utf-8")

    assert "--device=/dev/kfd" in text
    assert "--device=/dev/dri" in text
    assert "--security-opt seccomp=unconfined" in text
    assert "--group-add keep-groups" in text
    assert "reports/gpu_check.md" in text
    assert script.stat().st_mode & 0o111, "run-rocm.sh must be executable"
