"""Negative-path tests for the S08/T03 dataset integrity gate.

``scripts/verify_dataset.py`` is the abort mechanism that stops the
reproduction pipeline before training when the data is not what the pinned
digests say it is. These tests exercise each failure surface with inline
tampered fixtures in pytest's tmp tree (never a .gitignore'd project path):

* tampered raw CSV bytes      -> exit 2, clear checksum message;
* missing raw CSV             -> exit 2 (no silent download);
* split artifact drift        -> exit 3, the split manifest is enforced;
* clean tracked data          -> exit 0 (the positive control).
"""

from __future__ import annotations

import importlib.util
import shutil
from pathlib import Path

import pytest

VDS_PATH = Path(__file__).resolve().parents[1] / "scripts" / "verify_dataset.py"


@pytest.fixture()
def verify_dataset():
    spec = importlib.util.spec_from_file_location("verify_dataset_for_test", VDS_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run(verify_dataset, monkeypatch, tmp_path: Path, *, raw: bool, split: bool):
    """Run the gate with its data roots retargeted into tmp_path."""
    import heart.data.load as load_mod
    import heart.data.split as split_mod

    raw_dir, splits_root = tmp_path / "raw", tmp_path / "splits"
    if raw:
        raw_dir.mkdir(parents=True)
        shutil.copyfile(
            load_mod.RAW_DATA_DIR / load_mod.RAW_FILENAME, raw_dir / load_mod.RAW_FILENAME
        )
    split_root = splits_root / "v1"
    if split:
        src = Path(split_mod.PROCESSED_DATA_DIR) / "splits" / "v1"
        split_root.mkdir(parents=True)
        for name in ("train.csv", "test.csv", "manifest.json"):
            shutil.copyfile(src / name, split_root / name)
    monkeypatch.setattr(load_mod, "RAW_DATA_DIR", raw_dir)
    monkeypatch.setattr(split_mod, "PROCESSED_DATA_DIR", tmp_path)
    return verify_dataset.main([])


def test_gate_passes_on_clean_tracked_data(verify_dataset, monkeypatch, tmp_path):
    rc = _run(verify_dataset, monkeypatch, tmp_path, raw=True, split=True)
    assert rc == 0


def test_gate_aborts_on_tampered_raw_bytes(verify_dataset, monkeypatch, tmp_path):
    import heart.data.load as load_mod

    rc = _run(verify_dataset, monkeypatch, tmp_path, raw=True, split=True)
    assert rc == 0
    # Tamper with('append one byte to) the copy the gate now watches.
    target = load_mod.RAW_DATA_DIR / load_mod.RAW_FILENAME
    target.write_bytes(target.read_bytes() + b"\n")
    rc = verify_dataset.main([])
    assert rc == 2


def test_gate_aborts_without_silent_download_when_raw_missing(
    verify_dataset, monkeypatch, tmp_path
):
    monkeypatch.chdir(str(tmp_path))  # prove cwd does not rescue the gate
    rc = _run(verify_dataset, monkeypatch, tmp_path, raw=False, split=True)
    assert rc == 2


def test_gate_aborts_on_split_manifest_drift(verify_dataset, monkeypatch, tmp_path):
    import heart.data.split as split_mod

    assert _run(verify_dataset, monkeypatch, tmp_path, raw=True, split=True) == 0
    test_csv = Path(split_mod.PROCESSED_DATA_DIR) / "splits" / "v1" / "test.csv"
    test_csv.write_text(test_csv.read_text() + "40,M,ATA,140,1,0,Normal,100,N,1.0,Up,0\n")
    rc = verify_dataset.main([])
    assert rc == 3
