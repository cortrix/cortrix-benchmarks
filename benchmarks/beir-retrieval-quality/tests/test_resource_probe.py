from __future__ import annotations

import sys
from pathlib import Path

RUNNER_DIR = Path(__file__).resolve().parents[1] / "runner"
sys.path.insert(0, str(RUNNER_DIR))

import resource_probe  # noqa: E402


def test_host_disk_io_prefers_iostat(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(resource_probe.shutil, "which", lambda _name: "/usr/bin/iostat")
    monkeypatch.setattr(
        resource_probe,
        "run_cmd",
        lambda argv, timeout_seconds=30.0: {
            "returncode": 0,
            "stdout": "Device r/s w/s\nnvme0n1 1.0 2.0\n",
            "stderr": "",
        },
    )

    data, error = resource_probe.host_disk_io_snapshot(tmp_path / "missing-diskstats")

    assert error is None
    assert data is not None
    assert data["source"] == "iostat"
    assert data["command"] == ["/usr/bin/iostat", "-dx", "1", "1"]
    assert "nvme0n1" in str(data["raw"])


def test_host_disk_io_falls_back_to_proc_diskstats(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(resource_probe.shutil, "which", lambda _name: None)
    diskstats = tmp_path / "diskstats"
    diskstats.write_text("259 0 nvme0n1 10 0 20 0 30 0 40 0 0 50 60\n", encoding="utf-8")

    data, error = resource_probe.host_disk_io_snapshot(diskstats)

    assert error is None
    assert data is not None
    assert data["source"] == "proc_diskstats"
    assert data["path"] == str(diskstats)
    assert "nvme0n1" in str(data["raw"])


def test_host_disk_io_falls_back_when_iostat_fails(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(resource_probe.shutil, "which", lambda _name: "/usr/sbin/iostat")
    monkeypatch.setattr(
        resource_probe,
        "run_cmd",
        lambda argv, timeout_seconds=30.0: {
            "returncode": 1,
            "stdout": "",
            "stderr": "unsupported option",
        },
    )
    diskstats = tmp_path / "diskstats"
    diskstats.write_text("259 0 nvme0n1 10 0 20 0 30 0 40 0 0 50 60\n", encoding="utf-8")

    data, error = resource_probe.host_disk_io_snapshot(diskstats)

    assert error is None
    assert data is not None
    assert data["source"] == "proc_diskstats"
    assert data["iostat_error"]["returncode"] == 1


def test_host_disk_io_reports_structured_unavailable_error(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(resource_probe.shutil, "which", lambda _name: None)
    diskstats = tmp_path / "missing-diskstats"

    data, error = resource_probe.host_disk_io_snapshot(diskstats)

    assert data is None
    assert error is not None
    assert error["reason"] == "disk_io_probe_unavailable"
    assert error["iostat_path"] is None
    assert error["diskstats_path"] == str(diskstats)
    assert "FileNotFoundError" in str(error["diskstats_error"])
