from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional


def snapshot(label: str, paths: Optional[Mapping[str, Path]] = None, container: str = "cortrix") -> Dict[str, object]:
    data: Dict[str, object] = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "label": label,
        "paths": {},
        "docker_stats": None,
        "docker_stats_error": None,
        "host_nvidia_smi": None,
        "host_nvidia_smi_error": None,
        "container_nvidia_smi": None,
        "container_nvidia_smi_error": None,
        "container_data_du": None,
        "container_data_du_error": None,
        "container_data_df": None,
        "container_data_df_error": None,
        "host_memory_pressure": None,
        "host_memory_pressure_error": None,
        "host_disk_io": None,
        "host_disk_io_error": None,
        "host_disk_free": {},
    }
    path_data: Dict[str, object] = {}
    for name, path in (paths or {}).items():
        path_data[name] = {
            "path": str(path),
            "bytes": du_bytes(path),
        }
    data["paths"] = path_data
    stats = run_cmd(["docker", "stats", "--no-stream", "--format", "{{json .}}", container])
    if stats["returncode"] == 0 and str(stats["stdout"]).strip():
        try:
            data["docker_stats"] = json.loads(str(stats["stdout"]).strip().splitlines()[-1])
        except json.JSONDecodeError:
            data["docker_stats"] = {"raw": str(stats["stdout"]).strip()}
    else:
        data["docker_stats_error"] = stats
    host_gpu = nvidia_smi_query()
    if host_gpu["returncode"] == 0:
        data["host_nvidia_smi"] = parse_nvidia_smi_csv(str(host_gpu["stdout"]))
    else:
        data["host_nvidia_smi_error"] = host_gpu
    container_gpu = nvidia_smi_query(container=container)
    if container_gpu["returncode"] == 0:
        data["container_nvidia_smi"] = parse_nvidia_smi_csv(str(container_gpu["stdout"]))
    else:
        data["container_nvidia_smi_error"] = container_gpu
    data_du = run_cmd(["docker", "exec", container, "du", "-sb", "/data"])
    if data_du["returncode"] == 0 and str(data_du["stdout"]).strip():
        parts = str(data_du["stdout"]).strip().split()
        data["container_data_du"] = {
            "bytes": int(parts[0]) if parts and parts[0].isdigit() else None,
            "raw": str(data_du["stdout"]).strip(),
        }
    else:
        data["container_data_du_error"] = data_du
    data_df = run_cmd(["docker", "exec", container, "df", "-k", "/data"])
    if data_df["returncode"] == 0:
        data["container_data_df"] = str(data_df["stdout"]).strip()
    else:
        data["container_data_df_error"] = data_df
    memory = run_cmd(["memory_pressure"], timeout_seconds=20.0)
    if memory["returncode"] == 0:
        data["host_memory_pressure"] = str(memory["stdout"]).strip()
    else:
        data["host_memory_pressure_error"] = memory
    disk_io, disk_io_error = host_disk_io_snapshot()
    data["host_disk_io"] = disk_io
    data["host_disk_io_error"] = disk_io_error
    disk_free: Dict[str, object] = {}
    for name, path in (paths or {}).items():
        probe_path = path if path.exists() else path.parent
        disk_free[name] = run_cmd(["df", "-k", str(probe_path)], timeout_seconds=10.0)
    data["host_disk_free"] = disk_free
    return data


def host_disk_io_snapshot(
    diskstats_path: Path = Path("/proc/diskstats"),
) -> tuple[Optional[Dict[str, object]], Optional[Dict[str, object]]]:
    iostat_path = shutil.which("iostat")
    iostat_error: Optional[Dict[str, object]] = None
    if iostat_path:
        result = run_cmd([iostat_path, "-dx", "1", "1"], timeout_seconds=15.0)
        if result["returncode"] == 0 and str(result["stdout"]).strip():
            return {
                "source": "iostat",
                "command": [iostat_path, "-dx", "1", "1"],
                "raw": str(result["stdout"]).strip(),
            }, None
        iostat_error = result

    try:
        raw_diskstats = diskstats_path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raw_diskstats = ""
        diskstats_error: Optional[str] = repr(exc)
    else:
        diskstats_error = None

    if raw_diskstats:
        data: Dict[str, object] = {
            "source": "proc_diskstats",
            "path": str(diskstats_path),
            "raw": raw_diskstats,
        }
        if iostat_error is not None:
            data["iostat_error"] = iostat_error
        return data, None

    return None, {
        "reason": "disk_io_probe_unavailable",
        "iostat_path": iostat_path,
        "iostat_error": iostat_error,
        "diskstats_path": str(diskstats_path),
        "diskstats_error": diskstats_error,
    }


def du_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    total = 0
    for dirpath, _, filenames in os.walk(path):
        for filename in filenames:
            file_path = Path(dirpath) / filename
            try:
                total += file_path.stat().st_size
            except OSError:
                pass
    return total


def nvidia_smi_query(container: Optional[str] = None) -> Dict[str, object]:
    query = [
        "nvidia-smi",
        "--query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw",
        "--format=csv,noheader,nounits",
    ]
    if container:
        return run_cmd(["docker", "exec", container, *query], timeout_seconds=20.0)
    return run_cmd(query, timeout_seconds=20.0)


def parse_nvidia_smi_csv(raw: str) -> list[Dict[str, object]]:
    rows: list[Dict[str, object]] = []
    fields = [
        "index",
        "name",
        "utilization_gpu_percent",
        "utilization_memory_percent",
        "memory_used_mib",
        "memory_total_mib",
        "power_draw_w",
    ]
    for line in raw.strip().splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != len(fields):
            rows.append({"raw": line})
            continue
        row: Dict[str, object] = {}
        for key, value in zip(fields, parts):
            row[key] = coerce_number(value) if key != "name" else value
        rows.append(row)
    return rows


def coerce_number(value: str) -> object:
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value


def run_cmd(argv: Iterable[str], timeout_seconds: float = 30.0) -> Dict[str, object]:
    try:
        proc = subprocess.run(
            list(argv),
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_seconds,
        )
        return {"returncode": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr}
    except Exception as exc:  # noqa: BLE001
        return {"returncode": -1, "stdout": "", "stderr": repr(exc)}


def append_jsonl(path: Path, item: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(item, ensure_ascii=False) + "\n")
