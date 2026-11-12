"""Run record: the provenance file written next to the outputs of every run.

Schema ``fonderie.run-record/1``. The module uses the standard library only, so
it can be imported without the training extras and moved into a package of its
own once another atelier needs it.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import platform
import subprocess
import sys
import uuid
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

SCHEMA = "fonderie.run-record/1"
RECORD_NAME = "record.json"
_CHUNK = 4 * 1024 * 1024

REQUIRED_FIELDS = (
    "schema",
    "record_id",
    "kind",
    "status",
    "parents",
    "code",
    "command",
    "config",
    "base_model",
    "data",
    "seeds",
    "software",
    "hardware",
    "time",
    "outputs",
)


class RecordError(ValueError):
    """A run record is incomplete or does not match the files it describes."""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(_CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def file_entry(path: Path, relative_to: Path | None = None) -> dict[str, Any]:
    """Name, size and SHA-256 of one file; the name is relative when a root is given."""
    path = Path(path)
    name = path.relative_to(relative_to).as_posix() if relative_to else path.name
    return {"path": name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def weight_metadata(**fields: Any) -> dict[str, str]:
    """Metadata for a safetensors file as one key with sorted JSON.

    safetensors does not keep the order of several metadata keys, which would
    make the bytes of otherwise identical files differ.
    """
    return {"fonderie": json.dumps(fields, sort_keys=True)}


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def code_state(package: str, source_dir: Path | None = None) -> dict[str, Any]:
    """Package version and, when the source is a git checkout, its commit and dirty flag."""
    state: dict[str, Any] = {"package": package, "version": None, "commit": None, "dirty": None}
    with contextlib.suppress(metadata.PackageNotFoundError):
        state["version"] = metadata.version(package)
    source_dir = source_dir or Path(__file__).resolve().parent
    try:
        commit = _git(source_dir, "rev-parse", "HEAD")
        status = _git(source_dir, "status", "--porcelain", "--untracked-files=no")
    except (OSError, subprocess.SubprocessError):
        return state
    state["commit"] = commit
    state["dirty"] = bool(status)
    return state


def _git(cwd: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, check=True, timeout=30
    )
    return result.stdout.strip()


def software_state() -> dict[str, Any]:
    """Interpreter, platform and the version of every installed distribution."""
    packages = {}
    for dist in metadata.distributions():
        name = dist.metadata["Name"]
        if name:
            packages[name.lower()] = dist.version
    state: dict[str, Any] = {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "packages": dict(sorted(packages.items())),
    }
    torch = sys.modules.get("torch")
    if torch is not None:
        state["torch"] = {
            "version": torch.__version__,
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
        }
    return state


def hardware_state() -> dict[str, Any]:
    """Processor, memory and, when torch sees them, the accelerators."""
    state: dict[str, Any] = {
        "machine": platform.machine(),
        "cpu": _cpu_model(),
        "cpu_count": os.cpu_count(),
        "memory_bytes": _memory_bytes(),
        "accelerators": [],
    }
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            state["accelerators"].append(
                {"index": index, "name": props.name, "memory_bytes": props.total_memory}
            )
    return state


def _cpu_model() -> str | None:
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or None


def _memory_bytes() -> int | None:
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        return None


def new_record(kind: str, *, config: dict[str, Any], command: list[str]) -> dict[str, Any]:
    """Start a record. The caller fills base model, data, seeds and outputs, then finishes it."""
    return {
        "schema": SCHEMA,
        "record_id": uuid.uuid4().hex,
        "kind": kind,
        "status": "running",
        "parents": [],
        "code": code_state("fonderie-affinage"),
        "command": list(command),
        "config": config,
        "base_model": None,
        "data": None,
        "seeds": {},
        "software": software_state(),
        "hardware": hardware_state(),
        "time": {"start_utc": utc_now(), "end_utc": None},
        "outputs": [],
    }


def parent_reference(record_path: Path) -> dict[str, str]:
    """Reference to an earlier record: its id and the checksum of the record file."""
    record = json.loads(Path(record_path).read_text(encoding="utf-8"))
    return {
        "record_id": record["record_id"],
        "kind": record["kind"],
        "sha256": sha256_file(record_path),
    }


def finish_record(record: dict[str, Any], run_dir: Path, outputs: list[Path]) -> Path:
    """Checksum the outputs, close the record and write it atomically into the run directory."""
    run_dir = Path(run_dir)
    record["outputs"] = [file_entry(path, relative_to=run_dir) for path in sorted(outputs)]
    record["time"]["end_utc"] = utc_now()
    record["status"] = "complete"
    validate_record(record)
    target = run_dir / RECORD_NAME
    temporary = run_dir / (RECORD_NAME + ".tmp")
    temporary.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, target)
    return target


def validate_record(record: dict[str, Any]) -> None:
    """Raise if a record lacks a required field or names the wrong schema."""
    missing = [field for field in REQUIRED_FIELDS if field not in record]
    if missing:
        raise RecordError(f"Run record lacks: {', '.join(missing)}")
    if record["schema"] != SCHEMA:
        raise RecordError(f"Unknown run-record schema: {record['schema']!r}")
    if record["status"] == "complete" and not record["time"].get("end_utc"):
        raise RecordError("A complete record needs an end time.")


def verify_outputs(record_path: Path) -> list[str]:
    """Compare every output listed in a record with the file on disk; return the mismatches."""
    record_path = Path(record_path)
    record = json.loads(record_path.read_text(encoding="utf-8"))
    validate_record(record)
    problems = []
    for entry in record["outputs"]:
        path = record_path.parent / entry["path"]
        if not path.is_file():
            problems.append(f"missing: {entry['path']}")
        elif sha256_file(path) != entry["sha256"]:
            problems.append(f"checksum differs: {entry['path']}")
    return problems


def new_run_dir(output_root: Path, name: str) -> Path:
    """Create a run directory that did not exist before. Runs are never overwritten."""
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = Path(output_root) / f"{stamp}_{name}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir
