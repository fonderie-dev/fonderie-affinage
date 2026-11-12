import json

import pytest

from fonderie_affinage import record as rec


def test_record_round_trip_and_verification(tmp_path) -> None:
    output = tmp_path / "result.txt"
    output.write_text("a result\n", encoding="utf-8")
    record = rec.new_record("training", config={"seed": 1}, command=["affinage", "train"])
    record["base_model"] = {"id": "none"}
    record["data"] = {"note": "none"}
    path = rec.finish_record(record, tmp_path, [output])

    stored = json.loads(path.read_text(encoding="utf-8"))
    assert stored["schema"] == rec.SCHEMA
    assert stored["status"] == "complete"
    assert stored["outputs"] == [rec.file_entry(output, relative_to=tmp_path)]
    assert stored["software"]["packages"]["pytest"]
    assert rec.verify_outputs(path) == []

    output.write_text("changed\n", encoding="utf-8")
    assert rec.verify_outputs(path) == ["checksum differs: result.txt"]
    output.unlink()
    assert rec.verify_outputs(path) == ["missing: result.txt"]


def test_incomplete_record_is_rejected() -> None:
    record = rec.new_record("training", config={}, command=[])
    del record["seeds"]
    with pytest.raises(rec.RecordError, match="seeds"):
        rec.validate_record(record)


def test_parent_reference_names_the_record_file(tmp_path) -> None:
    record = rec.new_record("conversion", config={}, command=[])
    path = rec.finish_record(record, tmp_path, [])
    parent = rec.parent_reference(path)
    assert parent["record_id"] == record["record_id"]
    assert parent["sha256"] == rec.sha256_file(path)


def test_run_directories_are_never_reused(tmp_path, monkeypatch) -> None:
    first = rec.new_run_dir(tmp_path, "run")
    assert first.is_dir()

    class Frozen:
        @staticmethod
        def now(_tz):
            from datetime import datetime

            return datetime.strptime(first.name.split("_")[0], "%Y%m%dT%H%M%SZ")

    monkeypatch.setattr(rec, "datetime", Frozen)
    with pytest.raises(FileExistsError):
        rec.new_run_dir(tmp_path, "run")
