import json

import pytest

from support_eval.ingestion import InputError, load_rubric


def _criterion(identifier="quality"):
    return {
        "id": identifier,
        "name": "Quality",
        "description": "Checks quality",
        "weight": 1,
        "pass_threshold": 0.5,
        "keywords": ["good"],
    }


def test_loads_valid_rubric(tmp_path):
    path = tmp_path / "rubric.json"
    path.write_text(json.dumps({"name": "Test", "criteria": [_criterion()]}), encoding="utf-8")
    rubric = load_rubric(path)
    assert rubric.name == "Test"
    assert rubric.criteria[0].keywords == ["good"]


def test_rejects_duplicate_criterion_ids(tmp_path):
    path = tmp_path / "rubric.json"
    path.write_text(
        json.dumps({"name": "Test", "criteria": [_criterion(), _criterion()]}),
        encoding="utf-8",
    )
    with pytest.raises(InputError, match="criterion ids must be unique"):
        load_rubric(path)


def test_rejects_out_of_range_threshold(tmp_path):
    criterion = _criterion()
    criterion["pass_threshold"] = 2
    path = tmp_path / "rubric.json"
    path.write_text(json.dumps({"name": "Test", "criteria": [criterion]}), encoding="utf-8")
    with pytest.raises(InputError, match="Invalid rubric"):
        load_rubric(path)
