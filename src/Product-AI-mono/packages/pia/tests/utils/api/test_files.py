import json
import shutil
from pathlib import Path

import pytest
from pia.utils.api.files import (
    load_json,
    load_yaml,
    make_directory,
    read_jsonl,
    save_json,
    save_yaml,
)


def test_save_and_load_json(tmp_path: Path):
    data = {"name": "test", "value": 123}
    file_path = tmp_path / "test.json"

    save_json(data, file_path)
    assert file_path.exists()

    loaded = load_json(file_path)
    assert loaded == data

    shutil.rmtree(tmp_path)
    assert not tmp_path.exists()


def test_save_json_with_directory_creation(tmp_path: Path):
    data = {"hello": "world"}
    nested_dir = tmp_path / "nested" / "dir"
    file_path = nested_dir / "file.json"

    save_json(data, file_path, create_directory=True)
    assert file_path.exists()
    assert load_json(file_path) == data

    shutil.rmtree(tmp_path)
    assert not tmp_path.exists()


def test_load_json_file_not_found(tmp_path: Path):
    file_path = tmp_path / "not_exist.json"
    with pytest.raises(FileNotFoundError):
        load_json(file_path)


def test_save_and_load_yaml(tmp_path: Path):
    data = {"items": [1, 2, 3], "flag": True}
    file_path = tmp_path / "test.yaml"

    save_yaml(data, file_path)
    assert file_path.exists()

    loaded = load_yaml(file_path)
    assert loaded == data

    shutil.rmtree(tmp_path)
    assert not tmp_path.exists()


def test_save_yaml_with_directory_creation(tmp_path: Path):
    data = {"key": "value"}
    nested_dir = tmp_path / "nested" / "yaml"
    file_path = nested_dir / "file.yaml"

    save_yaml(data, file_path, create_directory=True)
    assert file_path.exists()
    assert load_yaml(file_path) == data

    shutil.rmtree(tmp_path)
    assert not tmp_path.exists()


def test_load_yaml_file_not_found(tmp_path: Path):
    file_path = tmp_path / "not_exist.yaml"
    with pytest.raises(FileNotFoundError):
        load_yaml(file_path)


def test_make_directory(tmp_path: Path):
    new_dir = tmp_path / "created_dir"
    make_directory(new_dir)
    assert new_dir.exists()
    assert new_dir.is_dir()

    shutil.rmtree(tmp_path)
    assert not tmp_path.exists()


def test_read_jsonl(tmp_path: Path):
    file_path = tmp_path / "data.jsonl"
    data = [{"id": 1, "text": "hello"}, {"id": 2, "text": "world"}]

    with open(file_path, "w", encoding="utf-8") as f:
        for row in data:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    result = read_jsonl(file_path)
    assert result == data

    shutil.rmtree(tmp_path)
    assert not tmp_path.exists()
