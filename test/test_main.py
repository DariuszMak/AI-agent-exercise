import pathlib

import pytest

from src.main import run_conversion

TEST_CASES_IDS = range(1, 4 + 1)


def read(p: pathlib.Path) -> str:
    return p.read_text().replace("\r\n", "\n").strip()


@pytest.mark.parametrize("example_id", TEST_CASES_IDS)
def test_converted_file(tmp_path: pathlib.Path, example_id: int) -> None:
    examples = pathlib.Path("test", "examples", str(example_id))
    output = pathlib.Path("output")
    source = pathlib.Path("source")

    input_file = examples / source / "input_timesheet.csv"
    expected_converted = examples / output / "expected_converted.csv"

    tmp_input = tmp_path / "input.txt"
    tmp_input.write_text(read(input_file))

    converted_path, _ = run_conversion(str(tmp_input))

    actual_converted = read(pathlib.Path(converted_path))
    assert actual_converted == read(expected_converted)


@pytest.mark.parametrize("example_id", TEST_CASES_IDS)
def test_pivot_file(tmp_path: pathlib.Path, example_id: int) -> None:
    examples = pathlib.Path("test", "examples", str(example_id))
    output = pathlib.Path("output")
    source = pathlib.Path("source")

    input_file = examples / source / "input_timesheet.csv"
    expected_pivot = examples / output / "expected_pivot.csv"

    tmp_input = tmp_path / "input.txt"
    tmp_input.write_text(read(input_file))

    _, pivot_path = run_conversion(str(tmp_input))

    actual_pivot = read(pathlib.Path(pivot_path))
    assert actual_pivot == read(expected_pivot)
