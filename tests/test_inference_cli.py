"""End-to-end tests of `python run_inference.py ...`."""
import shutil

import numpy as np
import pandas as pd
import pytest

COLUMNS = [
    "AA", "SS3", "SS8", "ASA", "HseU", "HseD", "CN", "Psi", "Phi", "Theta", "Tau",
    "P3C", "P3E", "P3H", "P8C", "P8S", "P8T", "P8H", "P8G", "P8I", "P8E", "P8B",
]
P3 = ["P3C", "P3E", "P3H"]
P8 = ["P8C", "P8S", "P8T", "P8H", "P8G", "P8I", "P8E", "P8B"]
SIMPLE = {"prot.fasta": ">prot\nMKVLA\n"}


def read_csv(path):
    return pd.read_csv(path, index_col=0, keep_default_na=False)


@pytest.fixture
def simple_list(make_list, write_embeddings):
    write_embeddings("prot", 5)
    return make_list(SIMPLE)


def test_writes_one_row_per_residue_with_documented_columns(run_script, simple_list, tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    result = run_script("run_inference.py", "--file_list", simple_list, "--save_path", out)
    assert result.returncode == 0, result.stderr
    table = read_csv(out / "prot.csv")
    assert list(table.columns) == COLUMNS
    assert "".join(table["AA"]) == "MKVLA"


def test_probability_columns_are_probabilities(run_script, simple_list, tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    result = run_script("run_inference.py", "--file_list", simple_list, "--save_path", out)
    assert result.returncode == 0, result.stderr
    table = read_csv(out / "prot.csv")
    assert np.allclose(table[P3].sum(axis=1), 1, atol=1e-4)
    assert np.allclose(table[P8].sum(axis=1), 1, atol=1e-4)
    assert (table[P3 + P8] >= 0).all().all()


def test_predicted_state_is_the_most_probable_one(run_script, simple_list, tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    run_script("run_inference.py", "--file_list", simple_list, "--save_path", out)
    table = read_csv(out / "prot.csv")
    assert (table[P3].idxmax(axis=1).str[-1] == table["SS3"]).all()
    assert (table[P8].idxmax(axis=1).str[-1] == table["SS8"]).all()


def test_creates_missing_save_directory(run_script, simple_list, tmp_path):
    out = tmp_path / "does" / "not" / "exist"
    result = run_script("run_inference.py", "--file_list", simple_list, "--save_path", out)
    assert result.returncode == 0, result.stderr
    assert (out / "prot.csv").exists()


def test_runs_from_a_different_working_directory(run_script, simple_list, tmp_path):
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    out = tmp_path / "out"
    out.mkdir()
    result = run_script("run_inference.py", "--file_list", simple_list, "--save_path", out,
                        cwd=elsewhere)
    assert result.returncode == 0, result.stderr
    assert (out / "prot.csv").exists()


def test_checkpoints_dir_option_finds_checkpoints_elsewhere(run_script, simple_list, install, tmp_path):
    moved = tmp_path / "weights"
    shutil.copytree(install / "checkpoints", moved)
    out = tmp_path / "out"
    result = run_script("run_inference.py", "--file_list", simple_list, "--save_path", out,
                        "--checkpoints_dir", moved)
    assert result.returncode == 0, result.stderr
    assert (out / "prot.csv").exists()


@pytest.mark.parametrize("sequence", ["MKBZUOJXA", "mkvla", "MKVLA*"])
def test_accepts_unusual_residues(run_script, make_list, write_embeddings, tmp_path, sequence):
    expected = sequence.rstrip("*").upper()
    write_embeddings("odd", len(expected))
    out = tmp_path / "out"
    result = run_script("run_inference.py", "--save_path", out,
                        "--file_list", make_list({"odd.fasta": f">odd\n{sequence}\n"}))
    assert result.returncode == 0, result.stderr
    assert "".join(read_csv(out / "odd.csv")["AA"]) == expected


def test_keeps_dots_in_names(run_script, make_list, write_embeddings, tmp_path):
    write_embeddings("P1.v1", 5)
    write_embeddings("P1.v2", 5)
    records = {"P1.v1.fasta": ">a\nMKVLA\n", "P1.v2.fasta": ">b\nMKVLA\n"}
    out = tmp_path / "out"
    result = run_script("run_inference.py", "--file_list", make_list(records), "--save_path", out)
    assert result.returncode == 0, result.stderr
    assert sorted(p.name for p in out.iterdir()) == ["P1.v1.csv", "P1.v2.csv"]


def test_ignores_blank_lines_in_file_list(run_script, make_list, write_embeddings, tmp_path):
    write_embeddings("prot", 5)
    out = tmp_path / "out"
    result = run_script("run_inference.py", "--save_path", out,
                        "--file_list", make_list(SIMPLE, trailer="\n  \n"))
    assert result.returncode == 0, result.stderr
    assert (out / "prot.csv").exists()


def assert_clean_failure(result, message):
    assert result.returncode != 0
    assert "Traceback" not in result.stderr
    assert message in result.stderr


def test_rejects_unknown_device(run_script, simple_list):
    result = run_script("run_inference.py", "--file_list", simple_list, "--device", "banana")
    assert_clean_failure(result, "banana")


def test_requires_file_list(run_script):
    assert_clean_failure(run_script("run_inference.py"), "--file_list")


def test_reports_missing_checkpoints_with_pointer_to_readme(run_script, simple_list, tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    result = run_script("run_inference.py", "--file_list", simple_list, "--checkpoints_dir", empty)
    assert_clean_failure(result, "README")


def test_reports_missing_embeddings(run_script, make_list):
    result = run_script("run_inference.py",
                        "--file_list", make_list({"noembed.fasta": ">x\nMKVLA\n"}))
    assert_clean_failure(result, "noembed_esm.npy")


def test_reports_embeddings_from_a_different_sequence(run_script, make_list, write_embeddings):
    write_embeddings("stale", 9)
    result = run_script("run_inference.py", "--file_list", make_list({"stale.fasta": ">x\nMKVLA\n"}))
    assert_clean_failure(result, "stale")


@pytest.mark.parametrize(
    "text, reason",
    [
        ("MKVLA\n", "header"),
        (">only_a_header\n", "no sequence"),
        (">a\nMKV\n>b\nLAA\n", "one sequence"),
    ],
)
def test_reports_malformed_fasta_by_name(run_script, make_list, text, reason):
    result = run_script("run_inference.py", "--file_list", make_list({"bad.fasta": text}))
    assert_clean_failure(result, "bad.fasta")
    assert reason in result.stderr
