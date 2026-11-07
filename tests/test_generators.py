"""Tests of the embedding scripts and the full pipeline, using tests/fakes."""
import numpy as np
import pytest

GENERATORS = [("generate_esm.py", "esm", 1280), ("generate_prottrans.py", "pt", 1024)]


@pytest.mark.parametrize("script, suffix, width", GENERATORS)
def test_saves_one_embedding_row_per_residue(run_script, make_list, install, script, suffix, width):
    result = run_script(script, "--file_list", make_list({"gen.fasta": ">g\nMKVLA\n"}))
    assert result.returncode == 0, result.stderr
    assert np.load(install / "inputs" / f"gen_{suffix}.npy").shape == (5, width)


@pytest.mark.parametrize("script, suffix, width", GENERATORS)
def test_runs_from_a_different_working_directory(run_script, make_list, install, tmp_path,
                                                 script, suffix, width):
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    result = run_script(script, "--file_list", make_list({"cwd.fasta": ">c\nMKVLA\n"}), cwd=elsewhere)
    assert result.returncode == 0, result.stderr
    assert (install / "inputs" / f"cwd_{suffix}.npy").exists()


@pytest.mark.parametrize("script, suffix, width", GENERATORS)
def test_reports_malformed_fasta_by_name(run_script, make_list, script, suffix, width):
    result = run_script(script, "--file_list", make_list({"bad.fasta": "MKVLA\n"}))
    assert result.returncode != 0
    assert "Traceback" not in result.stderr
    assert "bad.fasta" in result.stderr


def test_esm_rejects_sequences_over_its_limit_by_name(run_script, make_list):
    fasta = ">long\n" + "A" * 1023 + "\n"
    result = run_script("generate_esm.py", "--file_list", make_list({"long.fasta": fasta}))
    assert result.returncode != 0
    assert "Traceback" not in result.stderr
    assert "long.fasta" in result.stderr


def test_esm_accepts_a_sequence_at_its_limit(run_script, make_list):
    fasta = ">edge\n" + "A" * 1022 + "\n"
    result = run_script("generate_esm.py", "--file_list", make_list({"edge.fasta": fasta}))
    assert result.returncode == 0, result.stderr


def test_pipeline_produces_results(run_script, make_list, install, tmp_path):
    file_list = make_list({"pipe.fasta": ">p\nMKVLA\n"})
    result = run_script("run_SPOT-1D-LM.sh", file_list, "cpu", "cpu", "cpu", cwd=tmp_path)
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "results" / "pipe.csv").exists()


def test_pipeline_stops_at_the_first_failure(run_script, make_list, tmp_path):
    file_list = make_list({"pipe2.fasta": ">p\nMKVLA\n"})
    result = run_script("run_SPOT-1D-LM.sh", file_list, "banana", "cpu", "cpu", cwd=tmp_path)
    assert result.returncode != 0
    assert not (tmp_path / "results" / "pipe2.csv").exists()


def test_pipeline_explains_its_arguments(run_script):
    result = run_script("run_SPOT-1D-LM.sh")
    assert result.returncode != 0
    assert "usage" in result.stderr.lower()
