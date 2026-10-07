import numpy as np
import pandas as pd
import pytest

from biotech.qc import ROOT, ReadPolicy, assay_qc, benjamini_hochberg, fastq_qc, sample_design_qc, variant_qc


FIXTURE = ROOT / "data/biotech_fixture"


def test_fastq_pair_and_phred_error_arithmetic():
    report = fastq_qc(FIXTURE / "reads_R1.fastq", FIXTURE / "reads_R2.fastq")
    assert report["read_pairs"] == 2 and report["bases"] == 48
    assert report["expected_base_errors_from_reported_phred"] == pytest.approx(48 * 1e-4)
    assert report["error_probability_equivalent_phred"] == pytest.approx(40)
    assert report["status"] == "PASS"


def test_bad_read_quality_is_reviewed_not_silently_discarded(tmp_path):
    for mate in (1, 2):
        (tmp_path / f"R{mate}.fastq").write_text(f"@read/{mate}\nNNAC\n+\n!!!!\n")
    report = fastq_qc(tmp_path / "R1.fastq", tmp_path / "R2.fastq", ReadPolicy())
    assert report["status"] == "REVIEW"
    assert report["n_fraction"] == .5
    assert report["q30_fraction"] == 0
    assert report["expected_base_errors_from_reported_phred"] == 8


def test_orphan_reads_and_reference_mismatch_fail(tmp_path):
    empty = tmp_path / "empty.fastq"
    empty.write_text("")
    with pytest.raises(ValueError, match="Paired"):
        fastq_qc(FIXTURE / "reads_R1.fastq", empty)
    reference = tmp_path / "wrong.fasta"
    reference.write_text(">synthetic-contig\nTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT\n")
    with pytest.raises(ValueError, match="Reference allele mismatch"):
        variant_qc(FIXTURE / "calls.vcf", reference)


def test_replicates_are_not_counted_as_independent_biology():
    report = sample_design_qc(pd.read_csv(FIXTURE / "samples.csv"))
    assert report["samples"] == 5 and report["biological_units"] == 4
    assert report["technical_replicates"] == 1 and not report["rank_deficient"]


def test_condition_batch_confounding_is_detected():
    frame = pd.DataFrame({"sample_id": ["s1", "s2", "s3", "s4"], "biological_unit": ["u1", "u2", "u3", "u4"],
                          "condition": ["control", "control", "case", "case"], "batch": ["A", "A", "B", "B"]})
    assert sample_design_qc(frame)["rank_deficient"]


def test_variant_coordinates_and_filter_semantics():
    records = variant_qc(FIXTURE / "calls.vcf", FIXTURE / "reference.fasta")
    assert records[0]["interval_0based_half_open"] == [0, 1]
    assert records[0]["alt_fraction_of_allele_depths"] == pytest.approx(.52)
    assert records[0]["status"] == "PASS"
    assert "FILTER_NOT_PASS" in records[1]["flags"]
    assert "HET_ALLELE_BALANCE_REVIEW" in records[1]["flags"]


def test_assay_controls_use_sample_sd_and_separate_plates():
    reports = assay_qc(pd.read_csv(FIXTURE / "controls.csv"))
    assert reports[0]["z_prime"] == pytest.approx(1 - 3 * (np.std([98, 100, 102, 100], ddof=1) + np.std([4, 5, 6, 5], ddof=1)) / 95)
    assert reports[0]["status"] == "PASS" and reports[1]["status"] == "REVIEW"


def test_undefined_assay_and_malformed_pvalues_are_rejected():
    frame = pd.read_csv(FIXTURE / "controls.csv")
    frame["signal"] = 1
    with pytest.raises(ValueError, match="undefined"):
        assay_qc(frame)
    with pytest.raises(ValueError):
        benjamini_hochberg([.01, np.nan])


def test_numeric_batch_codes_are_categorical_not_a_linear_trend():
    frame = pd.DataFrame({"sample_id": list("abcdef"), "biological_unit": list("abcdef"),
                          "condition": [0, 1, 0, 1, 0, 1], "batch": [1, 1, 2, 2, 3, 3]})
    report = sample_design_qc(frame)
    assert len(report["design_columns"]) == 4
    assert not report["rank_deficient"]


def test_review_exit_status_preserves_a_machine_readable_report(tmp_path):
    import json
    import subprocess
    import sys

    output = tmp_path / "qc.json"
    process = subprocess.run([sys.executable, "-m", "biotech.qc", "--fail-on-review", "--output", str(output)],
                             cwd=ROOT, capture_output=True, text=True, timeout=30)
    assert process.returncode == 2
    assert json.loads(output.read_text())["overall_status"] == "REVIEW"


def test_multiple_testing_restores_original_feature_order():
    np.testing.assert_allclose(benjamini_hochberg([.01, .04, .03, .002]), [.02, .04, .04, .008])
