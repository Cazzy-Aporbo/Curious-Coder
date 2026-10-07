"""Research QC contracts for reads, samples, small variants, and assay controls."""

import argparse
from dataclasses import asdict, dataclass
import hashlib
from itertools import zip_longest
import json
import math
from pathlib import Path

from Bio import SeqIO
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class ReadPolicy:
    max_n_fraction: float = .05
    min_q30_fraction: float = .8


def read_pair_id(identifier):
    return identifier[:-2] if identifier.endswith(("/1", "/2")) else identifier


def fastq_qc(read1, read2, policy=ReadPolicy()):
    if not 0 <= policy.max_n_fraction <= 1 or not 0 <= policy.min_q30_fraction <= 1:
        raise ValueError("Read policy fractions must lie in [0, 1].")
    count = bases = unknown = q30 = gc = called = 0
    expected_errors = 0.0
    seen = set()
    lengths = []
    with Path(read1).open() as first, Path(read2).open() as second:
        records = zip_longest(SeqIO.parse(first, "fastq"), SeqIO.parse(second, "fastq"))
        for left, right in records:
            if left is None or right is None or read_pair_id(left.id) != read_pair_id(right.id):
                raise ValueError("Paired FASTQ records are missing or out of order.")
            identifier = read_pair_id(left.id)
            if identifier in seen:
                raise ValueError("Duplicate read-pair identifier.")
            seen.add(identifier)
            count += 1
            for record in (left, right):
                sequence = str(record.seq).upper()
                quality = np.asarray(record.letter_annotations["phred_quality"], dtype=float)
                if not sequence or set(sequence) - set("ACGTN"):
                    raise ValueError("This read contract accepts nonempty A/C/G/T/N sequences only.")
                lengths.append(len(sequence))
                bases += len(sequence)
                unknown += sequence.count("N")
                gc += sequence.count("G") + sequence.count("C")
                called += len(sequence) - sequence.count("N")
                q30 += int((quality >= 30).sum())
                expected_errors += float(np.power(10.0, -quality / 10).sum())
    if count == 0:
        raise ValueError("FASTQ pair is empty.")
    n_fraction, q30_fraction = unknown / bases, q30 / bases
    flags = []
    if n_fraction > policy.max_n_fraction:
        flags.append("N_FRACTION_ABOVE_POLICY")
    if q30_fraction < policy.min_q30_fraction:
        flags.append("Q30_FRACTION_BELOW_POLICY")
    return {"read_pairs": count, "bases": bases, "n_fraction": n_fraction, "q30_fraction": q30_fraction,
            "gc_fraction_called_bases": gc / called if called else None,
            "expected_base_errors_from_reported_phred": expected_errors,
            "error_probability_equivalent_phred": -10 * math.log10(expected_errors / bases),
            "min_read_length": min(lengths), "max_read_length": max(lengths), "flags": flags,
            "status": "PASS" if not flags else "REVIEW", "policy": asdict(policy)}


def sample_design_qc(frame):
    required = ["sample_id", "biological_unit", "condition", "batch"]
    if not set(required).issubset(frame.columns) or frame[required].isna().any().any():
        raise ValueError("Sample metadata is incomplete.")
    if any(frame[column].astype(str).str.strip().eq("").any() for column in required):
        raise ValueError("Sample identifiers and design fields cannot be blank.")
    if frame.empty or frame.sample_id.duplicated().any():
        raise ValueError("Require nonempty, unique sample identifiers.")
    if frame.groupby("biological_unit").condition.nunique().max() > 1:
        raise ValueError("A biological unit has contradictory conditions under this specimen-level contract.")
    design = pd.get_dummies(frame[["condition", "batch"]].astype(str), drop_first=True, dtype=float)
    matrix = np.column_stack([np.ones(len(frame)), design.to_numpy()])
    rank = int(np.linalg.matrix_rank(matrix))
    return {"samples": len(frame), "biological_units": int(frame.biological_unit.nunique()),
            "technical_replicates": int(len(frame) - frame.biological_unit.nunique()),
            "design_columns": ["intercept", *design.columns], "design_rank": rank,
            "rank_deficient": rank < matrix.shape[1],
            "condition_by_batch": pd.crosstab(frame.condition, frame.batch).to_dict(),
            "status": "REVIEW" if rank < matrix.shape[1] else "PASS"}


def variant_qc(vcf, reference_fasta, min_depth=20, balance_bounds=(.25, .75)):
    if min_depth < 1 or not 0 <= balance_bounds[0] <= balance_bounds[1] <= 1:
        raise ValueError("Invalid variant review policy.")
    reference = {record.id: str(record.seq).upper() for record in SeqIO.parse(reference_fasta, "fasta")}
    results, samples = [], None
    with Path(vcf).open() as source:
        for line in source:
            if line.startswith("##"):
                continue
            if line.startswith("#CHROM\t"):
                samples = line.rstrip().split("\t")[9:]
                if len(samples) != 1:
                    raise ValueError("This audited example supports exactly one VCF sample.")
                continue
            if not line.strip():
                continue
            if samples is None:
                raise ValueError("Missing VCF column header.")
            fields = line.rstrip().split("\t")
            if len(fields) != 10:
                raise ValueError("Expected ten columns in this single-sample VCF contract.")
            chrom, pos, _, ref, alt, _, filtering, _, layout, values = fields
            position = int(pos)
            if len(ref) != 1 or len(alt) != 1 or ref not in "ACGT" or alt not in "ACGT" or ref == alt:
                raise ValueError("Only explicit biallelic SNVs are supported; normalize other variants with established HTS tools.")
            if chrom not in reference or not 1 <= position <= len(reference[chrom]) or reference[chrom][position - 1] != ref:
                raise ValueError("Reference allele mismatch: check assembly, contig name, and 1-based position.")
            keys, entries = layout.split(":"), values.split(":")
            if len(keys) != len(entries) or len(set(keys)) != len(keys):
                raise ValueError("Malformed FORMAT fields.")
            call = dict(zip(keys, entries))
            if any(call.get(key, ".") == "." for key in ("GT", "DP", "AD")):
                raise ValueError("GT, DP, and AD are required for this review contract.")
            genotype = call["GT"].replace("|", "/").split("/")
            if len(genotype) != 2 or any(allele not in {"0", "1"} for allele in genotype):
                raise ValueError("This example requires a called diploid biallelic genotype.")
            depth = int(call["DP"])
            allele_depths = list(map(int, call["AD"].split(",")))
            if len(allele_depths) != 2 or min(depth, *allele_depths) < 0:
                raise ValueError("Depth fields must be nonnegative and allele-aligned.")
            observed = sum(allele_depths)
            balance = allele_depths[1] / observed if observed else None
            flags = []
            if filtering != "PASS":
                flags.append("FILTER_NOT_PASS")
            if depth < min_depth:
                flags.append("LOW_DEPTH")
            if observed > depth:
                flags.append("AD_EXCEEDS_DP_CHECK_CALLER_SEMANTICS")
            if set(genotype) == {"0", "1"} and (balance is None or not balance_bounds[0] <= balance <= balance_bounds[1]):
                flags.append("HET_ALLELE_BALANCE_REVIEW")
            results.append({"contig": chrom, "position_1based": position, "interval_0based_half_open": [position - 1, position],
                            "reference": ref, "alternate": alt, "sample": samples[0], "depth": depth,
                            "alt_fraction_of_allele_depths": balance, "flags": flags, "status": "PASS" if not flags else "REVIEW"})
    if samples is None or not results:
        raise ValueError("VCF contains no supported calls.")
    return results


def assay_qc(frame):
    if not {"plate", "well", "control", "signal"}.issubset(frame.columns) or frame.empty:
        raise ValueError("Assay controls require plate, well, control, and signal fields.")
    if frame.duplicated(["plate", "well"]).any() or frame[["plate", "well", "control"]].isna().any().any():
        raise ValueError("Assay wells must be unique with complete identifiers.")
    if set(frame.control) - {"positive", "negative"} or not np.isfinite(frame.signal.to_numpy(dtype=float)).all():
        raise ValueError("Use finite positive/negative control observations, not unclassified sample wells.")
    plates = []
    for name, rows in frame.groupby("plate", sort=True):
        controls = [rows.loc[rows.control == kind, "signal"].to_numpy(dtype=float) for kind in ("positive", "negative")]
        if any(len(values) < 2 for values in controls):
            raise ValueError("Each plate needs at least two observations of each control class.")
        means, deviations = [float(values.mean()) for values in controls], [float(values.std(ddof=1)) for values in controls]
        separation = abs(means[0] - means[1])
        if separation == 0:
            raise ValueError("Z-prime is undefined when control means coincide.")
        z_prime = 1 - 3 * sum(deviations) / separation
        plates.append({"plate": str(name), "control_counts": list(map(len, controls)), "control_means": means,
                       "sample_standard_deviations": deviations, "z_prime": z_prime,
                       "control_cv": [sd / abs(mean) if mean else None for sd, mean in zip(deviations, means)],
                       "status": "PASS" if z_prime >= .5 else "REVIEW", "policy_z_prime_min": .5})
    return plates


def benjamini_hochberg(p_values):
    values = np.asarray(p_values, dtype=float)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("Declare a nonempty family of finite p-values in [0, 1].")
    order = np.argsort(values, kind="stable")
    adjusted = np.minimum.accumulate((values[order] * len(values) / np.arange(1, len(values) + 1))[::-1])[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.minimum(adjusted, 1)
    return result


def run(directory, output):
    directory = Path(directory)
    inputs = [directory / name for name in ("reads_R1.fastq", "reads_R2.fastq", "samples.csv", "reference.fasta", "calls.vcf", "controls.csv")]
    report = {"fixture": "synthetic R&D acceptance/review cases, not patient or laboratory validation data",
              "inputs": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in inputs},
              "reads": fastq_qc(inputs[0], inputs[1]), "sample_design": sample_design_qc(pd.read_csv(inputs[2])),
              "variants": variant_qc(inputs[4], inputs[3]), "assays": assay_qc(pd.read_csv(inputs[5])),
              "multiple_testing_example": {"p_values": [.01, .04, .03, .002], "bh_adjusted": benjamini_hochberg([.01, .04, .03, .002]).tolist()},
              "decision_boundary": "PASS means these configured checks passed. It is not specimen identity, pathogenicity, assay validation, or compliance certification."}
    decisions = [report["reads"], report["sample_design"], *report["variants"], *report["assays"]]
    report["overall_status"] = "REVIEW" if any(item["status"] == "REVIEW" for item in decisions) else "PASS"
    report["implementation_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "data/biotech_fixture")
    parser.add_argument("--output", type=Path, default=ROOT / "studies/results/biotech_qc.json")
    parser.add_argument("--fail-on-review", action="store_true", help="Write the report, then exit 2 if any configured check needs review.")
    args = parser.parse_args()
    report = run(args.input, args.output)
    if args.fail_on_review and report["overall_status"] == "REVIEW":
        raise SystemExit(2)
