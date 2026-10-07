# Biotech Python: the checks I put before the model

I start with the observation, not the estimator. A sequencing read carries an encoding convention. A variant carries an assembly and coordinate system. An assay value carries a plate, a control definition, and a unit of replication. If I lose one of those meanings while cleaning a table, a more sophisticated model will not recover it.

This section is an executable R&D quality-control workflow. I use small synthetic fixtures to make acceptance and review cases unambiguous; I use public measured data and pretrained weights in the linked studies. I do not present synthetic fixtures as laboratory validation evidence.

```bash
python -m pip install -r requirements-torch.txt
python -m biotech.qc
python -m pytest tests/test_biotech_qc.py -v
```

The command reads [the fixture files](../data/biotech_fixture/samples.csv), checks six input files, records their SHA-256 hashes, and writes [a structured report](../studies/results/biotech_qc.json). To inspect compatible files of your own, pass `--input PATH --output PATH/report.json`. Add `--fail-on-review` to write the report and then return exit status 2 when any configured check requires review; without that flag, successful report generation returns 0. The fixture intentionally requires review. Keep patient records and restricted laboratory data out of this public repository.

## 1. Read quality: encoding is part of the data

I use Biopython's FASTQ parser rather than splitting arbitrary text into four-line chunks. The declared input is **Phred+33 FASTQ**, paired in the same order. I reject malformed records, orphan mates, duplicate pair identifiers, empty input, and symbols outside A/C/G/T/N.

For a reported Phred score Q, the implied base-call error probability is:

```text
p_error = 10^(−Q / 10)
expected number of base errors = sum(p_error)
equivalent aggregate Q = −10 log10(mean(p_error))
```

That last expression is not the arithmetic mean of the quality scores. Averaging a logarithmic score and then treating it as an average error probability changes the calculation. The expected-error sum uses linearity of expectation; it does not require independent errors, although the reported probabilities must be meaningful.

The report includes Q30 fraction, N fraction, read-length range, GC fraction among called bases, and expected base errors. My example policy flags N fraction above 5% or Q30 fraction below 80% for review. Those are **declared teaching thresholds**, not universal sequencing acceptance standards. Platform, library type, assay purpose, and validation data determine a defensible operating policy.

High Q scores do not establish sample identity, absence of contamination, correct alignment, or sufficient coverage of a target. For a real sequencing workflow I would add adapter/content diagnostics, reference-aware alignment and coverage checks, duplicate handling appropriate to the assay, contamination assessment, and run-level controls. The parser iterates over reads, but duplicate-ID tracking grows with the number of IDs; very large runs need partitioned or disk-backed tracking.

**Reference:** [NCBI SRA file-format guide](https://www.ncbi.nlm.nih.gov/sra/docs/submitformats/) describes FASTQ encoding variants and sequence/quality-length requirements.

## 2. Sample integrity: count biology separately from files

The sample table requires `sample_id`, `biological_unit`, `condition`, and `batch`. Under this specimen-level contract, sample IDs are unique and a biological unit cannot have contradictory conditions. A longitudinal or crossover study would need a different observation key, such as participant × visit × condition; I would not force it into this schema.

The fixture contains five sample rows but **four biological units**. One is a technical replicate. Treating all five as independent would overstate the amount of biological evidence. The same distinction affects train/test splitting, bootstrap units, mixed-effects models, and interpretation of uncertainty.

I also construct an intercept-plus-condition-plus-batch design matrix and inspect its rank. If every case was processed in one batch and every control in another, condition and batch are inseparable under that design. A batch-correction package cannot identify the missing contrast from those data alone. The appropriate next action is to reconsider the experiment or obtain bridging observations, not to search for an algorithm that produces a cleaner plot.

For a working laboratory interface, I would extend the schema with assay version, specimen derivation, extraction/library identifiers, collection and processing times, operator/instrument references, consent/use restrictions, and controlled terminology. I would validate those against the laboratory's actual data dictionary instead of inventing a universal LIMS schema.

## 3. Variant QC: a base is not a diagnosis

The implemented VCF path deliberately accepts a narrow contract: one sample, called diploid genotypes, explicit biallelic SNVs, and required GT/DP/AD fields. It rejects unsupported structural, symbolic, multiallelic, and indel records rather than quietly interpreting them incorrectly. For broader HTS processing I would use mature reference-aware tools and validate their normalization behavior against the relevant specification.

The critical checks are:

| Check | Why I keep it explicit | Action on failure |
| --- | --- | --- |
| Reference allele matches the FASTA | A valid-looking position can still refer to another assembly or contig | Stop; verify assembly, reference checksum, contig naming, and coordinates |
| VCF position is 1-based | Python slices and BED-style intervals are 0-based, half-open | Convert an SNV at POS to `[POS−1, POS)` exactly once |
| FILTER is `PASS` | `.` means filters were not applied or are unspecified, not that the call passed | Mark for review |
| Depth and allele depths are parseable | Missing values are not zero; caller fields may use different filtering rules | Stop malformed input; review AD/DP disagreement against caller semantics |
| Heterozygous allele balance is plausible under the declared policy | An imbalanced call can indicate technical or biological effects | Review, do not automatically infer contamination or pathogenicity |

The fixture includes one passing call and one low-depth, unfiltered, imbalanced call. The second stays in the report with reasons. I do not make it disappear and then report a reassuring average.

A QC-passing call is not a clinically classified variant. Pathogenicity assessment additionally requires appropriate annotation, population and functional evidence, inheritance/context, disease validity, and a validated interpretation process. This code does not implement ACMG/AMP classification or certify a diagnostic workflow.

**Reference:** [GA4GH-maintained HTS specifications](https://samtools.github.io/hts-specs/) define SAM/BAM, VCF/BCF, and coordinate conventions. A format specification is a representation contract, not a complete clinical quality system.

## 4. Assay controls: separate plate separation from assay validation

I calculate the Z-prime statistic independently for each plate:

```text
Z′ = 1 − 3(s_positive + s_negative) / |mean_positive − mean_negative|
```

The standard deviations use `ddof=1`. Each control class needs at least two observations for that calculation, but two observations are not a sufficient assay-validation design. Equal control means make the statistic undefined; I raise an error instead of serializing infinity. A coefficient of variation is also undefined at mean zero and is reported as missing.

The fixture's first plate has well-separated, low-variability controls. The second has broad overlapping controls and is marked for review. I use Z′ ≥ 0.5 as a stated screening-quality heuristic, not as a claim that the assay has been validated. Pooling the two plates would obscure the operational failure I need to investigate.

Before trusting an assay for an R&D decision, I would examine plate uniformity and edge effects, day/operator/lot effects, stability, concentration-response behavior, interference, replicate agreement, and the intended range of use. Those questions determine whether a screening hit deserves a confirmatory experiment and whether scarce assay capacity is being spent sensibly.

**Reference:** the [Assay Guidance Manual's HTS validation chapter](https://www.ncbi.nlm.nih.gov/books/NBK83783/) distinguishes stability/process work, plate uniformity, and replicate-experiment studies. Its [quality-control collection](https://www.ncbi.nlm.nih.gov/books/NBK343427/) provides the broader operational context.

## 5. Multiple testing: define the family before adjusting it

The workflow includes a Benjamini–Hochberg implementation with finite-input validation, stable ordering, reverse cumulative minima, and restoration of the original feature order. For p-values `[0.01, 0.04, 0.03, 0.002]`, the adjusted values are `[0.02, 0.04, 0.04, 0.008]`.

These are not posterior probabilities that individual null hypotheses are true. The procedure targets false-discovery rate under its assumptions; dependence structure, the declared hypothesis family, and upstream selection matter. I would not select attractive genes first and then adjust only that subset while describing the result as a genome-wide analysis.

## Turn a review flag into the next experiment

I use the report as a decision record: what was checked, what rule was applied, which observation triggered it, and what happens next. A failed reference check blocks interpretation. A confounded design triggers a design discussion. A weak control plate triggers assay investigation. None of those should be repaired by silently changing a label.

This separation also matters commercially. Avoiding an invalid downstream analysis can save sequencing, assay, and review capacity; that is a plausible value mechanism, not a measured return on investment. I would measure it through rejected/repeated runs, review time, turnaround time, and confirmed decision quality under a predefined evaluation plan.

For regulated or accredited work, software tests are only one part of the evidence. Approved procedures, intended use, risk assessment, validation records, access control, change control, traceability, retention, and qualified review belong to the organization's quality system. I do not label these examples ISO-, CLIA-, GxP-, or 21 CFR Part 11-compliant.

Continue with [the facility evidence workflow](facility_workflow.md) to inspect calibration-aware ingestion, a relational schema, signed decisions, a read-only API, and a conserved particle model. Its [console](facility_console.html) makes the source-to-decision path inspectable without presenting the fixture as a validated manufacturing system.

Continue with [pretrained protein adaptation](../studies/protein_adaptation.md), where I inspect exactly which weights change, or [training mathematics](../studies/training_math.md), where I test whether a distributed implementation still optimizes the intended objective.
