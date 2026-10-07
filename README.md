# Curious Coder

I use Python to work through a question that keeps returning in biological data: **what would make this result trustworthy enough to change the next experiment?** My starting point is the measurement and its provenance. From there I follow the assumptions into the mathematics, the implementation, and the decision the result is meant to support.

I’m **Cazandra Aporbo**. Here I keep that reasoning close to the code: public-data studies, deliberately difficult test cases, and walkthroughs of the steps that are easy to skip. You can read an analysis, reproduce its output, inspect a failure, and then change one assumption without having to reconstruct the whole project first.

<div class="route-buttons">
<a class="button" href="START_HERE.md">Start reading or running</a>
<a class="button berry" href="biotech/README.md">Work through biotech QC</a>
<a class="button secondary" href="studies/protein_adaptation.md">Inspect pretrained weights</a>
</div>

[![CI](https://github.com/Cazzy-Aporbo/Curious-Coder/actions/workflows/ci.yml/badge.svg)](https://github.com/Cazzy-Aporbo/Curious-Coder/actions/workflows/ci.yml)

## Choose the point where your question begins

| If you are asking… | I would start here | What you can run |
| --- | --- | --- |
| Can I trust these reads, sample labels, variants, or assay controls? | [Biotech Python and R&D QC](biotech/README.md) | Paired FASTQ checks, Phred arithmetic, replicate/design audits, reference-aware SNV checks, plate-level Z′, multiple-testing correction |
| Does a more complex model improve this measured problem? | [Diagnostic-feature comparison](studies/clinical_benchmark.md) | Prior, logistic, boosting, and residual-network baselines with fixed splits, uncertainty, and recorded predictions |
| Which weights actually change during biological-model adaptation? | [Pinned ESM-2 adaptation](studies/protein_adaptation.md) | Low-rank query/value updates, frozen-weight digests, masking contracts, adapter artifacts |
| Does my distributed loop optimize the objective I intended? | [Training mathematics](studies/training_math.md) | Actual two-process CPU DDP, unequal token counts, accumulation, and a single-process gradient reference |
| Will the result survive export, retries, or cancellation? | [End-to-end Python engineering](studies/engineering.md) | Preprocessing-inclusive model export, schema checks, numerical parity, bounded async workers, retry budgets, cancellation |
| How do observations, geometry, and durable state fit together? | [Spatial and system design](studies/spatial_systems.md) | NOAA observations, H3 indexing, uncertainty-aware geometry, transactional mutation and replay |

If GitHub or virtual environments are unfamiliar, [start here](START_HERE.md). I explain how to read files, run a study, interpret a failed check, and make a small reviewable change before introducing the heavier examples.

## First, keep the measurement attached to the result

I use 569 published Wisconsin diagnostic records to compare models on 30 image-derived features. Before fitting, I reserve 341 training, 114 validation, and 114 test records. That separation is straightforward; maintaining it through preprocessing, model selection, interpretation, and later experimentation takes more care.

![WDBC split counts and training-only feature correlations, with class counts, axes, and provenance.](assets/figures/cohort.svg)

In the recorded run, I select logistic regression by validation log loss. The residual network has a lower test Brier score but a higher test log loss. I keep the original selection rule rather than switching to the metric that makes the more elaborate model look best. The [analysis](studies/clinical_benchmark.md) explains that trade-off, the small calibration bins, the correlated predictors, and the limits of the bootstrap intervals.

![Recorded model comparisons, ROC curves, and calibration with explicit uncertainty and sample support.](assets/figures/diagnostics.svg)

<div class="interactive-results"></div>

The figures come from the [recorded run](studies/results/benchmark.json) and [exported predictions](studies/results/test_predictions.csv). They are not illustrative scores. The study is also not clinical validation: it does not establish generalization across hospitals, time, demographic groups, or acquisition procedures.

## Then test the parts that can fail quietly

A high score is not the only place to look for evidence. In the [biotech workflow](biotech/README.md), I keep a low-depth unfiltered variant in the report with its review reasons instead of dropping it. I count biological units separately from technical replicates. I calculate plate controls separately so that a poor plate is not hidden by a pooled summary.

In the [protein study](studies/protein_adaptation.md), I adapt 30,720 parameters in a pinned ESM-2 checkpoint and verify that the frozen parameters do not change. The same-fixture reconstruction loss decreases, but I do not treat two homologous hemoglobin sequences as an independent biological evaluation. The useful proof is that the update path, masking, and parameter boundaries behave as intended.

In the [distributed experiment](studies/training_math.md), I split four supervised tokens unevenly across two processes. The naïve mean of local means changes the gradient by about 0.258; the correctly weighted reduction agrees with the reference to numerical precision. That small example exposes an error that can otherwise hide inside a much larger training run.

These are established methods, not claims of new algorithms. My aim is to make the reasoning operational: a declared input, an intended quantity, a counterexample, a test, and a clear next step when the check fails.

## Connect the science to the decision

For an R&D team, a useful model might help choose which candidates deserve a confirmatory assay. For a sequencing workflow, useful software might stop an assembly mismatch before it becomes an interpretation. For an engineering team, it might prevent a retry from duplicating a committed state change.

Those are different value propositions. I would evaluate them using the relevant assay budget, error costs, turnaround time, repeat-work rate, and decision quality—not assume that a better benchmark metric automatically creates business value. Each walkthrough separates what I have measured from the study that would be needed to justify an operational claim.

The [spatial example](studies/spatial_systems.md) applies the same discipline to public NOAA temperature observations and explicitly synthetic canopy/material/utility scenarios. It also maps 25 system-design concepts to concrete decisions and failure tests. The local code implements transaction, indexing, and geometry contracts; the deployment topology remains a reference design, not claimed infrastructure.

## Run a complete path

Use Python 3.11 or 3.12. The core studies run on CPU and use retained public snapshots or labeled synthetic fixtures.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-torch.txt
python -m pip check
python -m biotech.qc
python -m studies.data
MPLBACKEND=Agg OMP_NUM_THREADS=1 python -m studies.run
MPLBACKEND=Agg python -m pytest
python scripts/build_site.py
python -m http.server 8000 --directory _site
```

In PowerShell, activate with `.venv\Scripts\Activate.ps1` and set `$env:MPLBACKEND = 'Agg'` and `$env:OMP_NUM_THREADS = '1'` before the Python commands. The [beginner route](START_HERE.md) explains what each step does and what to inspect when it fails.

Protein adaptation has a separate dependency profile and an explicit first download:

```bash
python -m pip install -r requirements-protein.txt
python -m studies.protein_transfer --download
```

Later protein runs load the pinned local resources offline. I keep large checkpoints and exported bundles under ignored `artifacts/`; small public sequence fixtures, provenance, and run reports stay with the studies. [Data provenance](data/README.md) distinguishes UCI and NOAA measurements, UniProt sequences, and synthetic QC/spatial inputs.

## Find the implementation without losing the context

| Location | What I keep there |
| --- | --- |
| [biotech](biotech/README.md) | Research QC contracts and the reasoning behind their acceptance/review policies |
| [studies](studies/README.md) | Measured analysis, protein adaptation, distributed math, export/inference, async processing, and recorded outputs |
| [data](data/README.md) | Source snapshots, licenses/attribution, hashes, quality flags, and clearly identified fixtures |
| `assets` | Generated figures and the site's accessible interaction layer |
| [Core algorithms](Core-algorithms/Algorithms.md) | From-scratch methods and small counterexamples |
| [PyTorch investigations](Explore-PyTorch/Exploratory_files.md) | Training-component behavior, masks, samplers, and gradients |
| `tests` | Numerical references, malformed inputs, failure injection, concurrency, and artifact checks |
| `Biological-Systems`, [Environmental](Environmental/readme.md) | Larger exploratory models; their scientific assumptions need more validation than import checks provide |

For deeper reading, I connect [method selection](algorithm_selection_bio.md), [PCA/ICA](pca_ica_comparison.md), [batch effects](batch_effects_bio.md), [missingness](explore_stuff/discovery_missing_data.md), [imbalance](imbalanced_medical_data.md), [survival](survival_analysis_medical.md), [time series](time_series_biological.md), [causal inference](causal_inference_biology.md), [network inference](network_inference_bio.md), [single-cell analysis](single_cell_analysis.md), and [multi-omics](multiomics_integration.md) back to the same questions about observation, independence, and inference. Historical narrative examples without datasets and run records remain reading exercises, not additional empirical evidence.

## Read the checks with their intended scope

CI validates workflows and source, runs the Python tests, regenerates the measured study offline, and builds the site with local-file-link checks. Numerical tests compare against explicit references; data tests verify snapshots; failure tests examine rollback, retries, and cancellation. The browser check exercises the reader controls rather than treating rendered HTML as proof that the buttons work.

I keep broad collection coverage separate from focused coverage gates. A passing software check does not establish assay validation, clinical utility, regulatory compliance, or scientific novelty. It tells me which implementation claim has survived a specified test—and where I still need a better experiment.

Primary sources and reuse terms are linked next to the methods they support: [UCI WDBC](https://doi.org/10.24432/C5DW2B), [NOAA GHCN-Daily](https://doi.org/10.7289/V5D21VHZ), UniProt and ESM-2 in the protein walkthrough, GA4GH/NCBI/Assay Guidance Manual in the biotech section, and distributed-systems references in the engineering chapters.
