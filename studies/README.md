# Executable studies

I keep the runnable investigations here, alongside their methods and recorded outputs. Start with the scientific question, inspect the implementation contract, and then use the tests to challenge the result.

| Study | Start with | Implementation | Evidence |
| --- | --- | --- | --- |
| Diagnostic-feature classification | [Measured-data ML study](clinical_benchmark.md) | [modeling.py](modeling.py) | [Recorded benchmark](results/benchmark.json), [held-out predictions](results/test_predictions.csv) |
| Environmental observations and spatial state | [Spatial/system-design study](spatial_systems.md) | [systems.py](systems.py) | Quality-filtered NOAA snapshot and transaction/geometry regression tests |
| Protein adaptation | [Weights, masks, and low-rank updates](protein_adaptation.md) | [protein_transfer.py](protein_transfer.py), [adapters.py](adapters.py) | [Pinned-checkpoint audit](results/protein_transfer.json) |
| Distributed training | [Global objective and accumulation](training_math.md) | [distributed_training.py](distributed_training.py) | [Actual two-rank result](results/distributed.json) |
| Model delivery and asynchronous work | [End-to-end walkthrough](engineering.md) | [inference.py](inference.py), [async_pipeline.py](async_pipeline.py) | Export parity, schema, retry, timeout, and cancellation tests |
| Data lineage | [Data provenance](../data/README.md) | [data.py](data.py) | [Acquisition manifest](../data/snapshots/manifest.json) |
| Figure production | [figures.py](figures.py) | One shared palette, typography, label, and source convention | SVG for sharp rendering; PNG for compatibility |

## Reproduce

```bash
python -m pip install -r requirements-torch.txt
python -m studies.data
MPLBACKEND=Agg OMP_NUM_THREADS=1 python -m studies.run
python -m pytest tests/test_studies.py -v
python scripts/build_site.py
```

The analysis does not download data. The checked-in snapshot fixes the evidence; acquisition is a separate, explicit command. Generated JSON records the software versions, configuration, source hash, split indices, checkpoint selection, and uncertainty assumptions. A local PyTorch checkpoint is written under ignored `artifacts/`; it is not published as a clinically usable model.

## Package choices are part of the explanation

| Package | Actual role in the studies | Why it is here |
| --- | --- | --- |
| NumPy | Seeded bootstrap, matrix operations, reproducible scenario inputs | Explicit numerical operations rather than opaque chart-only summaries |
| pandas | Typed observation tables, daily-calendar expansion, exported predictions | Preserve dates, quality flags, and table-level audits |
| scikit-learn | Pipelines, identical folds, logistic/boosting/prior baselines, calibration, permutation importance | Established references and train-only preprocessing |
| PyTorch | Residual tabular network, LayerNorm, AdamW, clipping, scheduler, early stopping, restored state dictionary | Make optimization and checkpoint behavior inspectable |
| SciPy | Normal quantile for Wilson calibration intervals | State the interval calculation rather than drawing unexplained error bars |
| Matplotlib | Deterministic SVG/PNG with units, sample sizes, legends, and provenance | Publication-oriented static figures that remain readable without JavaScript |
| NetworkX | Explicit reference-architecture flow | Represent edges as data rather than unrelated decorative arrows |
| H3 | Geographic cell indexing and neighborhood geometry | Show what a hierarchical index solves—and what exact geometry still must solve |
| SQLite / hashlib / json | Atomic mutation, tenant-scoped idempotency, deterministic journal envelopes | Test system invariants with no paid service or production side effects |

Adding a package is not evidence of knowing when to use it. Each dependency has an executable role, a testable contract, and a limitation in the accompanying analysis.
