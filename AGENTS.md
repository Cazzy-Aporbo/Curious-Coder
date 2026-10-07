# Project working notes

## Scope and environment

- This is an educational scientific-computing collection, not a clinically validated package.
- Supported verification environments are Python 3.11 and 3.12. Use a project virtual environment; never install dependencies globally.
- Install `requirements-torch.txt` for the complete test suite. It includes `requirements.txt`; `requirements-docs.txt` is sufficient for the standalone site builder.
- On Linux CI, install `requirements.txt` and then `torch==2.8.0` from the official PyTorch CPU index. No GPU, network datasets, or service credentials are needed by tests.
- SciPy 1.15.3 failed to load its PROPACK extension on the local Darwin 27 machine; the pinned 1.16.2 wheel loads successfully.

## Verification

Run from the repository root, with the virtual environment active:

```bash
python -m pip check
python -m ruff check . --select E9,F63,F7,F82
MPLBACKEND=Agg OMP_NUM_THREADS=1 python -m pytest
python -m pytest tests/test_learning_lab.py --cov=patient_leakage_lab --cov-branch --cov-fail-under=85
python scripts/build_site.py
```

Use actionlint 1.7.7 to validate `.github/workflows/`; the CI workflow runs its versioned Docker image. Locally the native binary also works. Docker Desktop is installed on the development machine but its daemon was not running during initial verification.

## Conventions and limits

- Keep scientific claims separate from executable correctness. Import tests do not validate a physiological model.
- Existing hyphenated example filenames are loaded through `importlib.util` in tests; preserve links when changing names.
- Do not run the large interactive training demonstrations as CI tests. Prefer small seeded inputs, analytic invariants, reference comparisons, and CPU forward/backward checks.
- Fit preprocessing on training observations only. Repeated observations need group-aware splits when estimating generalization to new individuals.
- The site builder publishes only root Markdown (excluding this file) and learning files in its explicit content directories. Do not publish the repository root, environment files, or virtual environments.
- Site builds verify local file links, not external URLs or anchor fragments. Existing Markdown code snippets are reading material, not all executable programs.
- The measured studies live in `studies/`. Verify the checked-in public snapshots with `python -m studies.data`; regenerate results and figures with `MPLBACKEND=Agg OMP_NUM_THREADS=1 python -m studies.run`. This is offline; acquisition is separately requested with `--sync` and refuses to overwrite an existing snapshot.
- `tests/test_studies.py` validates data, recorded metrics, training determinism, and local system contracts. Run it when modifying the studies. Site assets now include `studies`, `data`, and `assets`; never add private measurements or unlicensed datasets there.
- For Python.org macOS TLS-bundle errors during acquisition, `SSL_CERT_FILE=/etc/ssl/cert.pem` uses the system CA bundle without disabling certificate verification.
- Pages publication is manually requested from main and gated by CI. Release provenance must refer to the actual learning archive, never placeholder artifacts.
- Do not invent personal experiences, clinical results, citations, or measurements. Preserve uncertainty and label simulations and unverified narrative numbers.
- Biotech QC: `python -m biotech.qc` and `python -m pytest tests/test_biotech_qc.py`. Synthetic fixtures intentionally include REVIEW cases; these are not laboratory acceptance standards.
- Delivery/adaptation tests: `python -m pytest tests/test_delivery.py tests/test_adaptation.py`. Protein integration additionally needs `requirements-protein.txt`; use `python -m studies.protein_transfer --download` once, then rerun without downloads. Weights stay under ignored `artifacts/`.
- Before adapter training, use `torch.no_grad()` rather than inference mode for ESM checks: rotary-position buffers cached as inference tensors cannot later participate in autograd.
- Browser verification: install `requirements-browser.txt`, run `python -m playwright install chromium`, then `python scripts/check_site_browser.py`. Pure interaction calculations also run with `node --test tests/site.test.mjs`.
- Two-rank CPU check: `torchrun --nnodes=1 --nproc-per-node=2 --master-addr=127.0.0.1 --master-port=29517 -m studies.distributed_training --ddp`. CI uses Gloo on the loopback interface.
- Facility workflow: `python -m biotech.facility_demo --directory artifacts/NEW-RUN` creates a new synthetic database and attestation; never overwrite an existing run. Inspect with `python -m biotech.facility_api --database artifacts/NEW-RUN/facility.sqlite --attestation artifacts/NEW-RUN/attestation.json`. Verify with `python -m pytest tests/test_facility.py tests/test_facility_api.py` and `python -m scripts.check_facility_browser --directory artifacts/NEW-RUN`.
- The facility API is loopback-only and read-only. Fixed-point ingestion is separate from floating-point compartment physics. Signed events, head anchors, and projection hashes do not establish regulatory compliance or human authentication. No hardware control, genomic generation, causal multi-omics model, or WebGPU solver is implemented.
- Retrieval comparison: install `requirements-retrieval.txt`; use `python -m studies.evidence_retrieval --download-model` once, then run offline and render with `python -m studies.retrieval_figures`. Corpus acquisition uses only the bounded Europe PMC API. Never treat six authored anchors as complete relevance labels or warm local latency as a service SLA.
- Publication preference: the user objects to bot co-authorship attribution and requests publication to main only as Cazandra. The agreed route is that the user creates the commit in their IDE; the agent may then push that exact reviewed commit and verify CI. Do not substitute different commit attribution or rewrite existing history.
- Execution examples: `python -m studies.data_movement` verifies same-host CPU alias/copy boundaries; `python -m studies.resource_scheduling` and `python -m studies.execution_figures` reproduce a synthetic OR/PACU constraint study. Verify with `python -m pytest tests/test_execution_contracts.py`. These are not GPU transport, hospital interoperability, or clinical scheduling validations.
- Typography/palette preference: softer serif headings, warm paper, plum/rose/teal accents, clear transitions, and a personal but scientifically precise voice. Do not infer or invent a cultural identity, credentials, or clinical experience.
- Evidence contracts: `python -m studies.evidence_contracts` reproduces source eligibility, refusal, conflict, cache-invalidation, and reader-view checks. Verify with `python -m pytest tests/test_evidence_contracts.py`. It validates bounded scalar facts only, not open-ended generated text.
- Learning signals: `python -m studies.learning_signals` then `python -m studies.learning_figures`; verify with `python -m pytest tests/test_learning_signals.py`. Synthetic demonstrations of TD error timing, noisy-TV curiosity traps, and proxy overoptimization; cite primary sources, not only secondary syntheses.
- GitHub Pages deploys through `.github/workflows/static.yml` on `main` pushes after reusable CI succeeds. Repository README styling is GitHub-sanitized; use committed SVG/image assets there, while site CSS/JS applies on Pages.
- Statistical follow-up: run `python -m studies.statistical_validation`, then `python -m studies.statistical_figures`; verify with `python -m pytest tests/test_statistical_validation.py`. Keep original test rows out of nested fitting and distinguish fixed-model bootstrap uncertainty from retraining variability.
