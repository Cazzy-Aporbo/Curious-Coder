# Start here: read, reproduce, then change one thing

You do not need to know GitHub well to use these studies. I suggest taking the shortest route that answers your question, then returning to the implementation when you want to understand or challenge a result.

## If you only want to read

Open the repository's README and follow a study link. On GitHub, `.md` files are formatted pages, `.py` files are Python source, and `.csv`/`.json` files contain data or recorded results. You can inspect them without installing anything. The site version adds search, figure inspection, copy buttons, and an interactive view of fixed held-out predictions.

Start with [biotech QC](biotech/README.md) if you work with samples, reads, variants, or assays. Start with [the measured ML comparison](studies/clinical_benchmark.md) if you want to understand model selection. Start with [protein adaptation](studies/protein_adaptation.md) if you want to inspect pretrained weights and low-rank updates.

## If you want to run a study

Install Python 3.11 or 3.12 and Git, then open a terminal. These commands create a local copy and an isolated Python environment:

```bash
git clone https://github.com/Cazzy-Aporbo/Curious-Coder.git
cd Curious-Coder
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-torch.txt
python -m pip check
python -m biotech.qc
```

On Windows PowerShell, use `.venv\Scripts\Activate.ps1` instead of `source .venv/bin/activate`. The prompt may show `(.venv)` once the environment is active. This keeps the study's packages separate from other Python projects.

If you use GitHub's **Code → Download ZIP** instead, extract it and open a terminal in the extracted directory. The Python commands still work; Git history and branch commands do not apply until you have a Git clone.

The QC command prints a report and saves it under `studies/results`. Some fixture records intentionally require review. That is not an installation failure. Read the reason codes before changing a threshold.

## If a command fails

Read the final error line first, then the command that produced it. A missing package usually means the intended environment is not active or its requirements were not installed. A data-integrity error means the snapshot differs from its manifest; do not disable the check merely to obtain an output. A model-download error is different from a training error: ordinary core tests do not need to download a pretrained model.

For macOS Python installations with certificate-bundle errors during explicit data acquisition, the data guide describes selecting the system certificate bundle. Do not disable TLS verification.

## Reproduce before editing

```bash
python -m studies.data
MPLBACKEND=Agg OMP_NUM_THREADS=1 python -m studies.run
MPLBACKEND=Agg python -m pytest
python scripts/build_site.py
python -m http.server 8000 --directory _site
```

In PowerShell, set `$env:MPLBACKEND = 'Agg'` and `$env:OMP_NUM_THREADS = '1'` before the Python commands. Open the local address printed by the HTTP server. Stop the server with Ctrl+C when finished.

The recorded public-data snapshot is separate from the generated results. Keep that distinction when experimenting: changing a source file, split, threshold, or model configuration changes the experiment. It is not just a cosmetic edit.

## Make a small, reviewable change

```bash
git switch -c investigate-qc-policy
git status
```

Change one assumption, add or adjust a test, and run the relevant test file. Then inspect the difference:

```bash
git diff
python -m pytest tests/test_biotech_qc.py -v
```

A branch is a named line of work, not a separate copy of the data. A commit records a reviewed set of changes locally. A push publishes commits to a remote repository. A pull request asks maintainers to review a branch before merging it; it is not the same as pushing directly to the main branch.

Do not upload patient records, credentials, restricted utility maps, or data without appropriate reuse permission. The included fixtures are public measurements or explicitly synthetic teaching cases.

## Read CI as evidence with a scope

GitHub's **Actions** tab shows automated workflow runs. Open a failed job and find the first failing step; a red badge does not tell you whether the problem was a dependency, a test, or deployment configuration. A green run means the configured checks passed. It does not establish biological validity or regulatory compliance.

For your own investigation, leave a short record: the question, the source and version, the change, the comparison, the result, and what remains unresolved. That is enough to make the next conversation more useful than “it worked on my machine.”
