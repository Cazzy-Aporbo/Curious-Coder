# Data provenance and observation contracts

Two measured sources are used. They answer different questions and must not be merged into a fictional patient-to-city dataset.

| Source | Observation | Coverage used | Reuse and attribution |
| --- | --- | --- | --- |
| Wisconsin Diagnostic Breast Cancer (WDBC) | Numerical characteristics extracted from a digitized fine-needle-aspirate image; benign/malignant diagnosis | All 569 published records, 30 predictors | UCI lists CC BY 4.0. Credit the dataset creators and retain the source/license link. |
| NOAA/NCEI GHCN-Daily | Daily station-level minimum and maximum air temperature | Phoenix Airport, USW00023183; 2023-01-01 through 2024-12-31 | Public-access NOAA US station observations. Cite NOAA/NCEI and the dataset DOI; do not imply endorsement. NCEI supplies no warranty of suitability. |

## Acquire once, analyze offline

```bash
python -m studies.data --sync
python -m studies.data
```

The first command downloads only the necessary source files and creates `data/snapshots`. It refuses to overwrite an existing snapshot. The second command verifies the stored bytes against `manifest.json` without making a network request. The repository includes the acquired snapshot so tests and analysis do not depend on upstream availability.

To investigate a later upstream revision, acquire into a **new directory** with `--output`. Compare its manifest before adopting it; a newer download is not automatically the same experiment. On Python.org macOS installations without a configured certificate bundle, `SSL_CERT_FILE=/etc/ssl/cert.pem` selects the system trust bundle. TLS verification must remain enabled.

The manifest records retrieval time, source URLs, SHA-256 hashes, byte lengths, the station catalog hash, the GHCN version string, and transformations. Hashes detect changes relative to the checked-in manifest; they do not independently establish the scientific truth of a measurement or protect against an attacker rewriting both data and manifest.

## Diagnostic measurements

The original `wdbc.data` bytes are retained unchanged. The loader assigns descriptive feature names matching scikit-learn's canonical ordering and verifies the published dimensions, labels, unique record identifiers, and finite feature values.

- `record_id` is an identifier, **never a predictor**.
- `diagnosis = M` becomes positive class `1`; `B` becomes `0`. This is the reverse of scikit-learn's integer target convention, so a test explicitly checks the mapping.
- Ten nucleus descriptors are summarized by their mean, standard error, and “worst” value, producing 30 features. “Worst” is an aggregation in the source, not a future clinical outcome.
- The repository does not invent physical units where the supplied feature table leaves them unspecified. Standardized predictors are dimensionless.
- There are no missing values in the verified feature table. The pipeline fails if this changes rather than silently applying an unreported cleaning rule.
- The downloaded table does not establish site diversity, temporal independence, demographic representation, or an external clinical cohort. Random stratification is appropriate for this bounded benchmark, not evidence of transportability.

**Citation:** Wolberg, W., Mangasarian, O., Street, N., & Street, W. (1993). *Breast Cancer Wisconsin (Diagnostic)* [Dataset]. UCI Machine Learning Repository. DOI: [10.24432/C5DW2B](https://doi.org/10.24432/C5DW2B). [Dataset description and license](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic); [CC BY 4.0 terms](https://creativecommons.org/licenses/by/4.0/legalcode). The source credits the image-feature methodology to Street, Wolberg, and Mangasarian's 1993 *Nuclear feature extraction for breast tumor diagnosis*.

## Environmental observations

The raw retained `.dly` subset contains the original monthly `TMIN` and `TMAX` lines for the stated station and years. The derived `phoenix_daily.csv` contains one row per date and element, including the raw integer value and measurement, quality, and source flags.

- Raw temperature is in **tenths of degrees Celsius**; divide by ten exactly once.
- `-9999` denotes missing data. A nonblank quality flag means the value failed a quality check; neither is plotted as a usable observation.
- Excluded values remain present with missing `value_c`, preserving the distinction between absence and a measured zero.
- Calendar expansion respects leap years. No interpolation fills missing days.
- Station latitude/longitude and elevation come from the same acquisition's NOAA station catalog.
- GHCN-Daily is quality-controlled but not homogenized for every station-history change. A two-year subset is not sufficient for a climate trend claim.
- A station air-temperature measurement is not surface temperature, mean radiant temperature, indoor exposure, or a street-level microclimate map.

**Citation:** Menne, M. J., et al. (2012). *Global Historical Climatology Network–Daily (GHCN-Daily), Version 3*, Phoenix Airport TMIN/TMAX subset, 2023–2024; acquisition version and date in the manifest. NOAA National Climatic Data Center. DOI: [10.7289/V5D21VHZ](https://doi.org/10.7289/V5D21VHZ). [NCEI metadata, citation, and use constraints](https://www.ncei.noaa.gov/access/metadata/landing-page/bin/iso?id=gov.noaa.ncdc%3AC00861). Format and flags: [official GHCN-Daily readme](https://www.ncei.noaa.gov/pub/data/ghcn/daily/readme.txt). Methods: Menne et al. (2012), *An overview of the Global Historical Climatology Network-Daily Database*, DOI 10.1175/JTECH-D-11-00103.1.

## Protein sequences and checkpoint provenance

The [protein fixture manifest](protein_fixture/manifest.json) records unmodified UniProt FASTA downloads for P69905 and P68871, their accessions, retrieval time, lengths, and hashes. The [UniProt reuse notice](protein_fixture/LICENSE.txt) accompanies those records. They are homologous human hemoglobin sequences used for a training-mechanics audit, not an independent validation set.

The ESM-2 base checkpoint is pinned by model ID, revision, and publisher weight SHA-256 in `studies/protein_transfer.py`. Large weights are downloaded explicitly into ignored `artifacts/`, never silently fetched during ordinary core tests. The retained adaptation report names the exact base revision and distinguishes adapter updates from biological performance.

## Synthetic biotech fixtures

The `biotech_fixture` directory contains deliberately constructed paired reads, a sample sheet, a short reference sequence, SNV calls, and plate controls. These test file conventions, arithmetic, and acceptance/review behavior. They were not collected from patients or generated by a laboratory instrument. They cannot establish clinical or analytical validation.

## What is deliberately not called data

Canopy fractions, albedos, solar forcing, heat-transfer coefficients, proposed assets, and underground utility envelopes in the spatial study are **synthetic scenario inputs**. H3 cells are derived spatial indexes, not measured environmental conditions. No private patient data, real utility survey, municipal approval, or operational Loopchii telemetry is included.
