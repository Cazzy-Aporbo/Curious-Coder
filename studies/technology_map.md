# Choose the tools that serve the question

I want the computational layer to feel clear before it feels impressive. A reader should be able to see what moves through it, where meaning could be lost, and which measurement would justify adding more machinery.

This map separates **implemented examples**, **optional tools not integrated here**, and **clinical interfaces that still require conformance and validation work**. “Optional” is not an installation or performance claim. The small CPU examples remain useful references when a larger implementation moves to another engine.

## First, locate the evidence that already runs

| Capability | Present implementation | What I can inspect |
| --- | --- | --- |
| Scientific arrays, statistics, optimization | NumPy, SciPy, scikit-learn; pandas for bounded tabular studies | [Measured comparisons](clinical_benchmark.md), [statistical validation](statistical_validation.md), [resource constraints](execution_contracts.md) |
| Neural computation | PyTorch 2.x, Transformers, pinned ESM-2 and MiniLM | [Weight-update boundaries](protein_adaptation.md), [distributed objective](training_math.md), [precision/latency comparison](evidence_retrieval.md) |
| Model delivery | Schema validation, preprocessing-inclusive export, artifact hashes | [Delivery contracts](engineering.md) |
| Data movement | Memory mapping, NumPy views, CPU tensor aliasing, spawned reader | [Measured buffer contracts](execution_contracts.md) |
| Facility data integrity | SQLite, fixed-point normalization, Ed25519 event signatures, read-only inspection | [Facility workflow](../biotech/facility_workflow.md) and [console](../biotech/facility_console.html) |
| Biological input QC | Biopython, explicit FASTQ/sample/variant/assay contracts | [Biotech QC](../biotech/README.md) |

A Python loop around sequential time steps or bounded I/O is not automatically a performance defect. Nor is a pandas table automatically a bottleneck: 569 records do not justify a new execution engine merely by being biological. I measure the operation, its data shape, and its allocation behavior before replacing it.

## Molecular and pharmaceutical tools: useful, but not interchangeable

These packages are **not integrated or benchmarked in this repository**. They identify appropriate next experiments, not features hidden behind an import statement.

| Tool | Where it belongs | The decision I would make explicit first |
| --- | --- | --- |
| RDKit | Molecular parsing, fingerprints, substructure matching, descriptors | Invalid structures, salts, tautomer/protonation policy, stereochemistry, and scaffold-aware evaluation; a formula balancer is not a molecular toolkit |
| OpenMM | Molecular-dynamics execution on supported CPU/GPU platforms | Force field, boundary conditions, equilibration, sampling/convergence, precision, and reference observables |
| ParmEd | Molecular topology and parameter manipulation/conversion | Atom ordering, units, parameter completeness, and conversion fidelity; ParmEd is not itself a GPU MD engine |
| DeepChem | Molecular ML datasets, featurizers, and models | Assay definitions, missing labels, chemical-series leakage, and applicability domain |
| Chemprop | Message-passing molecular-property models | The particular ADMET endpoint, assay conditions, scaffold/temporal splits, calibration, and external evaluation |
| AutoDock Vina | Docking and pose/scoring workflows | Receptor/ligand preparation and search uncertainty; a docking score is not a measured binding affinity or clinical effect |
| OpenFE | Alchemical free-energy workflows | Perturbation mapping, force-field suitability, repeat sampling, convergence, uncertainty, and comparison with relevant experiments |

I would not connect a predicted property directly to an experimental or clinical action without checking what the label represents. “Toxicity,” for example, is not one interchangeable endpoint across assays, organisms, doses, and exposure windows.

## Acceleration begins with a reference result

| Technology | Status here | Proof I would require before relying on it |
| --- | --- | --- |
| PyTorch tensors/autograd/export | Implemented | Shapes, masks, gradients, preprocessing parity, and artifact integrity |
| `torch.compile` | Not benchmarked here | Separate compilation from steady-state time; check graph breaks, dynamic shapes, numerical agreement, and representative workload size |
| JAX `jit` / `vmap` | Not integrated | Precision settings, compilation/amortization, transformation semantics, and gradient/reference comparisons |
| Transformers / ESM-2 | Implemented for bounded inference/adaptation | Pinned resources, correct tokenization/masking, frozen-versus-trainable parameters, and task-specific evaluation |
| AlphaFold / structure-prediction wrappers | Not integrated | Structure-specific inputs, confidence interpretation, database/template provenance, and independent structural evaluation; ESM-2 adaptation is not a folding validation |
| FlashAttention-3 | Not installed or exercised | Supported Hopper hardware/CUDA, forward and backward agreement, mask behavior, precision, and end-to-end memory/latency measurements |

[FlashAttention-3](https://pytorch.org/blog/flashattention-3/) targets particular GPU capabilities; it is not a CPU optimization for this Mac. Reducing attention memory traffic does not make an arbitrary whole genome fit into a model's context, and training-corpus size is not the same quantity as inference context length.

The [distributed fixture](training_math.md) is a useful reference before any such acceleration: unequal token counts must still produce the intended global objective. A faster reduction of the wrong objective is not a successful optimization.

## Tabular, statistical, and single-cell processing

| Tool | Status here | What can go wrong at the boundary |
| --- | --- | --- |
| NumPy / SciPy | Implemented | Dtype conversion, allocation, numerical conditioning, invalid sampling assumptions, and unit mismatch |
| pandas | Implemented on bounded study tables | Implicit coercion, missingness, duplicated joins, and inappropriate row-wise operations; profile before replacing |
| Polars | Documented option; not integrated | Lazy plans and columnar execution can help, but zero-copy conversion depends on dtype, nulls, chunking, layout, and mutability |
| Statsmodels | Not integrated | Model specification, dependence structure, identifiability, diagnostics, and interpretation; it is not a universal Bayesian/survival engine |
| Scanpy / AnnData | Not integrated | Sparse-to-dense expansion, view/copy behavior, cell/gene identity, donor-level independence, batch structure, and normalization before evaluation |

[Polars documents the exact conditions](https://docs.pola.rs/api/python/stable/reference/dataframe/api/polars.DataFrame.to_numpy.html) under which `to_numpy` can avoid a copy. `allow_copy=False` is a useful assertion when that is part of the contract; it should fail rather than quietly make a misleading speed claim.

For single-cell work, a cell count is not a count of independent biological replicates. Donor/sample identity needs to survive filtering, joins, embeddings, and evaluation. A backed matrix also does not guarantee that every downstream operation remains out of core. The repository's existing single-cell and survival chapters are reading material, not executed clinical studies.

## Robotics and execution infrastructure

| Technology | Status here | Boundary to preserve |
| --- | --- | --- |
| Opentrons Python API | Not integrated; no robot control | Protocol/API version, labware, volume limits, liquid classes where supported, calibration, and simulation before authorized execution |
| ROS 2 / `rclpy` | Not integrated | Time synchronization, QoS, command ownership, emergency stops, device states, and hardware validation |
| Pydantic v2 | Not integrated; existing contracts use explicit validation | Strict versus coercive parsing, bounded payloads, units and semantic checks; a valid object is not a valid laboratory action |
| Ray | Not integrated | Task granularity, scheduling/serialization overhead, retries, object lifetime, and idempotent side effects |
| NumPy memory mapping | Implemented as a CPU proof | File permissions, layout, page faults, lifetime, copy-on-write, and what actually crosses the process boundary |
| Pybind11 | Not integrated | C++ ownership/lifetime, GIL handling, exceptions, and array layout; it is not a general Rust binding layer |
| Cython / Mojo | Not integrated | A measured hotspot, equivalence tests, build portability, and maintenance cost; changing language does not establish a real-time deadline |

The [Opentrons documentation](https://docs.opentrons.com/python-api/reference/execute-simulate/) distinguishes simulation from execution. That distinction belongs in the interface, not only in a warning after a command has been sent. This repository's facility API is deliberately read-only.

A mapped CPU file is not a terabyte-scale GPU transport system. Cross-host movement still needs transport; device transfer may allocate or copy; compression and image decoding change the storage problem. The [data-movement example](execution_contracts.md) verifies narrow buffer-sharing facts and leaves those broader claims unmade.

## Hospital operations: the objective needs a human owner

The implemented [resource model](execution_contracts.md) is a synthetic, time-indexed optimization example. It reserves operating rooms, a surgeon/team, a specific instrument set, turnover time, and recovery capacity. It checks the returned schedule independently of solver status and compares a small problem with exhaustive enumeration.

It does **not** ingest live patient information, forecast emergency-department demand, recommend triage priority, set staffing levels, or issue operational commands. Equal-weight start-time minimization is a teaching objective, not a statement about fair access or clinical urgency.

| Requested interface | Current status | Required integration and evidence |
| --- | --- | --- |
| HL7 v2 ADT events | Not implemented | Message/version agreement, source identifiers, corrections/cancellations, ordering, deduplication, acknowledgment, encounter/location semantics |
| FHIR events and resources | Not implemented | Declared FHIR version/profiles, resource-version handling, authorization, provenance, subscriptions where appropriate, conformance tests |
| Bed/EVS orchestration | Not implemented | Actual bed states, infection-control constraints, turnaround distributions, staff/resource availability, human ownership of overrides |
| ED surge prediction | Not implemented | Governed temporal data, prediction-time availability, shift/site evaluation, uncertainty, alert burden, and prospective operational validation |
| OR/PACU optimization | Synthetic model implemented | Real constraints and uncertainty, local priorities, staffing/clinical review, disruption handling, and a monitored deployment study |

A discharge notification does not necessarily mean a bed is ready. An order, encounter, procedure, and physical resource are different objects. I would preserve those distinctions rather than make an event-driven architecture look complete by joining identifiers with similar names.

Local practice also matters. Staffing, consent, terminology access, regulatory scope, and acceptable trade-offs differ between health systems. I would not export one institution's assumptions into another merely by translating its interface labels.

## Radiology and point-of-care interfaces

No PACS/VNA adapter, diagnostic imaging model, DICOMweb service, CDS Hooks service, or clinical-record write path is implemented here. The following are the contracts a real integration would have to satisfy.

| Boundary | Meaning | Details that must remain visible |
| --- | --- | --- |
| QIDO-RS | Search for DICOM resources | Query semantics, pagination, identifiers, authorization, and response representation |
| WADO-RS | Retrieve DICOM resources | Transfer syntax, compressed frames, pixel decoding, geometry, content negotiation, and access controls |
| STOW-RS | Store DICOM objects | Validation, instance identity, per-instance outcomes, retries, and intended destination; not an unreviewed upload shortcut |
| Vendor metadata normalization | Preserve meaning across producers | Keep originals and transformation provenance; do not casually rewrite UIDs, laterality, orientation, units, private tags, or per-frame geometry |
| IHE AIW-I | Manage imaging AI work | Its workflow transactions use DICOM UPS-RS; generic task queues are not proof of profile conformance |
| IHE AIR | Capture/distribute/display imaging analysis results | Declared result primitives, source references, negative/partial results, approval state, and display behavior |
| DICOM SR / SEG | Structured reports and segmentations | Referenced source instances, coordinate frames, coding, derivation, and intended clinical interpretation |
| CDS Hooks `order-select` | Respond during draft order selection | Selected orders versus the full draft bundle, incomplete context, scoped authorization, service failure behavior, and clinical review |
| FHIR DiagnosticReport | A clinical reporting resource | Correct references, status, codes, units, source/version provenance, and authorized record-update semantics |

[DICOM PS3.18](https://dicom.nema.org/medical/dicom/current/output/html/part18.html) defines the web services. [IHE AIW-I](https://www.ihe.net/uploadedFiles/Documents/Radiology/IHE_RAD_Suppl_AIW-I.pdf) and [AIR](https://www.ihe.net/uploadedFiles/Documents/Radiology/IHE_RAD_Suppl_AIR.pdf) add specific interoperability expectations. Listing their names is not conformance evidence.

A memory map cannot turn compressed DICOM pixels directly into a ready-to-use tensor without decoding and checking geometry. Nor does removing a patient-name tag de-identify an image: identifiers can remain elsewhere, including pixel content. The current CPU proof makes no medical-imaging or de-identification claim.

[CDS Hooks 2.0](https://cds-hooks.hl7.org/2.0/) and the [order-select definition](https://cds-hooks.org/hooks/order-select/) describe context and workflow timing. Missing allergy or renal-function information is not evidence that no contraindication exists. I would not implement clinical rules from guessed thresholds or issue recommendations from incomplete data simply to make a demonstration appear closed-loop.

SNOMED CT, LOINC, and RadLex also need appropriate terminology versions, code systems, licensing/access arrangements, and mapping review. Similar display text does not establish equivalent meaning. No terminology service or urgent-finding escalation workflow is currently deployed in this repository.

## Compliance and performance are both system properties

The [signed facility records](../biotech/facility_workflow.md) demonstrate integrity checks relative to trusted keys and an expected head. They do not establish HIPAA compliance, HITRUST certification, authenticated human approvals, confidentiality, or immutable storage. Those claims require a defined system boundary and organizational as well as technical evidence.

The same care applies to speed. I keep correctness, representative workload, cold versus warm paths, throughput, latency tails, allocation, and failure behavior together. There is no universal “sub-millisecond” guarantee here, and no reason to let one fast kernel stand in for the performance of an entire scientific workflow.
