# From a trained object to a repeatable operation

I separate two problems that are often hidden behind a demonstration: preserving a model's meaning when I export it, and preserving an operation's meaning when I retry it. Both need an input contract, a failure policy, and evidence that the final behavior matches the intended one.

## Export the preprocessing, not just the weights

The measured study trains its MLP on standardized features. If I export only the neural layers and later feed raw values to them, I have deployed a different function. Feature order is equally important: an array with the right shape can still contain the wrong measurements in each column.

The [export module](inference.py) bundles standardization buffers with the network and uses `torch.export` to produce a bounded dynamic-batch graph. The manifest records exact feature names/order, input dtype, batch limit, source-data hash, checkpoint hash, artifact hash, and export error.

```bash
python -m studies.run
python -m studies.inference export --output artifacts/inference-bundle
python -m studies.inference example-input --output artifacts/example-input.csv
python -m studies.inference predict --bundle artifacts/inference-bundle --input artifacts/example-input.csv
```

Use a new output directory if a bundle already exists. The export command refuses to overwrite it. The input command similarly refuses to replace an existing example file.

I compare the exported output against the original preprocessing path—float64 standardization followed by conversion to the network's float32 input—not merely against another instance of the converted wrapper. That distinction catches conversion changes that a self-comparison would miss. Tests cover batches of different sizes, exact schema order, missing/nonfinite values, and digest mismatch before loading.

The exported graph is the experimental MLP, not the validation-selected logistic model. This example teaches artifact delivery; it does not change the model-selection conclusion or produce a clinical service. The output is a table of experimental probabilities with row identifiers, not diagnostic instructions.

A hash establishes integrity relative to a trusted manifest, not provenance on its own. Load only trusted artifacts. An attacker who replaces both a manifest and its artifact can defeat a local checksum check; distribution requires authenticated provenance, access controls, and an appropriate trust boundary. Exported-program serialization is also version-sensitive, so I record and pin the runtime rather than promising arbitrary forward compatibility.

## Bound the work before adding retries

The [async pipeline](async_pipeline.py) uses Python 3.11 `TaskGroup`, a bounded queue, a fixed worker count, per-attempt deadlines, typed expected failures, jittered backoff, and task-local request context.

```bash
python -m studies.async_pipeline
python -m pytest tests/test_delivery.py -v
```

The fixture sends eight requests representing four distinct operations. One receiver commits a result and then loses its acknowledgement. Retrying with the same request identity returns the stored result rather than creating another mutation. The receiver is intentionally in-memory; the separate SQLite journal demonstrates durable local transaction behavior.

I keep several boundaries explicit:

- A transient failure or deadline may be retried within a finite attempt budget.
- An expected permanent failure is recorded without retry.
- An unexpected programming error propagates and cancels sibling work; it is not relabeled as a recoverable network problem.
- Cancellation reaches active handlers and cleanup resets task-local context.
- Queue capacity bounds queued work, while the worker count bounds active work. The returned list still uses O(number of jobs) memory; a large production stream should persist or consume outcomes incrementally.
- Async deadlines depend on cooperative cancellation. A blocking CPU function or handler that suppresses cancellation can defeat the intended responsiveness; use an appropriate process/executor boundary rather than pretending `async` makes CPU work nonblocking.

The tests deliberately create a blocked producer, timeouts, permanent errors, transient errors, and cancellation. That is where I expect mistakes to surface. A success-only example would not establish the useful contract.

## Carry the result into the next decision

In a biotech workflow, these patterns might deliver a reviewed QC report, an embedding job, or a versioned assay summary. They do not automatically authorize patient-data access or make an external side effect transactional. I would keep specimen/tenant authorization ahead of data access, redact sensitive payloads from telemetry, and coordinate external effects through a defined outbox/inbox or reconciliation design.

The practical value is fewer silent changes in meaning: the exported model receives the measurements it was trained to receive, and a retried operation retains its original identity. I would measure that value through reproducibility, failed/repeated jobs, recovery behavior, and review effort—not by calling a wrapper production-ready.

**References:** [PyTorch export documentation](https://docs.pytorch.org/docs/2.8/export.html), [Python task groups and cancellation](https://docs.python.org/3/library/asyncio-task.html#task-groups), and [AWS on idempotent retries](https://aws.amazon.com/builders-library/making-retries-safe-with-idempotent-APIs/).
