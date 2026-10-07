# Explanation should not change the claim

An explanation becomes more useful when it is shaped for the person reading it. It becomes less trustworthy when that shaping quietly changes the value, scope, context, or confidence of the underlying result.

I separate those responsibilities. First, I decide whether a bounded factual request is supported. Only then do I render it for a learner, researcher, or reviewer. The rendering can change the guidance around the evidence; it cannot change the decision.

## The contract I want every explanation to keep

| Stage | Question | Failure that must stay visible |
| --- | --- | --- |
| Select | Is this source approved for this request? | Withdrawn record or unapproved revision |
| Bound | Does it apply here, now, and for this purpose? | Wrong context, expired validity, or disallowed use |
| Verify | Are these the bytes that were approved? | Digest mismatch |
| Extract | Can the declared pointer resolve to one scalar with the expected unit? | Missing value, nested document, or unit mismatch |
| Reconcile | Do eligible sources agree exactly? | Conflict requiring review |
| Render | Can a reader view preserve the decision unchanged? | Different values, limits, or support across views |

This is narrower than open-ended language generation. That is deliberate. Before asking a model to phrase a conclusion, I want to know which facts the system was entitled to use.

## What the implementation does

The [evidence contract](evidence_contracts.py) receives a request, candidate source records, and an explicit policy. Each source declares its identity, revision, context, allowed uses, validity interval, expected SHA-256 digest, units for available scalar fields, and limits.

A fact is admitted only when all of these conditions hold:

```text
approved disposition
∧ approved revision
∧ valid_from ≤ policy_clock < valid_until
∧ matching context
∧ allowed purpose
∧ matching bytes
∧ resolved scalar with the declared unit
```

The output records the decision, admitted facts, missing facts, conflicts, source eligibility trace, preserved limits, policy revision, and an evidence-snapshot hash. Supported facts retain source identity, revision, digest, JSON pointer, value, unit, and context.

There are three outcomes:

| Decision | Meaning |
| --- | --- |
| `supported_within_contract` | Every requested fact has eligible, unit-consistent, non-conflicting support |
| `refused` | At least one fact lacks eligible support; no partial answer is released |
| `needs_review` | Eligible sources disagree; the system does not choose a winner by ranking |

“Supported” does not mean that the source is true, that a use is lawful, or that an open-ended generated statement is correct. It means the requested scalar facts satisfy this explicit registry and policy.

## Why reuse needs a strict cache key

Reusing an earlier decision can save work. It can also preserve an answer after its evidence has expired or been withdrawn.

The [decision cache](evidence_contracts.py) therefore keys reuse on the request, policy clock and revision, source metadata, and actual source bytes. Withdrawal, revision change, expiry, context change, purpose change, or byte change produces a new evaluation. Cache hits return deep copies, so a caller cannot mutate the stored decision.

```bash
python -m studies.evidence_contracts
python -m pytest tests/test_evidence_contracts.py -v
```

## Recorded checks

The [recorded run](results/evidence_contracts.json) uses the synthetic facility report as source material. Its approval labels and logical clock are fixtures, not real QA approvals or dates.

| Case | Decision | Why |
| --- | --- | --- |
| Approved current source | Supported | Facts resolve with matching units and approved revision |
| Validity end reached | Refused | The interval end is exclusive |
| Another facility context | Refused | A valid record from one context is not borrowed by another |
| Clinical-release purpose | Refused | The fixture permits software review, not release decisions |
| Altered bytes | Refused | The content no longer matches its approved digest |
| Two approved sources disagree | Needs review | Disagreement is surfaced rather than averaged or ranked away |

The tests also verify duplicate-source rejection, oversized-input rejection before parsing or cache lookup, JSON-pointer escaping, all-or-nothing missing-fact behavior, order independence for conflicts, and reader views that preserve every decision field.

## Where language models fit

A language model could help phrase a supported result. It should receive only admitted facts and limits, and its output still needs claim-level verification. If the generated text introduces a number, causal relationship, recommendation, or scope that is not in the admitted facts, that output should fail.

This module does not implement that open-ended generation verifier. It establishes the evidence boundary that such a verifier would need.

## Accountability has to be designed in, not described afterwards

The Council of Europe Framework Convention on Artificial Intelligence and Human Rights, Democracy and the Rule of Law is the first legally binding international AI treaty. As Marc Rotenberg summarises in *Techplomacy* (October 2026), it regulates consequences and responsibilities rather than particular architectures: transparency and oversight, accountability, the ability to contest adverse outcomes, and risk assessment before deployment.

Those principles translate into specific engineering properties. Here is where this module supports them, and where it stops.

| Principle | What this implementation provides | What it cannot provide |
| --- | --- | --- |
| Transparency and oversight | Every decision carries its source trace, eligibility reasons, policy revision, and snapshot hash | Organisational oversight, notice to affected people, or access rights |
| Accountability | The approved revision and policy are explicit inputs, so responsibility for approval is named rather than implied | Legal accountability or authenticated approvers |
| Contestability | Refusals and conflicts state which fact and which condition failed, giving a reviewer something specific to challenge | A remedy process or appeal workflow |
| Risk before deployment | Unsupported, expired, or out-of-scope use is blocked before any explanation is rendered | An impact assessment for a real deployment |

Good governance articles also point to a quieter infrastructure gap: shared evaluation methods and incident reporting that work across organisations. A decision record that names its sources, policy, and failure reasons is the kind of artefact that such reporting depends on. This mapping is a design aid, not a compliance claim.

## Connection to the rest of the repository

The [learning-signal study](learning_signals.md) shows why a proxy that is easy to raise can drift from the outcome it represents; this module refuses to let missing evidence raise confidence. The [facility workflow](../biotech/facility_workflow.md) preserves measurement history and audit evidence. The [retrieval study](evidence_retrieval.md) finds candidate literature. This layer decides whether specific facts can cross into an explanation without changing their meaning.

Together, they form a reviewable path from source to decision. Each step still has its own limits, and none substitutes for scientific judgment.
