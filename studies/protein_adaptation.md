# Adapting a protein model without losing track of the experiment

I want to inspect what changes when I adapt a pretrained biological model. That is a different question from whether adaptation improves a biological task. I answer the first with a small, executable audit and leave the second dependent on a properly designed study.

Here I use Meta's **ESM-2 t6 8M** masked protein-language model at a pinned repository revision, together with public UniProt sequences for human hemoglobin alpha and beta. These are amino-acid sequences, not genomic DNA. I do not feed DNA letters into a protein tokenizer and assume the resulting embeddings are meaningful.

## Start from the scientific decision

A useful downstream task might rank proteins for a confirmatory assay, estimate a measured property, or assist annotation review. Each task needs a target definition, an experimental unit, an evaluation population, and a cost of error. A masked-residue training loss is not a substitute for any of them.

For that reason, I use the two hemoglobin sequences only as **mechanics fixtures**. They are homologous, and their relationship to the checkpoint's pretraining corpus has not been excluded. I do not train on one and call the other an independent test set. The recorded loss change below describes the same training fixture before and after an update.

## Acquire the exact resources, then run offline

```bash
python -m pip install -r requirements-protein.txt
python -m studies.protein_transfer --download
python -m studies.protein_transfer
```

The first run explicitly downloads the small public checkpoint and sequences. Later runs load local files only. The weights use safetensors; remote model code is disabled. The publisher's weight SHA-256 and repository revision are pinned in [the implementation](protein_transfer.py). The base checkpoint stays under ignored `artifacts/`, while the small sequence fixtures and their provenance are retained with the teaching material.

| Resource | Identity | Role |
| --- | --- | --- |
| ESM-2 | `facebook/esm2_t6_8M_UR50D`, revision `c731040fcd8d73dceaa04b0a8e6329b345b0f5df` | Pretrained masked protein-language model; MIT model-card license |
| Hemoglobin alpha | UniProt `P69905`, 142 residues in the retained sequence | Public mechanics fixture |
| Hemoglobin beta | UniProt `P68871`, 147 residues in the retained sequence | Public mechanics fixture, not an independent validation cohort |

I retain accession, source URL, retrieval time, sequence length, and hash. The sequence loader rejects DNA-like misuse only through its declared amino-acid alphabet and length contract; some strings are valid in both alphabets, so metadata and task provenance still matter. Alphabet validation alone cannot establish biological identity.

## Work from the tensor contract upward

1. **Tokenize without silent truncation.** The fixture accepts at most 1,022 residues, leaving room for special tokens. Longer sequences require an explicit windowing or long-context policy.
2. **Keep masks distinct.** Attention padding, special tokens, and supervised masked positions have different meanings. Padding and BOS/EOS are not biological residues.
3. **Choose reconstruction positions reproducibly.** I mask every eleventh valid residue for this small audit and use `-100` elsewhere in the target tensor. This is a deterministic diagnostic fixture, not a reproduction of the original pretraining corruption schedule.
4. **Check the unmodified output.** Before training, a zero-initialized adapter must leave the pretrained function unchanged.
5. **Update only the declared parameters.** I freeze the base model, optimize only adapter matrices, and compare a digest of frozen parameters before and after training.
6. **Record what happened.** I save adapter weights and a machine-readable report with the source revision, counts, losses, hashes, and interpretation boundary.

For sequence-level representations, `masked_mean` separately excludes padding and special tokens before pooling. It rejects sequences with no remaining residues. An unmasked mean would make representation magnitude depend on the amount of padding or the presence of bookkeeping tokens.

## The low-rank update, with dimensions

For a linear projection with input width d_in and output width d_out:

```text
W₀ has shape (d_out, d_in)
A  has shape (r, d_in)
B  has shape (d_out, r)
W  = W₀ + (alpha / r) B A
output = x W₀ᵀ + (alpha / r) x Aᵀ Bᵀ + bias
```

The trainable addition has `r(d_in + d_out)` parameters rather than `d_in × d_out`. I initialize A conventionally and B to zero, so the initial update is exactly zero. On the first backward pass, B can receive a nonzero gradient while A's gradient is zero. That is expected from the product rule, not evidence of a disconnected adapter. A can receive gradients after B changes.

I apply rank-4 updates to query and value projections in six attention layers: **12 adapted projections and 30,720 trainable parameters**. The report counts unique loaded parameters; tied weights and serialization counts need not match a model card's rounded parameter label.

The [unit tests](../tests/test_adaptation.py) compare the wrapper with the merged linear weight, verify the initial zero update, inspect gradient behavior, and confirm that frozen parameters remain unchanged. The full checkpoint run independently checks initial output equality and frozen-weight integrity.

## A failure that matters in practice: caches and autograd

During development, a pre-adaptation check under `torch.inference_mode()` created ESM rotary-position cache tensors that could not later be saved for backward. The model then failed when adaptation began. I use `torch.no_grad()` for those checks instead, so reusable buffers remain compatible with the subsequent gradient-tracked pass.

Three controls are easy to confuse:

- `model.eval()` changes behaviors such as dropout; it does not turn off autograd.
- `requires_grad_(False)` freezes selected parameters; it does not remove the need to differentiate activations leading to trainable adapters.
- `no_grad()` and `inference_mode()` disable recording, but inference tensors have stricter reuse constraints.

I keep the model in evaluation mode for this deterministic mechanics audit while enabling gradients for the adapter update. That is a deliberate fixture choice, not a recommendation for every fine-tuning experiment.

## What the recorded run establishes

The [recorded run](results/protein_transfer.json) used 27 supervised residue positions and three updates. Same-fixture loss changed from approximately **2.5497 to 2.4731**, the initial adapter-output difference was exactly zero, and the frozen-parameter digest remained unchanged. The adapter changed model outputs, as expected.

This establishes that the intended update path works on the pinned checkpoint. It does **not** establish improved protein function prediction, variant-effect prediction, structural accuracy, or clinical usefulness. I would not describe that training loss reduction as a biological discovery.

## Design the next study from the decision downward

Before comparing frozen embeddings, adapters, and full fine-tuning, I would predefine:

- a measured target and its units, uncertainty, censoring, and assay provenance;
- biological units and sequence-identity/family clusters that must stay together across splits;
- whether evaluation is interpolation within known families or transfer to new families, organisms, or assay conditions;
- a pretraining-overlap audit where feasible, with unresolved overlap stated explicitly;
- a simple sequence/property baseline, frozen-embedding baseline, and identical selection budgets;
- repeated training seeds, validation-only hyperparameter selection, and an untouched external or cluster-held-out evaluation;
- calibration, uncertainty, error slices, assay capacity, and the consequences of selecting an incorrect candidate.

The business case follows from the intended decision. A triage model could reduce unnecessary assays, but only if prospective evaluation shows that it preserves the desired hit yield under the actual assay budget. Parameter efficiency and a lower training loss alone do not establish that value.

## Sources

- Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models*, ICLR 2022: [primary publication](https://www.microsoft.com/en-us/research/publication/lora-low-rank-adaptation-of-large-language-models/). I implement a small inspectable linear adapter; I do not claim LoRA as a new method.
- Lin et al., *Evolutionary-scale prediction of atomic-level protein structure with a language model*, Science 2023, DOI [10.1126/science.ade2574](https://www.science.org/doi/10.1126/science.ade2574). ESM-2 is the language-model family; this example is not ESMFold and does not output a protein structure.
- [Pinned model's model card](https://huggingface.co/facebook/esm2_t6_8M_UR50D).
- [UniProt P69905 sequence](https://rest.uniprot.org/uniprotkb/P69905.fasta), [P68871 sequence](https://rest.uniprot.org/uniprotkb/P68871.fasta), and [UniProt reuse terms](https://www.uniprot.org/help/license). Retained sequences are credited to the UniProt Consortium under CC BY 4.0.
