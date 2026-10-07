# PyTorch investigations: test the training claim

These examples connect physiological time series, optimization, and representation learning. They are **experiments in implementation**, not validated healthcare systems. The useful question is not “does the network run?” but “does this operation change the quantity I think it changes?”

Start with the [architecture walkthrough](pytorch_systems_documentation.html), then test the smaller contracts below. The walkthrough describes broader ambitions; the contracts here describe the implemented, tested behavior.

From the repository root, in a virtual environment:

```bash
python -m pip install -r requirements-torch.txt
MPLBACKEND=Agg python -m pytest tests/test_torch.py -v
```

No GPU or patient dataset is required. Use small CPU examples before attempting the larger training demonstrations.

## Investigation 1: a padding mask is not a missing-data model

**Question:** If a sensor sequence is padded, can those padding values change the prediction?

[MedicalTimeSeriesEncoder](medical_timeseries_implementation.py) accepts `(batch, channels, time)` tensors, with `time` at most its configured `sequence_length`. The optional boolean `(batch, time)` mask uses **True for excluded positions**.

The contract is:

1. Replace excluded input positions with zero **before convolution**, so arbitrary values or NaNs in padding cannot contaminate valid features.
2. Exclude those positions as attention keys.
3. Exclude them from both global maximum and mean pooling. Average by the count of valid steps, not the padded sequence length.
4. Return zero representations for masked time steps, and reject sequences with no valid steps.
5. Reject nonfinite values at valid positions. A single missing channel is not the same as an absent time step; imputation and channel-level missingness indicators belong upstream.

**Predict, then test:** fill the padding with zeros, large numbers, and NaNs. Keep the mask fixed. The pooled representation must be unchanged, and gradients with respect to padded inputs must be zero.

**Important boundary:** value-invariance at a fixed mask is not invariance to adding more padding. Training-mode BatchNorm includes padded positions in its statistics. This encoder also uses ordinary positional indices, not measured elapsed time, and non-causal attention/convolutions. It does not natively solve irregular sampling or prospective forecasting. Those require a different observation model and appropriate validation.

## Investigation 2: class weights are not class probabilities

**Question:** How do you give a rare class half the sampling probability without giving every rare example the same probability as every common example?

[ImbalancedDatasetSampler](adaptive_gradient_scheduler.py) maintains a probability mass `p_c` for each **observed** class. An example in class `c`, with `n_c` observations, receives weight:

```text
w_i = p_c / n_c
sum of w_i over examples in class c = p_c
```

With 90 examples of one class and 10 of another, uniform class masses give each class probability 0.5. That is an expectation over sampling with replacement—not a guarantee of exactly balanced counts in every batch.

Labels must be nonnegative `int64` output indices. Gaps are supported: labels `{2, 5}` use class masses in sorted-label order and require prediction logits with at least six columns. Explicit initial weights are **class masses**, not per-example weights; this meaning differs from the original exploratory implementation.

After each observed class has accumulated more than 100 training observations, smoothed classification errors can shift class masses. A uniform mixture preserves a minimum class probability of `min_sample_rate / number_of_observed_classes`. Updates accept training logits only; adapting from validation errors would turn validation data into training feedback.

**Try to break it:** use a single class, highly unequal counts, labels with gaps, and invalid initial weights. Check the total probability assigned to each class analytically before judging a random batch.

**Limit:** rebalancing changes the effective training distribution. It does not calibrate probabilities to deployment prevalence or resolve unequal costs of clinical errors. Keep validation sampling representative of the intended population.

## Investigation 3: a loss term must have the intended gradient

**Question:** Can a “class separation” term change the loss value without changing model parameters?

Yes. A penalty computed entirely from detached historical buffers is a constant with respect to the current features. The current [FeatureSpaceRegularizer](adaptive_gradient_scheduler.py) uses differentiable **current-batch class means** for separation. Historical exponential-moving-average prototypes remain diagnostics and update only in training mode.

The objective is explicitly experimental:

- Normalize each feature row to unit length and form its Gram matrix `G = Z Zᵀ`.
- Obtain nonnegative eigenvalues `lambda` of `G`, clipping negative numerical roundoff to zero.
- For a nonzero spectrum, compute `p = softmax(log(max(lambda, machine_epsilon)) / temperature)`.
- Minimize the negative spectral entropy, `sum(p * log(p + 1e-10))`. By convention, a zero spectrum contributes zero, rather than being rewarded as maximally diverse.
- Normalize the current-batch class means. Penalize the largest between-class cosine similarity with `max(0, similarity + 0.5)`. With fewer than two observed classes, the separation penalty is zero.
- Combine negative entropy and half the separation penalty.

This **changes the exploratory objective**; it is not merely a numerical refactor. Orthogonal representations, repeated identical nonzero representations, and all-zero representations are useful counterexamples for checking its direction. Compare gradients with all labels equal versus two labels: the separation term must contribute a gradient when its hinge is active. Also check finite backward passes across repeated batches and unchanged prototype buffers in evaluation mode.

**Limits matter:** the margin and weighting are heuristic. Spectral decomposition adds cost; entropy depends on batch size and available rank. A cosine threshold of -0.5 cannot be simultaneously achieved by arbitrarily many class centroids. Missing classes receive no current-batch separation gradient. A mathematically consistent gradient is not evidence that this regularizer improves representation quality or calibration.

To evaluate that claim, compare a plain cross-entropy baseline against entropy-only, separation-only, and combined objectives using identical patient-disjoint splits, training budgets, and seeds. Report calibration and class-wise errors as well as discrimination. Do not select the final objective using held-out test results.

## Investigation 4: check the quantity the optimizer actually uses

[AdaptiveGradientScheduler](adaptive_gradient_scheduler.py) must put each returned warmup learning rate into the optimizer's parameter groups and record it in diagnostics. Printing a smaller number while leaving the optimizer unchanged is not warmup.

Validation loss in [TrainingPipeline](medical_timeseries_implementation.py) is weighted by the number of examples in each batch. Averaging batch means equally gives an incomplete final batch too much influence. Test a dataset with three observations and batch size two; compare the result with the loss computed on all three observations directly.

The synthetic generator assigns random risk labels and uses rolled signal targets. It is a mechanics fixture, **not evidence of risk prediction or a valid forecasting benchmark**.

## Synthesis

For each investigation, record:

- the invariant and a counterexample that violates it;
- the tensor shapes and unit of independence;
- the test or calculation that supports your interpretation;
- a claim that remains unsupported even after the test passes.

A useful final statement distinguishes **correct software behavior**, **a defensible statistical experiment**, and **evidence for a biological or clinical claim**. They are three different achievements.
