# Does distributed training still optimize the same objective?

I use a small exact comparison before trusting a distributed training loop. If a single-process calculation and a distributed calculation disagree on a controlled fixture, scaling up only makes the disagreement more expensive to investigate.

The example uses four synthetic token representations, split into shards of one and three supervised tokens. It is a numerical test of the objective—not a biological performance result.

```bash
python -m studies.distributed_training
python -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=2 -m studies.distributed_training --ddp
```

The second command starts two local CPU processes with Gloo. It does not allocate cloud machines or GPUs. [Recorded two-process result](results/distributed.json).

## Begin with the estimand

Suppose rank r has n_r valid supervised tokens and a summed loss S_r. If I intend every valid token to receive equal weight, the objective is:

```text
L = (sum over ranks S_r) / (sum over ranks n_r)
```

Averaging local means instead gives:

```text
L_wrong = (1 / R) × sum over ranks (S_r / n_r)
```

These are equal only under particular conditions, including equal valid-token counts. Equal sequence counts are not enough when sequence lengths or masking patterns differ. A short sequence and a long sequence also need not receive equal weight unless that is the scientific objective I intend.

## Account for DDP's gradient averaging

Ordinary PyTorch DistributedDataParallel averages synchronized gradients across R ranks. To recover the gradient of the global valid-token mean, each rank backpropagates:

```text
rank-local backward quantity = R × S_r / N
where N = sum over ranks n_r
```

After DDP averages gradients, the extra R cancels its averaging factor. I reduce the **counts** globally before backward and use summed local losses. I do not divide each local loss by its local token count and then hope DDP repairs the weighting.

The [implementation](distributed_training.py) is intentionally CPU/Gloo-specific. A CUDA/NCCL version must place collective tensors on an appropriate device and preserve the same algebra; merely switching the backend string is not enough.

## Accumulation has the same denominator problem

Splitting a logical batch into microbatches does not justify dividing every microbatch by an arbitrary accumulation-step count. The denominator should reflect the intended logical batch's valid observations. A short final microbatch is a common place to get this wrong.

I accumulate sums using the global denominator for the entire logical batch. In DDP, both forward and backward are inside `no_sync()` for nonfinal local microbatches, followed by a synchronized final pass. Every rank still needs to participate. A rank with no valid tokens needs a graph-connected dummy masked batch; a globally empty supervised batch is rejected rather than producing a NaN mean.

I clip gradients once after accumulation in a real optimizer loop, not independently per microbatch. With automatic mixed precision, unscale before clipping and checkpoint the scaler when applicable. These ordering details change the update, even when every individual API call succeeds.

## Read the counterexample

The recorded fixture gives a maximum gradient error of approximately **0.25834** for the naïve mean of rank-local means. The globally weighted result and accumulated result agree with the single-process reference to approximately **2.78 × 10⁻¹⁷**. Both actual DDP ranks also agree to that precision.

The agreement is the useful outcome. I am not using the small residual to imply universal distributed correctness: dropout, BatchNorm, stochastic data transforms, different precision, nondeterministic kernels, and sampler behavior introduce additional questions. The fixture removes those complications so it can isolate the denominator error.

## Checkpoint recovery requires more than a weight file

For exact continuation, a training checkpoint generally needs model and optimizer state, scheduler and scaler state when used, epoch/step, RNG state, sampler position, preprocessing state, data version, and the distributed topology assumptions. A weights-only artifact is appropriate for inference, but is not automatically a resumable training checkpoint.

I separate the repository's inference export from that stronger recovery claim. The [export walkthrough](engineering.md) tests input-schema and numerical parity. The biological adapter report records a weight update and provenance, but it does not claim fault-tolerant multi-node training recovery.

For a production research workload, I would test interruption at known boundaries, restart on a clean process group, and compare the resumed update sequence with an uninterrupted reference. I would also test duplicated data after sampler restart, rank loss, exhausted retry budgets, and incomplete checkpoint writes. Those are separate acceptance tests, not consequences of a passing gradient-equivalence test.

## Decide the scientific weighting before optimizing the system

Token weighting is not always the right biological weighting. If the scientific unit is a protein, donor, specimen, or assay plate, a hierarchical objective may be appropriate. The software must implement that declared weighting; a distributed convenience should not choose it accidentally.

That is the connection between top-down and bottom-up reasoning here. From the top, I specify which observations should influence the result and why. From the bottom, I derive the reduction, inspect gradients, and test an adversarially uneven fixture. If those two descriptions do not meet, I stop before spending more compute.
