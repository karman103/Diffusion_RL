# RFdiffusion3: source-audited integration plan

**Historical pre-implementation audit.** The user subsequently requested RFD3 GRPO directly. The [new implementation](../rfd3_grpo/README.md) now has a real pretrained-checkpoint smoke update, stock-sampler parity and a denoising-head equivalence check. Its [results](../rfd3_grpo/RESULTS.md) are separate from Genie. The plan below records the original sampler audit and extensions that still require their own validation.

Audit revision: [`RosettaCommons/foundry@0d0b0ba`](https://github.com/RosettaCommons/foundry/tree/0d0b0ba44fab7ca76aa4b625c014ce5964c7d284). The exact source files inspected are saved under `references/`.

## Why a direct DDPO wrapper would be wrong

In [`models/rfd3/src/rfd3/model/inference_sampler.py`](https://github.com/RosettaCommons/foundry/blob/0d0b0ba44fab7ca76aa4b625c014ce5964c7d284/models/rfd3/src/rfd3/model/inference_sampler.py), the default sampler does the following:

1. Optionally realigns and augments the structure.
2. Adds Gaussian churn noise to the current coordinates, excluding fixed atoms.
3. Evaluates the denoiser at those **already-noised** coordinates.
4. Applies an Euler-style coordinate update with a configurable step scale.

Abstractly, `z = x + sigma * epsilon; y = F_theta(z)`. The distribution of `y | x` is a nonlinear pushforward of a Gaussian. It is generally **not** `Normal(F_theta(x), sigma² I)`. A log-probability computed with the latter would not be the sampler's likelihood. Treating the churn noise as the action does not fix ordinary score-function training either: its distribution is independent of theta, while the downstream transition depends on theta. A pathwise derivative or a different stochastic policy is then required.

The same source asserts that gradients are disabled during this inference routine. Removing that assertion alone is not a differentiable training implementation. Fixed-atom masks, guidance, possible realignment, the terminal readout, and saved versus actual coordinates all matter.

## Two legitimate extensions to investigate

**A. Explicit Gaussian coordinate policy.** Build a separate training sampler with a model-dependent mean evaluated at a fixed current state, followed by independent nonzero Gaussian noise on free coordinates. Record the exact mean, variance, free-atom mask and actions. For a degenerate mask, compute densities only in the free-coordinate subspace. Keep auxiliary random transformations explicit or disable them during the initial audit. Freeze any deterministic terminal decoding map shared by the reference and adapted policies. Compare this sampler against the unmodified inference sampler before attributing changes to RL. This is a modified RFD3 sampler, not an exact likelihood wrapper around the current one.

**B. Reward-weighted supervised adaptation.** Generate and score independent candidates, then adapt with the model's documented training loss and an explicit weighting rule. Compare against best-of-N and uniform-weight training. This does not require an exact reverse-transition likelihood, but it is a different objective from DDPO and may have different failure modes.

For either path, begin with unconditional short monomers and geometry diagnostics. Then add inverse folding and an independent folding evaluation, count failed designs as failures, and report sequence recovery/designability separately from the optimized proxy. An all-atom representation introduces side-chain and steric constraints absent from this Cα-only POC.

## Concrete first inference configuration, not yet executed

The upstream [RFD3 README](https://github.com/RosettaCommons/foundry/blob/0d0b0ba44fab7ca76aa4b625c014ce5964c7d284/models/rfd3/README.md) documents `pip install 'rc-foundry[rfd3]'`, checkpoint installation with `foundry install rfd3 --checkpoint-dir ...`, and the `rfd3 design` CLI. The [Foundry README](https://github.com/RosettaCommons/foundry/blob/0d0b0ba44fab7ca76aa4b625c014ce5964c7d284/README.md) describes a community Apple-MPS fork, float32 precision and inference support. Those installation routes must be pinned and validated separately; this project does not present them as a tested environment.

Use a clean environment, a checkpoint directory inside the project and an unconditional input such as `{"unconditional_64": {"length": "64"}}`, validated against the pinned input specification. Start with `diffusion_batch_size=1`, `n_batches=1`, `low_memory_mode=True`, and no trajectory dumping. Record the resolved configuration, model/source hashes, peak memory and runtime. Keep the original inference sampler for this first parity run. Low precision is a hardware-dependent optimization, not a precondition for this POC.

## Acceptance criteria before claiming RFD3 RL

- An unmodified checkpoint produces structures through a pinned, reproducible inference command.
- The chosen stochastic transition matches its asserted density, verified numerically.
- At initialization, adapted and reference policies match under shared randomness.
- Frozen/masked coordinates and deterministic readout are handled consistently.
- Independent training seeds, an unchanged baseline, a random-reward control and best-of-N are run with explicit compute accounting.
- Geometry, foldability/designability and novelty are reported as distinct evaluations.

None of those unexecuted RFD3 checks are substituted by the Genie results.
