# GRPO inside RFdiffusion3's denoiser

Implementation status and actual measurements are recorded in `STATUS.md` and each run directory. The [published-method audit](LITERATURE_AUDIT.md) maps the design to DDPO, Flow-GRPO, DPOK, EDM, LoRA and the RFD3 release, while identifying our RFD3-specific choices. The equations and numerical unit checks below are implemented; pretrained execution is a separate check and is never inferred from those tests.

**Executed:** `runs/smoke_v3` loaded the official 168,038,994-parameter checkpoint and completed a GRPO update to the 786 adapter parameters. The resulting weight-delta norm was 0.0002321, with zero frozen-network gradient tensors. Cached-head and stock-sampler parity errors were 6.20e-6 and 3.81e-5 Å. Its deliberately tiny eight-point schedule had poor geometry and establishes implementation correctness only. The subsequent 64-point, 32-residue calibration passed the declared Cα geometry checks for both stock and modified samplers (two samples each). Six longer training comparisons and fresh-noise evaluations at lengths 24, 32 and 40 completed. They showed no consistent gain on the declared Cα geometry reward; measured outcomes are in [BLOG.md](BLOG.md) and [RESULTS.md](RESULTS.md).

## What receives a gradient

The target is the final projection in `RFD3DiffusionModule.to_r_update[-1]`, which maps 128 atom features to a three-coordinate denoising update. A rank-two adapter predicts `ΔR = Bσ Aσ h` from its normalized decoder features. Three experts cover noise levels below2, 2..8 and above8. Their combined size is786 parameters when the checkpoint uses128 atom channels.

The original projection, conditioning initializer, atom encoder, diffusion transformer, atom decoder and sequence head are frozen. The adapter acts only on free Cα atoms, only inside the configured noise window, and only on the final recycling pass. This is a narrowly defined denoising-head fine-tune; the deeper transformer is not updated in this first experiment.

The last-pass restriction is useful. Earlier recycling still uses the original checkpoint, so its self-conditioning stays fixed at a recorded state. The last decoder features can then be cached. Replaying GRPO losses requires only the small adapter, rather than keeping a full RFD3 autograd graph. A real-model test injects the same delta into the actual final projection and compares outputs to the cached calculation.

## Why the training sampler is explicit

The [RFD3 inference sampler](https://github.com/fnachon/foundry/blob/7a9c246fbf140a743d14a35248f5039e7723a23a/models/rfd3/src/rfd3/model/inference_sampler.py) adds Gaussian churn noise **before** the nonlinear denoiser. Its final coordinate update is therefore a nonlinear pushforward, whose density cannot generally be replaced by a simple Gaussian density.

The implemented policy keeps that stock churn as a parameter-independent auxiliary transition. After the Euler update, selected steps receive an additional independent Gaussian action on the free Cα coordinates:

\[
z_t=x_t+\epsilon_{\mathrm{churn}},\qquad
a_t\sim\mathcal N(\mu_\theta(z_t),s_t^2 I).
\]

Other atoms follow a shared deterministic update at that visited state. The adapter also affects only the noisy Cα subspace. Consequently, deterministic coordinates do not acquire a parameter-dependent transition with zero variance. Outside the active window, all transitions use the frozen reference map.

This defines a new, tractable stochastic sampler. There is **no claim** that it preserves the stock sampler's marginal distributions. Three comparisons are needed: stock RFD3, the modified sampler with a zero adapter, and the modified sampler after GRPO. An implementation parity check additionally matches the wrapper's stock path to the original sampler under identical random draws.

This differs from adopting [Flow-GRPO's](https://arxiv.org/abs/2505.05470) marginal-preserving ODE-to-SDE construction. RFD3's EDM preconditioning, churn, atom masking and inference step scale need their own derivation before such a claim would be justified.

## EDM-aware gradient scaling

RFD3 predicts an update that is converted into coordinates through

\[
D_\theta(z,\sigma)=c_{\mathrm{skip}}(\sigma)z+
c_{\mathrm{out}}(\sigma)R_\theta(z,\sigma),\quad
c_{\mathrm{out}}=\frac{\sigma_{\mathrm{data}}\sigma}{\sqrt{\sigma_{\mathrm{data}}^2+\sigma^2}}.
\]

The adapter delta must pass through both this EDM output coefficient and the Euler step coefficient. The cached policy-mean correction is

\[
\Delta\mu=-\mathrm{step\_scale}\frac{\Delta t}{\hat\sigma}
c_{\mathrm{out}}(\hat\sigma)\,B_{\hat\sigma}A_{\hat\sigma}h.
\]

Applying the raw adapter output directly to coordinates would train a different intervention and give the wrong magnitude to its likelihood gradient.

## Group-relative learning

Following [Flow-GRPO's group-relative advantage](https://arxiv.org/html/2505.05470) and [DDPO's sequential denoising policy view](https://arxiv.org/html/2305.13301), each group contains independent trajectories for exactly the same RFD3 input features and length. For rewards `r₁,…,rG`,

\[
A_i=\begin{cases}
(r_i-\bar r)/s_r, & s_r>\varepsilon,\\
0, & s_r\le\varepsilon.
\end{cases}
\]

The code uses population standard deviation and returns zero advantages for a tied group. It never normalizes across different conditions. There is no learned value network.

For each recorded stochastic step, it computes the exact **joint** Gaussian likelihood ratio over all free Cα coordinates. The PPO/GRPO clipped objective is combined with a closed-form conditional Gaussian KL penalty to the frozen reference, motivated also by [DPOK](https://arxiv.org/html/2305.16381). Four reuse passes are the default. Recorded diagnostics include group reward spread, advantages, clipping frequency, likelihood ratios, conditional KL, gradient norm and nonzero adapter-weight change. The four-sibling group and fixed KL coefficient are local POC settings, not published RFD3 hyperparameters.

The log ratio is calculated from mean differences directly, avoiding cancellation from subtracting large Gaussian normalization constants. It is summed over coordinates, not averaged inside the exponential. This keeps the stated clipping threshold tied to the actual joint action distribution.

The exact quantity here is the conditional Gaussian action density at a recorded augmented state. The clipped objective remains a step-averaged policy surrogate, rather than an exact whole-trajectory importance-weighted output objective. The parameter-independent churn factors cancel at the same recorded states and auxiliary draws; integrating over churn does not generally yield a Gaussian marginal coordinate transition. Training KL is evaluated on collected states, while evaluation accumulates conditional KL along adapted-policy trajectories.

## What makes the study useful

1. **Noise-level specialization:** early, middle and late denoising can learn different small corrections. Band-specific deltas allow an ablation of where reward improves and where geometry degrades.
2. **Last-recycle adaptation:** retaining the earlier frozen recycles gives an exact, cheap replay calculation and a clean comparison with modifying the actual head.
3. **A declared action subspace:** a Cα intervention reduces the number of stochastic action dimensions while the pretrained all-atom model still supplies the surrounding context and subsequent denoising.
4. **Sampler controls:** a change caused by added exploration noise is separated from a change caused by learned weights.
5. **Reward controls:** compactness and a geometry-penalized reward can be compared with random group rewards and best-of-group selection. Better proxy reward alone is insufficient.

The first reward uses radius of gyration plus adjacent Cα-spacing and clash penalties. These are geometry diagnostics. Sequence design, independent folding evaluation, all-atom validity and experimental function require separate measurements.

## Extensions, not implemented results

The next deeper adaptation would place LoRA in selected diffusion-transformer or final atom-decoder attention projections, with activation checkpointing and explicit backpropagation. Those hidden states would then change with the adapter, so the current cached-head shortcut would no longer be exact. Time-conditioning/AdaLN adapters and shared-prefix group branching are additional experiments to test, not capabilities claimed by this first run.
