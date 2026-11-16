# GRPO inside RFdiffusion3: what is the policy actually allowed to change?

*Completed local proof of concept on a pinned, pretrained RFdiffusion3 checkpoint. Six queued training comparisons, a real-model smoke test, and a 16-seed fresh-noise evaluation are recorded in [RESULTS.md](RESULTS.md) and the [fresh-noise report](heldout_3000000_n16/RESULTS.md). The observed reward changes are small and do not establish improved protein design.*

I wanted to test reinforcement learning on a pretrained protein diffusion model, with enough visibility into the implementation to explain where the learning signal goes. The resulting experiment uses RFdiffusion3 and a small GRPO adapter on its denoising coordinate head.

The central problem turned out to be the policy definition. It is easy to generate several structures, score them, and call the result GRPO. It is harder to make sure that the likelihood used for the update describes the action that the sampler actually took.

The method follows three published ideas. [DDPO](https://arxiv.org/html/2305.13301) treats each denoising transition as a policy action and uses its likelihood for policy gradients. [Flow-GRPO](https://arxiv.org/html/2505.05470) supplies the same-condition group advantage and clipped, stepwise update. [DPOK](https://arxiv.org/html/2305.16381) motivates KL regularization toward a pretrained reference. The [literature audit](LITERATURE_AUDIT.md) maps those ideas to the exact code and states where this RFD3 sampler departs from the published methods. In particular, Flow-GRPO's marginal-preserving ODE-to-SDE derivation was **not** transferred to RFD3.

## Starting with the sampler

RFD3's inference procedure adds Gaussian churn noise to the coordinates, evaluates a nonlinear denoiser at the resulting coordinates, and then performs an Euler-style update. Adding Gaussian noise before a nonlinear function does not generally make its output Gaussian.

That rules out a tempting shortcut: treating the final coordinates as a Gaussian draw around the denoiser's output using the churn variance. The resulting log probability would describe a different sampler.

I instead defined an explicit stochastic policy. Stock churn produces an auxiliary noisy state. At selected noise levels, a model-dependent mean is followed by an independent Gaussian action on free Cα coordinates. Those are also the coordinates that the trainable adapter can change. Other atom coordinates follow the shared frozen update at that state.

The conditional action density is then tractable. The augmented chain contains a parameter-independent churn transition followed by the learned Gaussian action. When comparing policies at the same recorded state and churn draw, the churn factors cancel. The code does not claim that the marginal coordinate transition is Gaussian, or that the modified sampler preserves the stock sampler's output distribution.

This distinction requires three comparisons: stock RFD3, the modified sampler with a zero adapter, and the modified sampler after training. Otherwise an apparent improvement could come entirely from adding exploration noise.

## A small intervention in the real denoiser

The checkpoint contains 168,038,994 network parameters. The first adaptation targets the final atom-coordinate projection, `to_r_update[-1]`, rather than backpropagating through the entire network.

The projection takes 128 decoder features and produces three coordinate-update components. A rank-two adapter adds a correction of the form

\[
\Delta R=B_\sigma A_\sigma h.
\]

There are three sets of low-rank weights, selected by the denoising noise level: below 2, from 2 to 8, and above 8. Their total size is 786 trainable parameters. This gives the experiment a concrete question: does allowing different corrections at different noise levels help more than sharing one correction everywhere?

The single-adapter comparison has 262 parameters. A difference between those configurations would therefore combine specialization and parameter capacity. A capacity-matched experiment would be needed to attribute a gain to specialization alone.

The adapter is applied only on the final recycling pass. Earlier recycling uses the original checkpoint. At a recorded noisy state, the final decoder features are therefore independent of the adapter weights and can be cached. GRPO can replay those features through the small adapter without retaining an RFD3 autograd graph.

That shortcut has a testable consequence. Injecting the same correction into the actual final projection should agree with the cached calculation. On the real checkpoint, the maximum coordinate discrepancy was approximately **6.2 × 10⁻⁶ Å**. Separately, the wrapper's stock path matched the original sampler under shared random draws to approximately **3.8 × 10⁻⁵ Å**.

These checks establish implementation parity. They do not establish a better protein generator.

## Scaling the correction correctly

The projection does not directly emit the final physical coordinates. RFD3 applies EDM output preconditioning, and the sampler then applies its coordinate-update coefficient.

The adapter correction must pass through both operations before it becomes a change in the policy mean. The implementation uses

\[
\Delta\mu=-\mathrm{step\_scale}\frac{\Delta t}{\hat\sigma}
\frac{\sigma_\mathrm{data}\hat\sigma}{\sqrt{\sigma_\mathrm{data}^2+\hat\sigma^2}}
B_{\hat\sigma}A_{\hat\sigma}h.
\]

Applying the raw adapter output directly to coordinates would define a different intervention. It would also change the magnitude of the likelihood gradient across noise levels.

## Group-relative updates

The main experiments generate four independent siblings under identical input features and length. Their rewards are centered and divided by the population standard deviation within that group. Tied groups receive zero advantages. Different conditions are never mixed to construct a baseline.

The resulting advantages weight a clipped policy objective over recorded stochastic steps. A closed-form conditional Gaussian KL penalty anchors the adapter to the frozen reference. There is no learned value network.

The probability calculation sums over all stochastic Cα coordinates. Averaging coordinate log probabilities inside the exponential would change the importance ratio and its clipping behavior. The implementation also computes the ratio from differences in means, avoiding the numerical cancellation that can arise from subtracting large Gaussian log-density constants.

The action likelihood is exact for the declared conditional Gaussian policy. The clipped training objective is a surrogate averaged over collected steps; it is not an exact importance-weighted estimate of the entire trajectory's output distribution. Training KL is evaluated at collected states. Evaluation separately accumulates conditional KL on fresh adapted-policy trajectories.

## What the executed runs show

The real-checkpoint smoke completed one GRPO epoch with four reuse passes. Its two siblings had advantages of +1 and −1. The adapter's effective weight-delta norm became approximately **0.000232**, while zero frozen-network tensors received gradients.

The smoke used an eight-point schedule and 16 residues. Its geometry was poor, so its tiny reward change is not evidence of useful design improvement.

A subsequent calibration used 64 schedule points and 32 residues. Both stock and modified samplers passed the declared Cα geometry heuristic in two samples each. Mean adjacent-distance errors were approximately **0.0306 Å** for stock sampling and **0.0336 Å** for the modified zero-adapter sampler. Both length and schedule changed between smoke and calibration; this is not a controlled experiment attributing the difference to schedule alone.

The calibration is an operational check for proceeding. Two successful samples do not estimate a population success rate with useful precision.

The first full run trained the 786-parameter adapter for six group updates, using 24 sampled sibling trajectories. Its effective weight-delta norm reached **0.00347**, while no frozen-network tensor received a gradient. On four fixed evaluation noise seeds, the guarded reward moved from **−0.97604** with a zero adapter to **−0.97725** after training. Higher reward is better, so this is a small deterioration of **0.00121**. Mean adjacent-distance error also rose from **0.03141 Å** to **0.03191 Å**; all four samples still met the Cα geometry heuristic. The observed reward change is too small and the evaluation set too narrow to conclude that this method helps or hurts generally.

The training objective did perform numerically stable updates: its final reuse pass had a maximum absolute log ratio of **0.153** and no clipped transitions; the evaluated policy had an accumulated conditional path KL of **0.743 nats** from the zero adapter. Those measurements show a learned policy change, not a demonstrated quality gain.

The random-advantage control has now also completed six updates on the same input and evaluation seeds. Its common guarded score moved from **−0.97604** to **−0.97599** (approximately **+0.00006**) on four samples. This does not establish that random updates are better: both changes are very small, and the control is one training run. It does show that the first reward-directed run did not beat this drift control on the declared score. The control changed the policy substantially too, with evaluated path KL **1.056 nats**. Across the two runs, a policy change alone cannot be read as reward improvement.

All three reward-directed training seeds have now finished. Their paired changes in guarded score on the same four fixed evaluation noise seeds were **−0.00121**, **−0.00312** and **−0.00076**. The mean change across these three training runs was **−0.00170**. Higher is better, so each run moved downward on this narrow evaluation; all twelve post-training samples still met the Cα geometry heuristic. The random-advantage control changed by approximately **+0.00006**. This is an observed failure to improve the declared score in this small setup, not evidence that GRPO cannot help RFD3 generally.

The single-adapter comparison has now completed. Using one rank-two adapter across all active noise levels, its paired guarded score changed by approximately **+0.00010** on four samples, with evaluated path KL **0.130 nats**. The three-adapter seed-zero run changed by **−0.00121** and reached path KL **0.743 nats**. The shared adapter moved the model less and had 262 trainable parameters versus 786; this comparison cannot isolate the value of noise specialization from capacity or update magnitude. Its small positive score change is also too narrow to claim a gain.

The final ablation allowed the three noise experts to act only at σ≤4, reducing recorded stochastic transitions from 56 to 20 per four-sibling group. On its own zero-adapter modified sampler, the guarded reward rose from **−0.97371** to **−0.96913**, a paired change of **+0.00458** on four samples; its path KL was only **0.0093 nats** and all four samples passed Cα geometry. This was the most favorable four-sample result in the queue, but the bootstrap interval includes zero and the sampler itself differs because action noise is confined to a narrower window. Its own zero-adapter baseline is the appropriate comparison.

The fresh-noise evaluation used 16 new shared noise seeds, with no additional training. Against the full-window zero adapter, the three reward-directed adapters changed guarded reward by **−0.00179**, **+0.00047** and **−0.00058**. Averaging those three adapter outcomes per seed gives **−0.00063**, with a bootstrap 95% interval across the 16 noise seeds of approximately **[−0.00210, +0.00096]**. The random-advantage control changed by **+0.00070**. The late-window adapter changed reward by only **+0.00031** against its separately sampled, matched late-window zero adapter; its interval was **[−0.00003, +0.00095]**. The earlier +0.00458 estimate did not carry over as an appreciable average gain. All 144 audited Cα traces passed the declared geometry heuristic. The full-window zero baseline reward had a sample standard deviation near **0.074**, much larger than these mean policy differences; paired seeds are essential for seeing the small changes. These findings still cover one unconditional length and one proxy reward.

I then froze every adapter and ran a predeclared transfer check at [24 residues](cross_length/length24/RESULTS.md) and [40 residues](cross_length/length40/RESULTS.md), eight new shared noise seeds at each length. The three reward-directed adapters averaged **−0.00098** and **−0.00034** guarded reward change versus the full-window zero adapter at lengths 24 and 40, respectively; both bootstrap intervals included zero. The late-window adapter's matched-baseline changes were approximately **0.00000** and **+0.00039**. All 144 transfer samples passed the Cα geometry heuristic. These checks broaden the tested lengths, but they remain three unconditional inputs and do not establish performance on target-conditioned protein design.

## Interpreting the comparison

The completed queue used 32 residues, 64 schedule points and four siblings per group. It included three reward-directed training seeds, a random-advantage control, a shared-adapter comparison and a restricted noise-window comparison. Each run records the checkpoint and source hashes, exact sampler configuration, source snapshots, group rewards and advantages, clipping diagnostics, weight changes and raw coordinates.

The proxy reward combines compactness with penalties for adjacent Cα-distance errors and nonlocal Cα clashes. Evaluation also keeps those metrics separate so a reward gain cannot conceal the particular constraint that deteriorated. The random-control training advantages are drawn independently of geometry; its evaluation log uses compactness alone. The results report recomputes the same guarded reward from saved coordinates for all runs, allowing an outcome comparison on one scale.

These measurements address a geometric adaptation problem. They do not measure sequence designability, folding agreement, novelty or all-atom physical validity. Those require additional evaluation stages.

The three reward-directed updates did not show a consistent gain over their zero adapter or random drift on fresh noise. The late-window result suggests a testable hypothesis—that restricting the intervention can limit policy movement—but its heldout mean change is too small to establish an improvement. The defensible result is the reproducible method and its measured failure to improve this proxy reward under the tested budget. A follow-up would need stronger conditioning tasks, more independent targets and a reward tied to the intended design objective.

## What I would change next

The current adapter changes the denoising readout. A deeper intervention would place low-rank adapters in selected diffusion-transformer or final atom-decoder attention projections. Those adapters would change hidden states, making the present cached-head replay inexact. They would need their own backward pass, memory strategy and parity checks.

Another experiment would branch a group from a shared frozen denoising prefix. That could reduce repeated inference work and compare local refinements of the same intermediate structure. It would optimize a conditional refinement problem, which should be stated explicitly.

Neither extension is claimed as an executed result here. The current code and measurements are described in [METHOD.md](METHOD.md), with reproduction instructions in [README.md](README.md).
