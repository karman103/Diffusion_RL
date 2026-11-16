# Can group-relative RL change an RFdiffusion3 sampler?

I adapted a released RFdiffusion3 checkpoint for a small, PDB-conditioned experiment. The point was to test a concrete question: can a terminal Cα geometry reward update the *diffusion sampling policy* through a Flow-GRPO-style objective, while keeping the large pretrained denoiser frozen? The answer from this local run is split. The policy and gradient calculations worked, and a real checkpoint changed. The measured output effect was essentially zero.

## Why the sampler is the policy

In a diffusion generator, a structure is the endpoint of many denoising decisions. A terminal reward gives no direct label for any one step, but a stochastic reverse process lets us record each transition and compute its conditional probability. [Flow-GRPO](https://arxiv.org/html/2505.05470) uses this idea for flow-matching models: sample a group of outputs under the same condition, normalize their rewards within the group, and optimize clipped per-step action ratios with a reference-policy KL term.

RFdiffusion3 is not the rectified-flow model in that paper. Its released inference path uses an EDM-style denoiser and stock churn. I therefore implemented a **VE reverse-SDE family** from the denoiser's approximate score, following the [score-SDE framework](https://arxiv.org/abs/2011.13456) and [EDM](https://arxiv.org/abs/2206.00364). For denoiser output \(D_\theta(x,\sigma)\), the approximate score is \((D_\theta-x)/\sigma^2\). At noise strength \(a\), my smaller-\(\sigma\) Euler step has mean

\[
\mu_\theta=x+(1+a^2)\,\Delta\sigma\,(x-D_\theta)/\sigma
\]

and variance \(2a^2\sigma|\Delta\sigma|\) on unfixed coordinates. At \(a=0\), the stochastic term vanishes and the mean reduces to the corresponding EDM Euler probability-flow step. The family has the usual continuous-time marginal-equivalence argument **only for an exact score**. A learned score, atom masks, conditioning and finite steps break any guarantee of exact equivalence. In particular, this is *not* the stock RFD3 sampler and is *not* the literal rectified-flow conversion from Flow-GRPO.

## What actually learns

The 168-million-parameter pretrained denoiser stayed frozen. A 786-parameter, rank-two adapter with three noise-level bands modified only the **free Cα coordinate means** on the final recycling pass. Fixed motif atoms remained clamped. This is a deliberately narrow control surface: it avoids updating the entire structure network and makes it possible to inspect where the policy changes along the noise schedule.

With nonzero SDE variance, each free-coordinate action has a Gaussian density. At a recorded state, the old/new log ratio and same-variance reference KL are analytic. The ratio is exact **for this conditional transition**; the clipped, step-averaged GRPO objective is a surrogate, and I did not compute an exact probability for the final protein structure. Group-relative advantages came from four sibling trajectories under the same 1CRN condition. The terminal score penalized radius of gyration, Cα bond-length error and nonlocal Cα clashes; it never used native coordinates for the generated gap.

The local run made three group updates, twelve training trajectories total. Advantages varied within each group, adapter weights moved, and the frozen backbone received zero gradients. Numerically, the updates were small: the final adapter delta norm was 0.00259 and the largest logged within-update absolute log ratio was 0.00854. Those checks establish that the implementation trained; they do not establish that it improved a structure.

## A benchmark with a null result and missing data

I predeclared three public structures: [1CRN](https://www.rcsb.org/structure/1CRN) for RL, with [1PGA](https://www.rcsb.org/structure/1PGA) and [1UBQ](https://www.rcsb.org/structure/1UBQ) held out from this RL run. In each, chain-A residues 1–14 and 23–40 were fixed, and RFD3 generated eight residues between them. Four new seeds per structure were specified before evaluation. The primary comparison was trained versus zero adapter under the **same VE sampler and seed**. Stock RFD3 was a separate sampler reference, not an RL-effect control.

The predeclared deadline stopped evaluation after **17 of 36** samples: all twelve 1CRN outputs, five 1PGA outputs, and no 1UBQ outputs. On the four complete 1CRN pairs, trained-minus-zero mean guarded score was **−0.00000005**, with a seed-bootstrap interval of **[−0.00000425, +0.00000460]**. Mean gap-to-native Cα RMSD changed by **−0.000022 Å**, interval **[−0.000068, +0.000024] Å**. Those effects are negligible next to seed-to-seed variation. The sole 1PGA pair also differed only at the sixth decimal place in score; one pair cannot support a held-out claim. The plotted data and all saved coordinates are in [the results report](RESULTS.md) and [paired plot](paired_effects.png).

All 17 saved raw coordinate arrays were finite. Recomputed Cα rewards and native-gap alignments matched the stored values; fixed-motif RMSD was under 0.001 Å. Eight numerical/unit tests passed. Gap-to-native RMSD is only descriptive here because the generated gap has no specified sequence or unique native answer. These Cα heuristics say nothing about all-atom validity, sequence designability, stability or function. “Held out” refers only to this RL update; I have not established whether the PDBs were absent from RFD3 pretraining.

The useful outcome is a reproducible boundary: a real RFD3 checkpoint can be wrapped in a stochastic, conditional-density policy and optimized with group-relative rewards, yet this tiny, strongly regularized adapter run produced no visible geometric improvement. A next study would need a sharper task reward, more than one training condition, enough fresh-seed evaluation to complete the held-out protocol, and independent structure-quality checks. Those are proposals, not results from this run.
