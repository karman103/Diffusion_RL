# Flow-GRPO-style RFdiffusion3 PDB benchmark

This is a new, PDB-conditioned experiment using the released RFD3 checkpoint. It follows [Flow-GRPO](https://arxiv.org/html/2505.05470) for group-normalized terminal rewards, per-step clipped probability ratios and reference KL. Because RFD3 is an EDM denoiser rather than the rectified-flow model in that paper, the stochastic sampler is derived from the [score-SDE / probability-flow formulation](https://arxiv.org/abs/2011.13456) and [EDM](https://arxiv.org/abs/2206.00364), **not copied from Flow-GRPO's rectified-flow equation**.

For a narrative account of the implementation and its null result, see the [finished blog post](BLOG_POST.md). The [earlier draft](BLOG_DRAFT.md) is kept as an editing record.

Let the approximate score be `s_theta(x,σ)=(D_theta(x,σ)-x)/σ²`. With noise level `a`, the reverse-VE family integrated toward smaller σ uses drift `(1+a²)(x-D_theta)/σ` and action standard deviation `sqrt(2 a² σ |Δσ|)`. At `a=0` its Euler mean reduces to the EDM probability-flow step with scale 1. The derivation gives a marginal-equivalent family in continuous time **if the score is exact**; the actual learned score, atom masking, finite Euler steps and RFD3 conditioning prevent an exact equivalence claim here. Stock RFD3 uses different churn and step scaling and is a separate comparison.

At each active step the policy samples a Gaussian action on unfixed atoms. The adapter changes only free Cα means, so the other action-density factors cancel in old/new ratios. An exact conditional Gaussian log ratio and KL are available at recorded states. This is still a clipped, step-averaged GRPO surrogate, not an exact output-density ratio. The pretrained 168-million-parameter denoiser stays frozen; a 786-parameter, three-noise-band rank-two LoRA changes the final coordinate update on the last recycling pass. EDM output scaling and the VE drift coefficient are included in the policy-mean delta.

## Predeclared benchmark

The machine-readable [protocol](PROTOCOL.json) was saved before training. Public RCSB structures [1CRN](https://www.rcsb.org/structure/1CRN) (train), [1PGA](https://www.rcsb.org/structure/1PGA) and [1UBQ](https://www.rcsb.org/structure/1UBQ) (held out) contribute chain-A residues 1–40. Residues 1–14 and 23–40 are fixed; RFD3 generates an eight-residue gap between them. The gap is de novo and has no specified sequence. One 16-step 1CRN smoke informed feasibility; the confirmatory protocol uses 32 schedule points, three groups of four siblings for training, and four new evaluation seeds per PDB. One checkpoint is trained only on 1CRN; no held-out PDB coordinates enter its reward.

Primary comparisons: trained VE-SDE policy versus the **same VE-SDE sampler with a zero adapter**, paired by noise seed, on guarded Cα geometry reward and loop bond error. Stock RFD3 is an independent sampler baseline. Gap-to-native Cα RMSD after alignment on fixed residues is a separate descriptive check; with an unspecified gap sequence, native coordinates are not a unique correct answer. Fixed-motif RMSD checks the input parsing. The two test PDBs are held out from **this RL fine-tuning run**, not known to be absent from RFD3's original pretraining data. These are small-sample geometric benchmarks, not tests of folding, designability or function.

## Reproduce

The checkpoint and Apple-MPS Foundry fork are documented in [the earlier RFD3 report](../rfd3_grpo/FINAL_REPORT.md). The new code keeps earlier run folders untouched.

```bash
.venv-rfd3/bin/python -m unittest rfd3_grpo.test_grpo -v
.venv-rfd3/bin/python -u -m rfd3_grpo.train --out rfd3_flow_benchmark/train_1crn --spec-json rfd3_flow_benchmark/1CRN.json --sampler-kind ve_sde --steps 32 --length 40 --epochs 3 --group-size 4 --eval-n 4 --extra-noise 0.1 --active-min 0.5 --active-max 32 --seed 71 --evaluate-stock --deadline 2026-11-14T06:00:00Z
.venv-rfd3/bin/python -u -m rfd3_flow_benchmark.benchmark
```

The benchmark saves every coordinate set. The read-only `--report` mode audits saved coordinates and writes the [results table](RESULTS.md), including partial runs. Existing samples are skipped on resume; no finished sample is overwritten.

## Measured outcome (2026-11-14)

The predeclared evaluation stopped at its 08:00 UTC start-sample deadline with **17/36 samples** saved. The 1CRN block is complete (four seeds × three policies); 1PGA has one trained-versus-zero pair, and 1UBQ has none. The [results table](RESULTS.md) and [paired-effects plot](paired_effects.png) explicitly show the missing data. All saved raw atom coordinates were finite; scores and gap alignments were recomputed from the saved Cα coordinates; every fixed motif stayed within 0.001 Å of its supplied coordinates. The eight numerical/unit tests pass.

The trained adapter changed the fresh-seed 1CRN guarded score by **−0.00000005** on average versus the zero adapter; mean native-gap RMSD changed by **−0.000022 Å**. On the sole 1PGA pair, the changes were +0.00000122 and −0.000025 Å. These are effectively null effects, far smaller than between-seed variation. They show a mathematically working local policy-gradient implementation and a real pretrained-model parameter update, **not successful performance improvement**. The incomplete held-out coverage does not support a generalization claim. In particular, the stock sampler's different scores on 1PGA cannot be attributed to RL because its stochastic process is different.

The diffusion-specific choices were: (1) add controlled stochasticity to an EDM/VE step so that actions have a Gaussian density; (2) use exact *conditional* action ratios and KL at recorded states for the clipped group-relative objective, while avoiding any claim of exact endpoint likelihood; (3) alter only free Cα updates through a 786-parameter, three-noise-band rank-two adapter so motif coordinates and the 168M-parameter denoiser remain fixed; and (4) compare trained and zero adapters under the **same** VE sampler and seeds. The tiny measured change suggests this adapter and three-group local budget were too conservative for a visible geometric effect. That is an interpretation of this run, not a validated hyperparameter conclusion.
