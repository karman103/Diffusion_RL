# PDB-conditioned RFD3 Flow-GRPO-style benchmark

**Incomplete predeclared benchmark: 17/36 samples saved.** The evaluation stopped at its predeclared 2026-11-14 08:00 UTC start-sample deadline. Missing samples were not rerun; results below are descriptive and unevenly sampled.

Predeclared condition: fixed chain-A residues 1–14 and 23–40; eight residues generated between them. The trained adapter saw only 1CRN. 1PGA and 1UBQ are held-out structures. Higher guarded score and lower gap-to-native Cα RMSD are better. Native gap coordinates were **not** used by the training reward. Gap RMSD is descriptive because the generated gap sequence is unconstrained.

| PDB | Policy | n | Guarded score | Geometry passes | Loop bond MAE (Å) | Gap-to-native RMSD (Å) | Path KL |
|---|---|---:|---:|---:|---:|---:|---:|
| 1CRN | stock | 4 | -0.97913 | 4/4 | 0.0411 | 1.070 | 0.000 |
| 1CRN | ve_zero | 4 | -0.97699 | 4/4 | 0.0387 | 1.156 | 0.000 |
| 1CRN | ve_trained | 4 | -0.97699 | 4/4 | 0.0387 | 1.156 | 0.000 |
| 1PGA | stock | 2 | -1.05524 | 2/2 | 0.0190 | 0.366 | 0.000 |
| 1PGA | ve_zero | 2 | -1.06475 | 2/2 | 0.0378 | 0.634 | 0.000 |
| 1PGA | ve_trained | 1 | -1.06270 | 1/1 | 0.0319 | 0.591 | 0.000 |
| 1UBQ | stock | 0 | — | — | — | — | — |
| 1UBQ | ve_zero | 0 | — | — | — | — | — |
| 1UBQ | ve_trained | 0 | — | — | — | — | — |

## Paired trained-minus-zero differences

Both policies use the same VE-SDE sampler and evaluation seeds. For n≥2, intervals bootstrap the available seeds for a fixed adapter and one PDB; with at most four seeds they are exploratory, not a population claim.

- **1CRN** (n=4): Δreward=-0.00000005 [-0.00000425, +0.00000460]; Δgap_rmsd_to_native=-0.00002228 [-0.00006842, +0.00002385].
- **1PGA** (n=1): Δreward=+0.00000122 (one seed; no interval); Δgap_rmsd_to_native=-0.00002506 (one seed; no interval).
- **1UBQ**: no trained-versus-zero pair was saved.

The stock policy uses RFD3's original churn and is not a density-matched zero-adapter reference. VE-SDE uses an approximate EDM score and a finite Euler schedule; no exact stock-marginal equivalence is claimed. Fixed-motif RMSD was checked below 0.001 Å for every saved sample. These metrics do not establish sequence designability, all-atom physical validity or function.
