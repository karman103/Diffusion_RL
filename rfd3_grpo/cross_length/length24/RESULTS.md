# Fresh-noise RFdiffusion3 evaluation · 24 residues

Updated 2026-11-13T08:08:56.763403+00:00. Fresh noise seeds 4000000–4000007; no adapter was updated using these samples.

This is an untrained length-transfer check: adapters were fitted at 32 residues and evaluated at 24 residues.

| Policy | n | Guarded reward | Rg (Å) | Bond MAE (Å) | Geometry pass | Path KL | Paired Δreward vs matched zero [95% bootstrap] |
|---|---:|---:|---:|---:|---:|---:|---:|
| stock | 8 | -0.88275 | 8.3718 | 0.0228 | 100.0% | 0.0000 | -0.00158 [-0.00822, +0.00340] (n=8) |
| zero | 8 | -0.88117 | 8.3692 | 0.0221 | 100.0% | 0.0000 | reference |
| late_zero | 8 | -0.88338 | 8.3737 | 0.0230 | 100.0% | 0.0000 | reference |
| main_s0 | 8 | -0.88256 | 8.3701 | 0.0228 | 100.0% | 0.5410 | -0.00139 [-0.00449, +0.00067] (n=8) |
| main_s1 | 8 | -0.88245 | 8.3698 | 0.0227 | 100.0% | 0.6450 | -0.00129 [-0.00445, +0.00077] (n=8) |
| main_s2 | 8 | -0.88144 | 8.3691 | 0.0223 | 100.0% | 1.3113 | -0.00027 [-0.00122, +0.00039] (n=8) |
| random_s0 | 8 | -0.88207 | 8.3686 | 0.0226 | 100.0% | 0.7603 | -0.00090 [-0.00385, +0.00116] (n=8) |
| single_s0 | 8 | -0.88143 | 8.3690 | 0.0223 | 100.0% | 0.1095 | -0.00026 [-0.00137, +0.00070] (n=8) |
| late_s0 | 8 | -0.88339 | 8.3737 | 0.0230 | 100.0% | 0.0064 | -0.00000 [-0.00002, +0.00001] (n=8) |

Three reward-directed adapters, averaged per shared noise seed: Δguarded reward -0.00098, bootstrap 95% interval [-0.00304, +0.00041] over 8 fresh noise seeds.

[Paired difference figure](paired_deltas.png) shows individual noise-seed differences as points, means as horizontal bars and bootstrap intervals as vertical bars when at least four pairs are present.

Late-window adaptation must be compared to late_zero, which uses the same restricted action-noise window. Other adapted policies use the full-window zero reference.

Coordinate audit passed for 72 saved evaluations. All comparisons within this report use the same unconditional 24-residue input and share noise seeds. Bootstrap intervals describe only variation across these noise seeds for fixed trained adapters; they do not measure variation across protein targets, training seeds, or reward definitions. Cα geometry is a heuristic, not foldability or sequence validation.
