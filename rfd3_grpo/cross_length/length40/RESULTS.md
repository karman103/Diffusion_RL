# Fresh-noise RFdiffusion3 evaluation · 40 residues

Updated 2026-11-13T08:08:56.767690+00:00. Fresh noise seeds 4000000–4000007; no adapter was updated using these samples.

This is an untrained length-transfer check: adapters were fitted at 32 residues and evaluated at 40 residues.

| Policy | n | Guarded reward | Rg (Å) | Bond MAE (Å) | Geometry pass | Path KL | Paired Δreward vs matched zero [95% bootstrap] |
|---|---:|---:|---:|---:|---:|---:|---:|
| stock | 8 | -0.98253 | 9.3264 | 0.0249 | 100.0% | 0.0000 | +0.00033 [-0.00034, +0.00093] (n=8) |
| zero | 8 | -0.98285 | 9.3273 | 0.0251 | 100.0% | 0.0000 | reference |
| late_zero | 8 | -0.98149 | 9.3263 | 0.0244 | 100.0% | 0.0000 | reference |
| main_s0 | 8 | -0.98493 | 9.3295 | 0.0260 | 100.0% | 0.9108 | -0.00207 [-0.00481, -0.00007] (n=8) |
| main_s1 | 8 | -0.98183 | 9.3287 | 0.0245 | 100.0% | 1.0675 | +0.00103 [-0.00037, +0.00333] (n=8) |
| main_s2 | 8 | -0.98284 | 9.3266 | 0.0251 | 100.0% | 2.1295 | +0.00001 [-0.00177, +0.00222] (n=8) |
| random_s0 | 8 | -0.98418 | 9.3298 | 0.0256 | 100.0% | 1.2935 | -0.00133 [-0.00274, -0.00020] (n=8) |
| single_s0 | 8 | -0.98214 | 9.3252 | 0.0248 | 100.0% | 0.1852 | +0.00071 [-0.00008, +0.00211] (n=8) |
| late_s0 | 8 | -0.98110 | 9.3225 | 0.0244 | 100.0% | 0.0115 | +0.00039 [-0.00002, +0.00116] (n=8) |

Three reward-directed adapters, averaged per shared noise seed: Δguarded reward -0.00034, bootstrap 95% interval [-0.00129, +0.00037] over 8 fresh noise seeds.

[Paired difference figure](paired_deltas.png) shows individual noise-seed differences as points, means as horizontal bars and bootstrap intervals as vertical bars when at least four pairs are present.

Late-window adaptation must be compared to late_zero, which uses the same restricted action-noise window. Other adapted policies use the full-window zero reference.

Coordinate audit passed for 72 saved evaluations. All comparisons within this report use the same unconditional 40-residue input and share noise seeds. Bootstrap intervals describe only variation across these noise seeds for fixed trained adapters; they do not measure variation across protein targets, training seeds, or reward definitions. Cα geometry is a heuristic, not foldability or sequence validation.
