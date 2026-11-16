# Fresh-noise RFdiffusion3 evaluation

Updated 2026-11-13T07:28:58.153724+00:00. Fresh noise seeds 3000000–3000015; no adapter was updated using these samples.

| Policy | n | Guarded reward | Rg (Å) | Bond MAE (Å) | Geometry pass | Path KL | Paired Δreward vs matched zero [95% bootstrap] |
|---|---:|---:|---:|---:|---:|---:|---:|
| stock | 16 | -0.95533 | 8.9351 | 0.0309 | 100.0% | 0.0000 | -0.00047 [-0.00703, +0.00475] (n=16) |
| zero | 16 | -0.95487 | 8.9260 | 0.0311 | 100.0% | 0.0000 | reference |
| main_s0 | 16 | -0.95665 | 8.9282 | 0.0319 | 100.0% | 0.7375 | -0.00179 [-0.00405, -0.00039] (n=16) |
| main_s1 | 16 | -0.95439 | 8.9246 | 0.0310 | 100.0% | 0.8618 | +0.00047 [-0.00243, +0.00459] (n=16) |
| main_s2 | 16 | -0.95545 | 8.9282 | 0.0313 | 100.0% | 1.7080 | -0.00058 [-0.00195, +0.00046] (n=16) |
| random_s0 | 16 | -0.95417 | 8.9256 | 0.0308 | 100.0% | 1.0460 | +0.00070 [-0.00171, +0.00388] (n=16) |
| single_s0 | 16 | -0.95626 | 8.9278 | 0.0317 | 100.0% | 0.1450 | -0.00140 [-0.00422, +0.00148] (n=16) |
| late_zero | 16 | -0.95601 | 8.9352 | 0.0312 | 100.0% | 0.0000 | reference |
| late_s0 | 16 | -0.95570 | 8.9346 | 0.0311 | 100.0% | 0.0090 | +0.00031 [-0.00003, +0.00095] (n=16) |

Three reward-directed adapters, averaged per shared noise seed: Δguarded reward -0.00063, bootstrap 95% interval [-0.00210, +0.00096] over 16 fresh noise seeds.

[Paired difference figure](paired_deltas.png) shows individual noise-seed differences as points, means as horizontal bars and bootstrap intervals as vertical bars when at least four pairs are present.

Late-window adaptation must be compared to late_zero, which uses the same restricted action-noise window. Other adapted policies use the full-window zero reference.

Coordinate audit passed for 144 saved evaluations. All comparisons use the same unconditional 32-residue input and share noise seeds. Bootstrap intervals describe only variation across these noise seeds for fixed trained adapters; they do not measure variation across protein targets, training seeds, or reward definitions. Cα geometry is a heuristic, not foldability or sequence validation.
