# RFdiffusion3 GRPO: measured results

Updated 2026-11-13T07:29:22.843544+00:00. Values are recomputed from saved RFD3 coordinates.

| Run | Sampler / checkpoint | n | Guarded reward | Rg (Å) | Bond MAE (Å) | Geometry pass | Path KL |
|---|---|---:|---:|---:|---:|---:|---:|
| calibration64 | stock | 2 | -1.0285 | 9.6740 | 0.0306 | 100.0% | 0.0000 |
| calibration64 | modified / zero adapter | 2 | -1.0288 | 9.6157 | 0.0336 | 100.0% | 0.0000 |
| late_s0 | stock | 4 | -0.9739 | 9.1600 | 0.0289 | 100.0% | 0.0000 |
| late_s0 | modified / zero adapter | 4 | -0.9737 | 9.1602 | 0.0288 | 100.0% | 0.0000 |
| late_s0 | modified / epoch 6 | 4 | -0.9691 | 9.1422 | 0.0275 | 100.0% | 0.0093 |
| main_s0 | stock | 4 | -0.9739 | 9.1600 | 0.0289 | 100.0% | 0.0000 |
| main_s0 | modified / zero adapter | 4 | -0.9760 | 9.1323 | 0.0314 | 100.0% | 0.0000 |
| main_s0 | modified / epoch 6 | 4 | -0.9773 | 9.1344 | 0.0319 | 100.0% | 0.7434 |
| main_s1 | stock | 4 | -0.9739 | 9.1600 | 0.0289 | 100.0% | 0.0000 |
| main_s1 | modified / zero adapter | 4 | -0.9760 | 9.1323 | 0.0314 | 100.0% | 0.0000 |
| main_s1 | modified / epoch 6 | 4 | -0.9792 | 9.1626 | 0.0315 | 100.0% | 0.8630 |
| main_s2 | stock | 4 | -0.9739 | 9.1600 | 0.0289 | 100.0% | 0.0000 |
| main_s2 | modified / zero adapter | 4 | -0.9760 | 9.1323 | 0.0314 | 100.0% | 0.0000 |
| main_s2 | modified / epoch 6 | 4 | -0.9768 | 9.1621 | 0.0303 | 100.0% | 1.7138 |
| random_s0 | stock | 4 | -0.9733 | 9.1598 | 0.0287 | 100.0% | 0.0000 |
| random_s0 | modified / zero adapter | 4 | -0.9760 | 9.1323 | 0.0314 | 100.0% | 0.0000 |
| random_s0 | modified / epoch 6 | 4 | -0.9760 | 9.1627 | 0.0299 | 100.0% | 1.0563 |
| single_s0 | stock | 4 | -0.9733 | 9.1598 | 0.0287 | 100.0% | 0.0000 |
| single_s0 | modified / zero adapter | 4 | -0.9760 | 9.1323 | 0.0314 | 100.0% | 0.0000 |
| single_s0 | modified / epoch 6 | 4 | -0.9759 | 9.1322 | 0.0314 | 100.0% | 0.1303 |
| smoke_v3 | stock | 2 | -30.6129 | 16.2990 | 14.4640 | 0.0% | 0.0000 |
| smoke_v3 | modified / zero adapter | 2 | -30.6042 | 16.2890 | 14.4602 | 0.0% | 0.0000 |
| smoke_v3 | modified / epoch 1 | 2 | -30.6041 | 16.2889 | 14.4601 | 0.0% | 0.0118 |

Completed reward-directed training seed changes in guarded reward: main_s0 -0.00121, main_s1 -0.00312, main_s2 -0.00076. Mean across 3 completed seeds: -0.00170. These runs share the same four evaluation noise seeds; this is a descriptive comparison, not independent validation across evaluation sets.

The table recomputes the **same guarded reward** (`−Rg/10 − 2×bond MAE − 10×clash fraction`) for every run from saved coordinates. This matters for the random-control run: its training advantages are random and its original evaluation log records compactness only. The common table score is suitable for comparing outcomes, but the control was not trained on it.

Geometry pass is an observed small-sample fraction under a Cα-distance/clash heuristic. It does not establish foldability or all-atom validity. Stock and modified samplers must be compared before attributing changes to optimization. Path KL refers only to the modified stochastic policy and its reference.

## Actual update evidence

### calibration64

State: complete; 168,038,994 frozen network parameters, 786 trainable adapter parameters. 32 residues, 63 denoising calls per trajectory, 2 recycling passes per call.

- last_recycle_head_equivalence_max_error: 1.060963e-05

### late_s0

State: complete; 168,038,994 frozen network parameters, 786 trainable adapter parameters. 32 residues, 63 denoising calls per trajectory, 2 recycling passes per call.

- last_recycle_head_equivalence_max_error: 9.059906e-06
- Completed recorded updates: 6; training siblings: 24.
- Adapter weight-delta norm: 0.002287474; frozen-network tensors receiving gradients: 0.
- Final effective weight-delta norms by low/mid/high noise expert (or one shared expert): [0.0020673598628491163, 0.000979060772806406, 0.0]; active sigma levels per trajectory: 5.
- Final group rewards: [-0.8798794150352478, -0.8168089985847473, -0.8912166357040405, -0.9719631671905518]; relative advantages: [0.18281687796115875, 1.3258365392684937, -0.022646598517894745, -1.4860090017318726].
- Final reuse clip fraction: 0.000%; max absolute log ratio: 0.0198.
- Paired Δguarded reward: +0.00458; evaluation-sample bootstrap 95% interval [-0.00008, +0.01308].
- Paired ΔRg: -0.01805 Å; evaluation-sample bootstrap 95% interval [-0.05400, +0.00002] Å. These intervals do not measure uncertainty across training seeds.

### main_s0

State: complete; 168,038,994 frozen network parameters, 786 trainable adapter parameters. 32 residues, 63 denoising calls per trajectory, 2 recycling passes per call.

- last_recycle_head_equivalence_max_error: 7.152557e-06
- Completed recorded updates: 6; training siblings: 24.
- Adapter weight-delta norm: 0.003467036; frozen-network tensors receiving gradients: 0.
- Final effective weight-delta norms by low/mid/high noise expert (or one shared expert): [0.001864943653345108, 0.002193232998251915, 0.001931851962581277]; active sigma levels per trajectory: 14.
- Final group rewards: [-0.8829967379570007, -0.7977770566940308, -0.8950671553611755, -0.9707019925117493]; relative advantages: [0.05933079123497009, 1.448771595954895, -0.13746799528598785, -1.3706353902816772].
- Final reuse clip fraction: 0.000%; max absolute log ratio: 0.1528.
- Paired Δguarded reward: -0.00121; evaluation-sample bootstrap 95% interval [-0.00347, +0.00001].
- Paired ΔRg: +0.00204 Å; evaluation-sample bootstrap 95% interval [+0.00007, +0.00576] Å. These intervals do not measure uncertainty across training seeds.

### main_s1

State: complete; 168,038,994 frozen network parameters, 786 trainable adapter parameters. 32 residues, 63 denoising calls per trajectory, 2 recycling passes per call.

- last_recycle_head_equivalence_max_error: 1.573563e-05
- Completed recorded updates: 6; training siblings: 24.
- Adapter weight-delta norm: 0.004226335; frozen-network tensors receiving gradients: 0.
- Final effective weight-delta norms by low/mid/high noise expert (or one shared expert): [0.0029919843655079603, 0.0024119012523442507, 0.0017585994210094213]; active sigma levels per trajectory: 14.
- Final group rewards: [-0.8812506198883057, -0.8766710162162781, -1.0660173892974854, -0.9857965707778931]; relative advantages: [0.9036403894424438, 0.9617763757705688, -1.441890835762024, -0.42352357506752014].
- Final reuse clip fraction: 0.000%; max absolute log ratio: 0.1551.
- Paired Δguarded reward: -0.00312; evaluation-sample bootstrap 95% interval [-0.00935, +0.00001].
- Paired ΔRg: +0.03025 Å; evaluation-sample bootstrap 95% interval [+0.00001, +0.09071] Å. These intervals do not measure uncertainty across training seeds.

### main_s2

State: complete; 168,038,994 frozen network parameters, 786 trainable adapter parameters. 32 residues, 63 denoising calls per trajectory, 2 recycling passes per call.

- last_recycle_head_equivalence_max_error: 1.347065e-05
- Completed recorded updates: 6; training siblings: 24.
- Adapter weight-delta norm: 0.003613294; frozen-network tensors receiving gradients: 0.
- Final effective weight-delta norms by low/mid/high noise expert (or one shared expert): [0.0019395501585677266, 0.0014904879499226809, 0.0026594146620482206]; active sigma levels per trajectory: 14.
- Final group rewards: [-0.943604588508606, -1.0378360748291016, -1.0223056077957153, -1.0276641845703125]; relative advantages: [1.7127912044525146, -0.799332857131958, -0.3853050172328949, -0.5281596779823303].
- Final reuse clip fraction: 2.232%; max absolute log ratio: 0.4353.
- Paired Δguarded reward: -0.00076; evaluation-sample bootstrap 95% interval [-0.00219, +0.00008].
- Paired ΔRg: +0.02980 Å; evaluation-sample bootstrap 95% interval [-0.00001, +0.08904] Å. These intervals do not measure uncertainty across training seeds.

### random_s0

State: complete; 168,038,994 frozen network parameters, 786 trainable adapter parameters. 32 residues, 63 denoising calls per trajectory, 2 recycling passes per call.

- last_recycle_head_equivalence_max_error: 9.655952e-06
- Completed recorded updates: 6; training siblings: 24.
- Adapter weight-delta norm: 0.003591979; frozen-network tensors receiving gradients: 0.
- Final effective weight-delta norms by low/mid/high noise expert (or one shared expert): [0.002188938669860363, 0.0015131221152842045, 0.0024127415381371975]; active sigma levels per trajectory: 14.
- Final group rewards: [-0.6411830186843872, -1.0647474527359009, 0.22152774035930634, -1.6096279621124268]; relative advantages: [0.1977241486310959, -0.43518051505088806, 1.4868165254592896, -1.2493598461151123].
- Final reuse clip fraction: 0.000%; max absolute log ratio: 0.1342.
- Paired Δguarded reward: +0.00006; evaluation-sample bootstrap 95% interval [-0.00011, +0.00029].
- Paired ΔRg: +0.03037 Å; evaluation-sample bootstrap 95% interval [+0.00000, +0.09106] Å. These intervals do not measure uncertainty across training seeds.

### single_s0

State: complete; 168,038,994 frozen network parameters, 262 trainable adapter parameters. 32 residues, 63 denoising calls per trajectory, 2 recycling passes per call.

- last_recycle_head_equivalence_max_error: 1.049042e-05
- Completed recorded updates: 6; training siblings: 24.
- Adapter weight-delta norm: 0.001230469; frozen-network tensors receiving gradients: 0.
- Final effective weight-delta norms by low/mid/high noise expert (or one shared expert): [0.001230468973517418]; active sigma levels per trajectory: 14.
- Final group rewards: [-0.8852681517601013, -0.8184537887573242, -0.8947396874427795, -0.9706759452819824]; relative advantages: [0.13000038266181946, 1.3679698705673218, -0.045492902398109436, -1.4524773359298706].
- Final reuse clip fraction: 0.000%; max absolute log ratio: 0.03699.
- Paired Δguarded reward: +0.00010; evaluation-sample bootstrap 95% interval [+0.00003, +0.00017].
- Paired ΔRg: -0.00010 Å; evaluation-sample bootstrap 95% interval [-0.00018, -0.00002] Å. These intervals do not measure uncertainty across training seeds.

### smoke_v2

State: failed; 168,038,994 frozen network parameters, 786 trainable adapter parameters. 16 residues, 7 denoising calls per trajectory, 2 recycling passes per call.


### smoke_v3

State: complete; 168,038,994 frozen network parameters, 786 trainable adapter parameters. 16 residues, 7 denoising calls per trajectory, 2 recycling passes per call.

- last_recycle_head_equivalence_max_error: 6.198883e-06
- stock_sampler_parity_max_error: 3.814697e-05
- Completed recorded updates: 1; training siblings: 2.
- Adapter weight-delta norm: 0.0002321208; frozen-network tensors receiving gradients: 0.
- Final effective weight-delta norms by low/mid/high noise expert (or one shared expert): [0.0, 0.00023212081578094512, 0.0]; active sigma levels per trajectory: 1.
- Final group rewards: [-26.914600372314453, -38.236881256103516]; relative advantages: [1.0, -1.0].
- Final reuse clip fraction: 0.000%; max absolute log ratio: 0.1389.
- Paired Δguarded reward: +0.00013; evaluation-sample bootstrap 95% interval [-0.00043, +0.00069].
- Paired ΔRg: -0.00006 Å; evaluation-sample bootstrap 95% interval [-0.00019, +0.00006] Å. These intervals do not measure uncertainty across training seeds.

Coordinate audit: passed for 23 saved evaluations.

The tiny smoke run tests implementation and an actual pretrained-weight update. It is not a benchmark or evidence of useful protein design. See METHOD.md for the action-density derivation and the exact diffusion-head adaptation.
