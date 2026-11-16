# POC progress log (RL fine-tuning of diffusion models)

Loop: cron job f86868d8 fires every 10 min. Each tick: read this file, check jobs, do next step.

## Plan
Domain: MNIST 28x28, unconditional DDPM (1.5M-param UNet, cosine schedule), 20-step DDIM eta=1 sampler
viewed as an MDP (Gaussian policy per denoising step).
Rewards: proxy RM = weak MLP trained on 1k labels (log p(7)); gold = strong CNN on 60k labels;
black-box = JPEG (in)compressibility. Feature CNN (independent) for FD / diversity.

Experiments
- E1 main: proxy:7 with ddpo (kl 0 / small / large), ddpo_sf, rwr, draft  -> reward hacking + Goodhart curves
- E2 black-box: jpeg_compress / jpeg_incompress with ddpo
- E3 best-of-N baseline (inference-time), gold vs proxy vs KL bound
- E4 seeds for key configs; gold-reward oracle run
- Figures + blog draft (blog/POST.md) + artifact page

## Status
- [x] env (.venv, torch 2.15 MPS), MNIST downloaded; machine slow (~0.37 s/it base train)
- [x] base DDPM stopped at 10k steps (good samples). best-of-N done: gold acc 0.10 (N=1) -> 0.91 (N=32, KL 2.5) -> 1.0 (N=256); no overoptimization for BoN in range.

- NOTE: user's other job train_ml.py (pid 75032) + Chrome using RAM; 8.4GB swap. Don't kill it. Keep jobs sequential.
- [x] scripts written: rl_finetune.py, best_of_n.py, make_figures.py
- [x] classifiers: gold 99.23, feat 99.03, proxy(MLP,1k labels) 89.89
- [x] RL smoke test OK. Findings: lr 3e-5 => catastrophic proxy hacking (gold acc->0, ink 0.49); lr 1e-5 learns (gold .04->.20 @20ep).
  random-reward control w/ clamp: tiny drift (no ratchet bug). Path-KL dominated by final tiny-sigma step => t_min=20 default.
  KL penalty now in per-step nats (no /784). Defaults: lr 1e-5, epochs 100, clip_x0=1, t_min=20.
- watcher: stops base at 10k steps, then runs scripts/queue.sh over runs/queue.txt (log runs/queue.log). Append lines to queue.txt to add jobs.
- [ ] experiment queue (started 10:23, ~14 runs)
- 10:35 e1_ddpo (proxy log p) => pure reward hacking: blobs, gold 0.09->0.00, KL 148, reward -4.26->-3.30 (BoN gets -0.16 @KL2.5).
  Hypothesis: log-p reward variance dominated by "ink" among non-7s. Queued e1p_ddpo with reward=proxyp:7 (bounded p) right after draft.
  If p-reward works, switch rest of E1 queue (sf, rwr, kl sweep, seeds, oracle) to proxyp:7 and keep log-p runs as the "shaping" ablation.
- 10:45 CONFIRMED: proxyp:7 (bounded p) works: e1p_ddpo gold .086->.49 @ep80, FD_target 20.8->9.4, diversity ~kept.
  e1_draft (log p) catastrophic: KL 7700, FD nan, blobs. => log-p runs = "reward shaping" ablation.
  Queue rebuilt: main runs prefixed p_* (proxyp:7, 150 epochs): ddpo, draft, ddpo_sf, rwr, kl{0.1,1,0.01}, oracle(goldp), jpeg x2, random, seeds s1 s2.
  TODO: update make_figures.py to p_* names; FD nan handling.
- 11:05 HEADLINE: p_ddpo (150ep) gold .086->.961, FD_target 20.8->0.82, KL 22, diversity 5.7->1.85 (mode-seeks within 7s).
  p_draft: proxy p .08->.95 but gold ->0.008, KL 6200: adversarial ghost strokes + grey texture (gradient exploits MLP proxy). Goodhart via gradients.
  Note: runs are deterministic on MPS (e1p_ddpo == p_ddpo first 100 ep).
- 11:16 p_ddpo_sf == p_ddpo bit-for-bit: with inner=1 and per-pixel-mean logprobs, PPO ratio ~1 so clip never active => IS==SF. Killed SF run (partial log kept).
  Added p_ddpo_reuse3 (inner=3, 75 epochs) to test sample reuse where clipping matters. p_draft final: proxy .984, gold .008, KL 6737.
- 11:46 beta=0.1 too strong (KL<1, no learning). Replaced beta=1.0 with 0.001 in queue; sweep = {0,0.001,0.01,0.1}. reuse3: gold .977 @ep45 (3.3x sample-eff).
- 11:56 beta=0.001 caps KL~2.5, gold ~.19@70 (BoN at KL2.5: .91 — but path-KL >= marginal KL caveat). Swapped beta 0.01 -> 0.0003.
- 12:06 appended: p_draft_oracle (DRaFT on gold), p_reuse3_kl (inner3+beta3e-4), p_draft_s1/s2. ETA queue end ~14:10.
- 12:46 p_kl_0.0003: gold .49 KL 6.5 div 4.49 (vs b=0 div 1.85). p_oracle (goldp DDPO): gold .91, KL ~70.
  e2_incompress: jpeg .79->1.03kB, thick textured strokes, shift to 0/3/8. e2_compress: fades to blank black (ink 0, diversity 0) by ep60 — degenerate optimum.
- 13:20 seeds p_ddpo: gold .961/.945/.969 (mean .958±.010); KL 22/165/41 (noisy). e0_random: KL 65, ink .12->.19, fd_all .45->2.85 => PG noise drift is real; KL meter partly noise.
  Fixed BoN proxy_p (mean p, not exp mean log p). Added fig_proxy_vs_gold (DDPO & BoN on diagonal, DRaFT falls off) + fig_sample_eff.
  NEXT: after queue ends (~14:10) -> final figures, blog/POST.md, artifact page.
- 13:26 p_draft_oracle (DRaFT on gold CNN): gold .76 (circular), FD_t 20.8->9.0, KL 3560, MLP-proxy only .31 => mostly real 7s w/ faint ghost overlays; robust RM helps DRaFT but still partly adversarial (classifiers disagree).
- 13:45 blog/POST.md drafted. Measured PPO ratios: |r-1| max 6e-3 after 1 update, <0.01 after 6 passes -> clip(0.2) never fires even w/ reuse; corrected blog. TODO: DRaFT seeds into table, artifact page.
- 13:55 ALL RUNS DONE. DRaFT 3 seeds: proxy .988±.003 gold .007±.002 KL 6500±300. Blog updated. NEXT: artifact page (site/index.html).
- 14:05 DONE: artifact published https://claude.ai/artifact/2c75MqvP1cTmnKCqyo4szE ; blog/POST.md, README.md, figures/. Loop stopped.
