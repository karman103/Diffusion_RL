# Text-to-video diffusion GRPO: model and data selection

Status: historical model-selection protocol. The pinned ZeroScope checkpoint was verified, the stock and backward gates were resolved, and a one-update real-checkpoint residual-adapter GRPO pilot completed. Its audited outcome is in [RESULTS.md](runs/red_ball_residual_grpo_v1/RESULTS.md) and the final interpretation is in [BLOG_POST.md](BLOG_POST.md). The prospective plan below predates the exploratory protocol amendments and should not be read as the method actually run.

## Choice

### Current laptop-feasible candidate (2026-11-15)

- **Generator:** `cerspense/zeroscope_v2_576w` at revision `6963642a64dbefa93663d1ecebb4ceda2d9ecb28`. This is a pretrained text-to-video diffusion model with a 3D conditional U-Net, CLIP text encoder, VAE, and DDIM scheduler. It is smaller than CogVideoX-2B and its required Diffusers components fit the current disk. The source model card states 576×320, 24-frame generation; lower-resolution or shorter clips are only memory probes, not the image-quality benchmark.
- **License:** CC-BY-NC-4.0. This study is local and research-only; a blog may describe results, but the checkpoint's noncommercial terms matter for reuse.
- **Pinned weight hashes (SHA-256):** U-Net `42b51ea486109b7479366e01aae7e267a81bb750c13eb5c6b765a3d5bc7987c9`; text encoder `76877281ed10a4a71f6c2aa0edd286a9e5e23a852a05d13fb05965b464a305bb`; VAE `8b0d11ff25d00ceaa02f602831d9cfe650509fdc850c0a1bcb2acdfa03bd5d56`.
- **Dataset:** MSR-VTT remains a candidate for prompt selection and held-out evaluation. No dataset has yet been downloaded or split for this new model. Predeclare prompt IDs and seeds before any training.
- **Gates:** Run stock generation at native dimensions, inspect the actual clip, then measure one LoRA backward on the real U-Net. Only if both pass should a stochastic-DDIM policy and GRPO training be attempted. A finite low-resolution or random-latent probe alone is insufficient.

### Earlier, disk-blocked choice

- **Generator:** `zai-org/CogVideoX-2b`, the released pretrained text-to-video diffusion transformer. Pin a Hub commit and hashes before any future run. Fine-tune LoRA weights in the video denoiser while freezing the original checkpoint and text encoder. This is an actual learned video generator, unlike the prior fixed-motion renderer.
- **Prompt and reference-video dataset:** MSR-VTT, using the directly hosted `TornadoLabs/msrvtt` mirror. It has 4,923 currently hosted source videos and human-written English captions, with a source-video identifier suitable for disjoint train/evaluation splits. Start with a size-capped, motion-focused subset and record every selected ID, caption, byte count, source terms, and hash. A 200-video miniature mirror (`zhiqiulin/video_caption_datasets/MSR-VTT`) can be used for *data-pipeline* smoke tests, not as evidence of a representative benchmark.
- **Training/evaluation split:** split by source video ID before selecting prompts. GRPO uses the training captions as prompts and rewards generated clips; it does not require the reference clips as supervised targets. Reserve held-out prompts and reference clips for evaluation. Predeclare prompt IDs, seeds, output length/resolution, and reward/evaluator models.

## Method and evidence gates

1. Confirm a pinned checkpoint loads and produces finite, recognizable text-conditioned video. Audit frames, motion, and prompt alignment before RL.
2. Introduce a stochastic reverse sampler with recorded conditional action densities and numerical checks. Do not call its final-video likelihood exact or claim it is identical to the stock deterministic sampler.
3. Run group-of-four rollouts per prompt, group-relative advantages, clipped ratios, and frozen-reference KL. Update only LoRA weights. Compare trained, zero-LoRA on the same sampler, untouched stock sampling, and best-of-four selection at explicit compute cost.
4. Evaluate prompt alignment with a separate video-text model, motion/flicker with independent metrics, diversity, and blinded human review of selected clips. Report per-prompt paired differences and uncertainty; keep reward-model scores separate from independent outcomes.

## Low-precision execution gate (2026-11-14)

The current laptop has 16 GB unified memory per the established project constraints and ~11 GiB free disk as checked on 2026-11-14. MPS is available outside the tool sandbox. The CogVideoX authors report **16 GB NVIDIA VRAM** for fp16 rank-128 LoRA under their own training stack; the Diffusers documentation reports **47 GB** for its separate batch-1 LoRA setup. My earlier categorical statement that CogVideoX LoRA necessarily needs 47 GB was too strong. Neither published measurement validates GRPO training on 16 GB of *shared* Apple memory.

The official Diffusers-format CogVideoX-2B repository totals 13.8 GB: ~3.39 GB video transformer, 9.52 GB T5 text encoder, and 0.862 GB VAE. It cannot be downloaded as a whole with the current free disk. Weight-only int8 could reduce the transformer's resident weight memory, but not the original download size or activation peaks; fp16 is already used in the 16 GB NVIDIA claim. A staged route would encode and cache a fixed prompt set, release the text encoder, then load the quantized video transformer and decode only after sampling. This is a proposal until a pinned implementation demonstrates one finite text-conditioned rollout **and** one adapter backward/update without swap or OOM. Group-of-four GRPO can collect rollouts sequentially, but its backward pass still has to fit. No full-model download should start while the disk gate fails.

An alternative 16 GB inference candidate is the released INT8 `FastVideo/FastMetal-1.3B-QAD` Wan-derived text-to-video model. Its DiT is 1.51 GB, but its full repository is 13.4 GB because the text encoder is large; it is a three-step distilled sampler, so adding stochastic transitions for GRPO would alter the released policy. Published **inference** memory (~3.87 GiB peak on M4 Max) does not prove adapter-training memory, gradient support, or quality after stochasticization. If CogVideoX cannot pass the measured gate, this candidate needs its own sampler/gradient validation before being called a GRPO run.

## Sources

- Model: https://huggingface.co/zai-org/CogVideoX-2b
- Training requirements: https://huggingface.co/docs/diffusers/training/cogvideox
- CogVideoX authors' 16 GB NVIDIA fp16 LoRA claim: https://github.com/zai-org/CogVideo/blob/main/finetune/README.md
- Official checkpoint file sizes: https://huggingface.co/zai-org/CogVideoX-2b/tree/main
- Diffusers quantization caveats: https://huggingface.co/docs/diffusers/quantization/torchao
- FastMetal INT8 candidate and inference-only memory: https://huggingface.co/FastVideo/FastMetal-1.3B-QAD and https://haoailab.com/blogs/fastmetal/
- MSR-VTT mirror and coverage: https://huggingface.co/datasets/TornadoLabs/msrvtt
- 200-video pipeline sample: https://huggingface.co/datasets/zhiqiulin/video_caption_datasets/tree/main/MSR-VTT
