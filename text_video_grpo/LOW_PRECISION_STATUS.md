# Low-precision text-to-video GRPO feasibility status

Updated 2026-11-15 00:01 America/Toronto. No full video checkpoint has been downloaded or trained in this study.

## Measured locally

- Machine: Apple arm64, macOS 15.6.1, PyTorch 2.8.0; MPS accessible outside the tool sandbox.
- Free workspace disk at check: ~11 GiB. No unrelated compute jobs were stopped.
- `quantized_gradient_probe.py` exercised a frozen torchao INT8 1024×1024 linear layer on MPS with a rank-4 LoRA-style adapter. The initial fp16-adapter AdamW run had a finite forward/backward but produced a NaN parameter update. A corrected fp32 adapter/optimizer run had finite forward, gradient and update, no frozen-base gradient, adapter change norm 0.06398, and 61,652,992 bytes MPS driver allocation for this tiny probe. This tests one kernel path, not a video-transformer block or end-to-end GRPO.

## Source-backed feasibility and correction

- The CogVideoX authors list a 16 GB *NVIDIA* fp16 LoRA configuration, whereas the Diffusers recipe lists 47 GB. The earlier blanket statement that CogVideoX-2B LoRA requires 47 GB was incorrect. Neither number establishes feasibility on an M3 with 16 GB *shared* memory, especially with GRPO rollouts.
- The official CogVideoX-2B Diffusers repository is 13.8 GB on disk, including 9.52 GB text encoder, 3.39 GB denoiser, and 0.862 GB VAE; it exceeds current free disk before artifacts or cache duplication. INT8 quantization after download does not reduce download size. A staged, prompt-embedding cache could lower peak *resident* memory and persistent disk, but requires measured implementation.
- An INT8 Wan-derived alternative, `FastVideo/FastMetal-1.3B-QAD`, claims 16 GB Apple inference support and has a 1.51 GB DiT. Its repository totals 13.4 GB due to the fp16 text encoder. It uses a 3-step distilled sampler; a stochastic GRPO sampler would be a changed policy and requires quality and numerical gates. Its published ~3.87 GiB inference peak is not a training measurement.
- A complete **inference** package under the current disk cap exists: `off-grid-ai/Wan2.1-T2V-1.3B-GGUF` Q4 generator plus the `off-grid-ai/Video-Companions-GGUF` Q4 UMT5 encoder (3.655 GB) and VAE (0.254 GB). The GGUF/stable-diffusion.cpp runtime is an inference path, however; it does not supply the autograd needed for LoRA/GRPO. It should not be counted as a training solution merely because it fits on disk.

## Remaining gate for an honest run

Secure sufficient workspace storage for one pinned checkpoint and intermediate files (or an external volume), then run a staged low-precision *full-model* forward, stochastic-rollout, and adapter-backward test. Require finite recognizable samples and measured peak memory without swap before starting any GRPO benchmark. No claim of video-model GRPO training is supported yet.

Sources for the additional compact inference package: https://huggingface.co/off-grid-ai/Wan2.1-T2V-1.3B-GGUF and https://huggingface.co/off-grid-ai/Video-Companions-GGUF.
