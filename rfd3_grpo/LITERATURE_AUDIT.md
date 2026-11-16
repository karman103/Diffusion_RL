# Published-method audit for the RFD3 GRPO proof of concept

This is a literature-grounded **adaptation**, not a reproduction of a published RFdiffusion3 RL result. The papers establish the policy-gradient design; the RFD3 action space, adapter placement, noise schedule and reward are local choices. Measured outcomes are in [FINAL_REPORT.md](FINAL_REPORT.md).

| Published work | What the implementation adopts | Where it differs |
|---|---|---|
| [DDPO, Black et al., ICLR 2024](https://arxiv.org/html/2305.13301), §4.3 | Denoising as sequential policy actions; exact conditional per-step log ratios; clipped importance-ratio reuse. | DDPO's transition is Gaussian by sampler design. RFD3's stock pre-denoiser churn is pushed through a nonlinear model, so assigning its output a Gaussian density would be wrong. |
| [Flow-GRPO, Liu et al., NeurIPS 2025](https://arxiv.org/html/2505.05470), §4 | Same-input sibling trajectories, group-normalized terminal rewards, stepwise clipped ratios and reference KL. | Flow-GRPO derives a marginal-preserving ODE-to-SDE conversion for its flow model. **No such conversion is claimed for RFD3 here.** The added post-Euler noise changes this sampler. Its larger group size and denoising-reduction findings are not copied as RFD3 results. |
| [DPOK, Fan et al., NeurIPS 2023](https://arxiv.org/html/2305.16381), §4.1 | Online policy optimization with KL to a frozen reference. | The zero adapter of our *modified* sampler is the reference. KL coefficient 0.01 is a POC setting, not an RFD3-validated trust-region target. |
| [EDM, Karras et al., NeurIPS 2022](https://arxiv.org/abs/2206.00364) | Account for denoiser preconditioning separately from the sampler: the LoRA correction passes through RFD3's EDM output scale and Euler coefficient before entering the action mean. | EDM does not prescribe three noise bands, rank-two head LoRA, or the Cα reward. |
| [LoRA, Hu et al., ICLR 2022](https://arxiv.org/abs/2106.09685) | Freeze pretrained weights and learn low-rank matrices on the final atom-coordinate projection. | That paper does not establish that this layer or 786 parameters are sufficient for protein-design RL. |
| [RFD3, Butcher et al., bioRxiv 2025](https://doi.org/10.1101/2025.09.18.676967) and [Rosetta Commons' Foundry release](https://rosettacommons.org/2025/12/22/rfdiffusion3-is-now-available-in-foundry/) | Use its real released checkpoint and denoising module, leaving the base and earlier recycling passes frozen. | The POC tests only unconditional Cα geometry, not RFD3's conditioned design tasks. |

## Exact policy implemented

Given a recorded post-churn state `z_t`, let `m_0(z_t)` be the frozen Euler update on free Cα coordinates. In an active noise window, the action is Gaussian with mean `m_0(z_t) + d_θ(z_t)` and fixed variance `s_t²I`, where

`d_θ = -step_scale (Δt/σ_hat) c_out(σ_hat) B_band A_band h_t`.

At a **fixed recorded state and churn draw**, the old/new log ratio is `[(a-m_old)·(m_θ-m_old) - ||m_θ-m_old||²/2]/s_t²`, summed over all free Cα action coordinates. The zero-adapter reference KL at that state is `||d_θ||²/(2s_t²)`. These are exact **conditional** Gaussian calculations. The clipped, step-averaged objective is a policy surrogate, not an exact whole-trajectory importance weight. Integrating over stock churn generally yields a non-Gaussian marginal transition. Thus trained sampling is compared to its own zero-adapter modified sampler; stock RFD3 is a separate reference.

## Research implications

The policy-gradient construction is defensible, but efficacy is underpowered: four siblings per group, six group updates per run, one unconditional input at each length and a weak geometry reward. Flow-GRPO reports sensitivity to group size and exploration noise; DDPO emphasizes per-step likelihoods and constrained reuse. These motivate **predeclared** group-size, noise, clip and KL sweeps with independent conditions and diversity checks. They do not justify tuning to the current held-out seeds and presenting the result as confirmed.

A stronger RFD3 study would specify a conditioned task, use independent geometry and designability checks, and compare stock sampling, zero-adapter modified sampling, trained sampling and inference-time best-of-group at matched compute. Moving LoRA into deeper attention layers would invalidate cached-head replay and require a new backward path and parity tests. These extensions were not run in the completed POC.
