# Theory section (draft — merged into POST.md once results are in)

## Denoising is a sequential decision problem

A DDPM/DDIM sampler with stochasticity (η = 1) produces an image by a chain

x_T ~ N(0, I) → x_{t_{K-1}} → … → x_{t_1} → x̂_0

where every transition is a Gaussian

π_θ(x_{t'} | x_t) = N( μ_θ(x_t, t),  σ_t² I ).

Read this as an MDP: the **state** is (x_t, t), the **action** is the next, less-noisy image x_{t'},
the **policy** is the denoiser, the **transition** is deterministic given the action, and the only
**reward** arrives at the end: r(x̂_0). Crucially, because each step is Gaussian, we can compute
log π_θ(x_{t'} | x_t) exactly — even though log p_θ(x_0) (the marginal likelihood of the image)
is intractable. That one fact is what makes policy-gradient RL work for diffusion (DDPO, Black et al. 2023; DPOK, Fan et al. 2023).

### Policy gradient (DDPO-SF, REINFORCE)

∇_θ J = E_τ [ A(x̂_0) · Σ_k ∇_θ log π_θ(x_{t_{k+1}} | x_{t_k}) ],    A = (r − mean r) / std r

### Importance-sampled + clipped (DDPO-IS / PPO)

Re-use each batch of trajectories for several gradient steps with the ratio
ρ_k = π_θ / π_θ_old per denoising step and the PPO clipped surrogate
max(−A ρ_k, −A clip(ρ_k, 1−ε, 1+ε)).

### KL regularisation (DPOK-style)

Because both policy and the frozen pretrained model π_ref are Gaussians with the same σ_t,
the per-step KL is closed-form:

KL_k = ‖μ_θ(x_t, t) − μ_ref(x_t, t)‖² / (2σ_t²)

and the KL between the two **path** distributions is the sum over steps. This upper-bounds the KL
between the final image distributions (data-processing inequality), so it's a conservative
"how far did we move" meter — which we use on the x-axis of every Goodhart plot.

### Reward-weighted regression (RWR)

Sample, weight each sample by softmax(r/τ), and do ordinary (weighted) denoising-score-matching on
your own samples. Simple, stable, "supervised-learning-shaped" — but it ignores the denoising MDP
entirely, which makes it sample-inefficient.

### Direct reward backprop (DRaFT-K)

If r is differentiable, skip RL altogether: back-propagate ∇_x r(x̂_0) through the last K sampling
steps into θ. Very low variance, very fast — and, as we'll see, very good at finding the
adversarial directions of a learned reward model.

## A practical gotcha we hit (worth a paragraph in the post)

The log-prob of the *last* stochastic step has σ ≈ 0.006 for a 20-step cosine schedule ending at
t = 0, so its score term (x − μ)/σ² is ~100× larger than early steps. It dominates both the
gradient variance and the path-KL estimate (602 nats vs 7.7 nats when the chain ends at t = 20,
with indistinguishable sample quality: FD 1.35 vs 1.39). We therefore end the stochastic chain at
t_min = 20 and output the model's x̂_0 prediction from there.

A null-reward control (advantages ~ N(0,1)) confirmed that what remains is real optimisation,
not optimiser drift: FD-to-data moved 1.35 → 1.46 in 10 epochs with random rewards vs 1.39 → 6.95
under the proxy reward.
