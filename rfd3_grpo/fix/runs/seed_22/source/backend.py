"""Real RFD3 checkpoint backend and an explicitly modified stochastic sampler.

Stock churn remains an auxiliary, parameter-independent random transition.
At selected noise levels, add independent post-update Gaussian noise on Cα atoms
and adapt only those coordinates. The other atom coordinates remain a shared,
deterministic function of the visited state. This gives a valid conditional action
density in the Cα subspace. It is NOT asserted to preserve stock marginals.
"""
from dataclasses import asdict
import gc
import json
import math
import os
from pathlib import Path
import time

import torch

from .grpo import NoiseLoRA,ve_sde_coefficients

ROOT = Path(__file__).resolve().parents[1]


def verified_checkpoint(path):
    path = Path(path)
    status = path.with_suffix(".download.json")
    if status.exists():
        state = json.loads(status.read_text())
        count = (state["size"]+state["block"]-1)//state["block"]
        if len(set(state["completed"])) != count or path.stat().st_size != state["size"]:
            raise RuntimeError("Checkpoint download is incomplete; do not load the sparse file")
    elif not path.exists():
        raise FileNotFoundError(path)
    return path


class RFD3Backend:
    def __init__(self,checkpoint=None,steps=20,length=32,recycles=2,device="mps",rank=2,noise_experts="binned",adapter_seed=0,specification=None):
        os.environ["FOUNDRY_DEVICE"] = device
        os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO","0.8")
        os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO","0.6")
        os.environ.setdefault("RFD3_LOW_MEMORY_MODE","1")
        os.environ.setdefault("WANDB_MODE","disabled")
        from rfd3.engine import RFD3InferenceEngine,RFD3InferenceConfig
        from rfd3.inference.datasets import ContigJsonDataset

        self.device,self.steps,self.length,self.recycles = device,steps,length,recycles
        checkpoint = verified_checkpoint(checkpoint or ROOT/"rfd3_grpo/checkpoints/rfd3_latest.ckpt")
        config = RFD3InferenceConfig(ckpt_path=checkpoint,inference_mode="legacy",
            diffusion_batch_size=1,inference_num_workers=0,compile_model=False,
            low_memory_mode=True,inference_sampler=dict(num_timesteps=steps,
            n_recycle=recycles,use_classifier_free_guidance=False,allow_realignment=False),seed=0)
        self.engine = RFD3InferenceEngine(**asdict(config))
        self.engine.initialize()
        self.net = self.engine._rfd3_net().eval().requires_grad_(False)
        self.dm = self.net.diffusion_module
        self.sampler = self.net.inference_sampler.sampler
        self.channels = self.dm.to_r_update[-1].in_features
        self.adapter = NoiseLoRA(self.channels,rank=rank,seed=adapter_seed,
            boundaries=() if noise_experts=="single" else (2.,8.)).to(device)
        self.dm.to_r_update[-1].register_forward_pre_hook(self._capture)
        self.specification = specification or {"unconditional":dict(length=str(length))}
        if len(self.specification)!=1:
            raise ValueError("One input condition per backend; group siblings share this condition")
        ds = ContigJsonDataset(data=self.specification,transform=self.engine.pipeline,
                              cif_parser_args=None,name="grpo-condition",subset_to_keys=None,eval_every_n=1)
        self.example = self.engine.trainer.fabric.to_device(ds[0])
        self.f = self.example["feats"]
        self.ca = self.f["is_ca"].bool()
        fixed = self.f["is_motif_atom_with_fixed_coord"].bool()
        # Default (all earlier studies): CA-only actions. RFD3_ACTION_ATOMS=all moves every free atom, so the
        # frozen denoiser does not see CA displaced relative to its own residue and pull it back.
        self.action_mask = (~fixed) if os.environ.get("RFD3_ACTION_ATOMS") == "all" else (self.ca & ~fixed)
        if not self.action_mask.any():
            raise ValueError(f"No free Cα action coordinates (CA={int(self.ca.sum())}, fixed={int(fixed.sum())}, atoms={len(self.ca)})")
        self.fixed = fixed
        self.coordinates = self.example["coord_atom_lvl_to_be_noised"][:1]
        self.schedule_tensor = self.sampler._construct_inference_noise_schedule(device=torch.device(device))
        self.schedule = self.schedule_tensor.tolist()
        with torch.no_grad():
            self.initializer = self.net.token_initializer(self.f)
        self.model_parameters = sum(p.numel() for p in self.net.parameters())
        gc.collect()

    def _capture(self,module,args):
        # Last hook invocation wins: prior recycle uses the unmodified head.
        self.features = args[0].detach()

    @torch.no_grad()
    def denoise(self,x,sigma):
        features = dict(self.f)
        t = torch.full((len(x),),float(sigma),device=self.device)
        init=dict(self.initializer)
        if "chunked_pairwise_embedder" in init:
            chunked=init.pop("chunked_pairwise_embedder")
            out=self.dm(X_noisy_L=x,t=t,f=features,n_recycle=self.recycles,P_LL=None,
                chunked_pairwise_embedder=chunked,initializer_outputs=init,**init)
        else:
            out=self.dm(X_noisy_L=x,t=t,f=features,n_recycle=self.recycles,**init)
        return out["X_L"],self.features,out

    @torch.no_grad()
    def rollout(self,seed,adapted=True,record=True,extra_noise=.1,
                active_min=.5,active_max=32.,stock=False,sampler_kind="stock_churn"):
        if sampler_kind=="ve_sde":
            if stock:
                raise ValueError("stock=True is incompatible with the VE-SDE sampler")
            return self._rollout_ve_sde(seed,adapted,record,extra_noise,active_min,active_max)
        if sampler_kind!="stock_churn":
            raise ValueError(f"Unknown sampler_kind {sampler_kind}")
        rng = torch.Generator().manual_seed(seed)
        action_rng = torch.Generator().manual_seed(seed+9_000_000_000)
        def noise(shape):
            return torch.randn(shape,generator=rng).to(self.device)
        x = self.coordinates+self.schedule_tensor[0]*noise(self.coordinates.shape)*(~self.fixed)[None,:,None]
        records=[]; path_kl=0.; trajectory=[]
        for index,(current,next_sigma) in enumerate(zip(self.schedule_tensor,self.schedule_tensor[1:])):
            gamma = self.sampler.gamma_0 if next_sigma>self.sampler.gamma_min else 0.
            sigma_hat = current*(1+gamma)
            churn = self.sampler.noise_scale*torch.sqrt((sigma_hat.square()-current.square()).clamp_min(0))
            z = x+churn*noise(x.shape)*(~self.fixed)[None,:,None]
            denoised,features,output = self.denoise(z,sigma_hat)
            dt = next_sigma-sigma_hat
            ref_mean = z+self.sampler.step_scale*dt*(z-denoised)/sigma_hat
            active = not stock and active_min<=sigma_hat<=active_max and next_sigma>0
            if active:
                std = extra_noise*min(float(next_sigma),1.)
                if std<=0:
                    raise ValueError("Action variance must be strictly positive")
                # Exact EDM output scaling followed by the Euler step coefficient.
                c_out = self.dm.sigma_data*sigma_hat/math.sqrt(self.dm.sigma_data**2+sigma_hat**2)
                coefficient = -self.sampler.step_scale*dt/sigma_hat*c_out
                h = features[:,self.action_mask,:]
                ref_ca = ref_mean[:,self.action_mask,:]
                shift = coefficient*self.adapter(h,sigma_hat) if adapted else torch.zeros_like(ref_ca)
                old_mean = ref_ca+shift
                action_noise=torch.randn(old_mean.shape,generator=action_rng).to(self.device)
                action = old_mean+std*action_noise
                x = ref_mean.clone(); x[:,self.action_mask,:] = action
                path_kl += float(shift.square().sum()/(2*std*std))
                if record:
                    records.append(dict(features=h,ref_mean=ref_ca,old_mean=old_mean,
                        action=action,std=std,coefficient=float(coefficient),sigma_hat=float(sigma_hat),index=index))
            else:
                x = ref_mean  # shared deterministic transition; adapter disabled
            if not torch.isfinite(x).all():
                raise FloatingPointError(f"Non-finite RFD3 coordinates at step {index}")
            trajectory.append(x[:,self.ca,:].cpu())
        return dict(coordinates=x,ca=x[:,self.ca,:],records=records,path_kl=path_kl,
                    trajectory=torch.stack(trajectory),output=output)

    @torch.no_grad()
    def _rollout_ve_sde(self,seed,adapted,record,noise_level,active_min,active_max):
        """EDM/VE reverse-SDE family, not the stock RFD3 churn sampler.

        For an ideal score s=(D-x)/sigma^2, reverse-sigma drift is
        -(1+a^2)*sigma*s, with variance 2*a^2*sigma*|d_sigma|.
        a=0 recovers the probability-flow Euler step (step_scale=1).
        Finite-step and learned-score marginals are not claimed to match stock.
        """
        if not 0<noise_level<=1:
            raise ValueError("VE-SDE noise_level must be in (0,1]")
        rng=torch.Generator().manual_seed(seed)
        def noise(shape):
            return torch.randn(shape,generator=rng).to(self.device)
        free=(~self.fixed)[None,:,None]
        x=self.coordinates+self.schedule_tensor[0]*noise(self.coordinates.shape)*free
        records=[]; path_kl=0.; trajectory=[]
        for index,(current,next_sigma) in enumerate(zip(self.schedule_tensor,self.schedule_tensor[1:])):
            sigma=float(current); nxt=float(next_sigma)
            denoised,features,output=self.denoise(x,current)
            active=active_min<=sigma<=active_max and nxt>0
            a=noise_level if active else 0.
            drift_scale,dt,std=ve_sde_coefficients(sigma,nxt,a)
            ref_mean=x+drift_scale*dt*(x-denoised)/sigma
            if active:
                c_out=self.dm.sigma_data*current/math.sqrt(self.dm.sigma_data**2+sigma*sigma)
                coefficient=-drift_scale*dt/sigma*c_out
                h=features[:,self.action_mask,:]
                ref_ca=ref_mean[:,self.action_mask,:]
                shift=coefficient*self.adapter(h,current) if adapted else torch.zeros_like(ref_ca)
                old_mean=ref_ca+shift
                action=old_mean+std*noise(old_mean.shape)
                x=ref_mean+std*noise(ref_mean.shape)*free
                x[:,self.action_mask,:]=action
                path_kl+=float(shift.square().sum()/(2*std*std))
                if record:
                    records.append(dict(features=h,ref_mean=ref_ca,old_mean=old_mean,
                        action=action,std=std,coefficient=float(coefficient),sigma_hat=sigma,index=index))
            else:
                x=ref_mean
            x=torch.where(self.fixed[None,:,None],self.coordinates,x)
            if not torch.isfinite(x).all():
                raise FloatingPointError(f"Non-finite VE-SDE coordinates at step {index}")
            trajectory.append(x[:,self.ca,:].cpu())
        return dict(coordinates=x,ca=x[:,self.ca,:],records=records,path_kl=path_kl,
                    trajectory=torch.stack(trajectory),output=output)

    @torch.no_grad()
    def stock_parity_check(self,seed=45):
        """Stock path and the wrapper share all random draws, no adapter/no added noise."""
        from unittest.mock import patch
        wrapped = self.rollout(seed,adapted=False,record=False,stock=True)["coordinates"]
        rng = torch.Generator().manual_seed(seed)
        def normal(*,mean,std,size,device):
            return (mean+std*torch.randn(size,generator=rng)).to(device)
        with patch("torch.normal",normal):
            official = self.sampler.sample_diffusion_like_af3(
                f=dict(self.f),diffusion_module=self.dm,diffusion_batch_size=1,
                coord_atom_lvl_to_be_noised=self.coordinates,
                initializer_outputs=self.initializer,f_ref=None,ref_initializer_outputs=None)["X_L"]
        torch.testing.assert_close(wrapped,official,rtol=1e-4,atol=2e-3)
        return float((wrapped-official).abs().max())

    @torch.no_grad()
    def head_equivalence_check(self,seed=123):
        """Compare cached correction with an actual last-recycle head injection."""
        rng = torch.Generator().manual_seed(seed)
        x = torch.randn(self.coordinates.shape,generator=rng).to(self.device)*4
        sigma=4.
        ref,features,_ = self.denoise(x,sigma)
        # A nonzero diagnostic delta is restored, never used as trained evidence.
        saved = {k:v.clone() for k,v in self.adapter.state_dict().items()}
        try:
            self.adapter.b.fill_(.001)
            coefficient=self.dm.sigma_data*sigma/math.sqrt(self.dm.sigma_data**2+sigma*sigma)
            expected=ref.clone()
            expected[:,self.action_mask,:] += coefficient*self.adapter(features[:,self.action_mask,:],sigma)
            calls=[0]
            def inject(module,args,out):
                calls[0]+=1
                if calls[0]==self.recycles:
                    delta=self.adapter(args[0],sigma)
                    return out+delta*self.action_mask[None,:,None]
                return out
            handle=self.dm.to_r_update[-1].register_forward_hook(inject)
            try:
                actual,_,_=self.denoise(x,sigma)
            finally:
                handle.remove()
            torch.testing.assert_close(actual,expected,rtol=2e-4,atol=2e-4)
            return float((actual-expected).abs().max())
        finally:
            self.adapter.load_state_dict(saved)
