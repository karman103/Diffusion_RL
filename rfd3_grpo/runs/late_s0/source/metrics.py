"""Declared Cα geometry proxies; these do not estimate foldability."""
import torch


def metrics(ca):
    if ca.ndim==2:
        ca=ca[None]
    centered=ca-ca.mean(1,keepdim=True)
    rg=centered.square().sum(-1).mean(-1).sqrt()
    bonds=(ca[:,1:]-ca[:,:-1]).norm(dim=-1)
    n=ca.shape[1]
    idx=torch.arange(n,device=ca.device)
    nonlocal_d=torch.cdist(ca,ca)[:,(idx[:,None]-idx[None,:])>2]
    bond_mae=(bonds-3.8).abs().mean(-1)
    bond_ok=((bonds>=3.4)&(bonds<=4.2)).float().mean(-1)
    clash_fraction=(nonlocal_d<3).float().mean(-1)
    return dict(rg=rg,bond_mae=bond_mae,bond_ok=bond_ok,clash_fraction=clash_fraction,
                geometry_pass=((bond_ok>=.95)&(nonlocal_d.min(-1).values>=3)).float())


def reward(ca,kind="guarded"):
    m=metrics(ca)
    r=-m["rg"]/10
    if kind=="guarded":
        r=r-2*m["bond_mae"]-10*m["clash_fraction"]
    elif kind!="compact":
        raise ValueError(kind)
    return r


def save_ca_pdb(path,x):
    with open(path,"w") as f:
        f.write("REMARK   1 RFD3 CA TRACE; SEQUENCE UNSPECIFIED; GEOMETRY PROXY EXPERIMENT\n")
        for i,(a,b,c) in enumerate(x.detach().cpu().tolist(),1):
            f.write(f"ATOM  {i:5d}  CA  UNK A{i:4d}    {a:8.3f}{b:8.3f}{c:8.3f}  1.00  0.00           C\n")
        f.write("TER\nEND\n")
