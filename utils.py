import json

import torch


def hysteresis_area(E, I, d):
    # first, move the entire curve over the x-axis
    I = I - I.min()
    E_fwd, I_fwd = E[d > 0], I[d > 0]
    E_rev, I_rev = E[d < 0], I[d < 0]
    idx_fwd = torch.argsort(E_fwd)
    idx_rev = torch.argsort(E_rev)
    A_fwd = torch.trapz(I_fwd[idx_fwd], E_fwd[idx_fwd])
    A_rev = torch.trapz(I_rev[idx_rev], E_rev[idx_rev])
    area = A_fwd - A_rev
    return area

def get_sorted_checkpoints(model_path):
    checkpoints = sorted(
        model_path.glob("*.pth"),
        key=lambda p: float(p.stem.split("_")[-1])
    )
    config_path = model_path / "config.json"
    with open(config_path, "r") as fh:
        config = json.load(fh)
    return checkpoints, config
    

def get_checkpoint(model_path, epoch):
    checkpoints = model_path.glob("*.pth")
    config_path = model_path / "config.json"

    with open(config_path, "r") as fh:
        config = json.load(fh)
    for checkpoint in checkpoints:
        chk_epoch = int(checkpoint.stem.split("_")[2])
        if chk_epoch == epoch:
            return checkpoint, config
    print(f"Epoch {epoch} not found in {model_path}")
    return None, config