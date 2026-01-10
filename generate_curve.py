import sys
import torch
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from dataloading import load_data
from model import CV_PINN


def hysteresis_area(E, I, d):
    # first, move the entire curve over the x-axis
    I = I - I.min()
    E_fwd, I_fwd = E[d > 0], I[d > 0]
    E_rev, I_rev = E[d < 0], I[d < 0]
    idx_fwd = np.argsort(E_fwd)
    idx_rev = np.argsort(E_rev)
    A_fwd = np.trapz(I_fwd[idx_fwd], E_fwd[idx_fwd])
    A_rev = np.trapz(I_rev[idx_rev], E_rev[idx_rev])
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
    

def load_pretrained_model(config, checkpoint):
    model = CV_PINN(config).cuda()

    checkpoint_obj = torch.load(checkpoint)

    model.load_state_dict(
        checkpoint_obj["model_dict"]
    )
    
    return model

def make_curve(model, dataset, ds_stats, run_name, epoch):
    I_pred_all = []
    I_real_all = []
    E_real_all = []
    scan_dir_all = []

    print(ds_stats)
    dataset = dataset.cuda()
    I_target = dataset[:, 0].unsqueeze(1)
    # print(I_target, I_target.shape)
    exit()
    E = dataset[:, 1].unsqueeze(1)
    scan_dir = dataset[:, -1].unsqueeze(1)
    cond_features = dataset[:, 2: -1]

    outs = model(E, scan_dir, cond_features)

    I_pred_all.append(outs["I_pred"])
    I_real_all.append(I_target)
    E_real_all.append(E)
    scan_dir_all.append(scan_dir)


    xs = torch.cat(E_real_all).squeeze().detach().cpu().numpy().flatten()
    pred_curve = torch.cat(I_pred_all).squeeze().detach().cpu().numpy().flatten()
    real_curve = torch.cat(I_real_all).squeeze().detach().cpu().numpy().flatten()
    dir_curve = torch.cat(scan_dir_all).squeeze().detach().cpu().numpy().flatten()
    mass = cond_features.squeeze().detach().cpu().numpy()[0, 0]

    E_halfrange = ds_stats["voltage_sweep_stats"]["halfrange"]
    E_mid = ds_stats["voltage_sweep_stats"]["mid"]
    I_mean = ds_stats["current_stats"]["mean"]
    I_std = ds_stats["current_stats"]["std"]
    mass_mean = ds_stats["mass_stats"]["mean"]
    mass_std = ds_stats["mass_stats"]["std"]



    xs = (xs * E_halfrange) + E_mid
    pred_curve = (pred_curve * I_std) + I_mean
    real_curve = (real_curve * I_std) + I_mean
    mass_g = (mass * mass_std) + mass_mean

    scanrate_V = 0.1
    # delta_V = 0.8 # constant for all experiments? ask Alex why
    # use formula: C_sp =∫ I(V)dV / 2νmΔV
    delta_V = xs.max() - xs.min()
    real_area = hysteresis_area(xs, real_curve, dir_curve) / 2
    pred_area = hysteresis_area(xs,pred_curve, dir_curve) / 2
    C_sp_real = (real_area * mass_g) / (scanrate_V * (mass_g/2)**2 * delta_V)
    C_sp_pred = (pred_area * mass_g) / (scanrate_V * (mass_g/2)**2 * delta_V)
    grav_txt = f"real: {C_sp_real:.3f}$Fg^{{-1}}$ | pred: {C_sp_pred:.3f}$Fg^{{-1}}$"
    print(f"Hysteresis area: {real_area}")

    plt.plot(xs, real_curve, label="real I")
    plt.plot(xs, pred_curve, label="pred I")
    plt.legend()
    plt.title(f"{run_name}\n{grav_txt}")
    plt.savefig(f"saved_models/{run_name}/epoch_{epoch}_curve.png")
    

 
if __name__ == "__main__":
    args = sys.argv[1:] 
    if len(args) > 0:
        run_name = args[0]
    else:
        assert False, f"Need run name!"

    run_path = Path("saved_models") / run_name
    all_checkpoints, config = get_sorted_checkpoints(run_path)
    latest_checkpoint = all_checkpoints[0]
    print(f"Loading from checkpoint: {latest_checkpoint}")
    best_epoch = int(latest_checkpoint.stem.split("_")[2])
    _, val_dl, ds_handler = load_data(config)
    model = load_pretrained_model(config, latest_checkpoint)
    model = model.cuda()
    make_curve(model, val_dl, ds_handler.get_ds_stats(), run_name, best_epoch)
