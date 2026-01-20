import sys
import torch
from pathlib import Path

import matplotlib.pyplot as plt

from dataloading import load_data
from model import CV_PINN

from utils import hysteresis_area, get_sorted_checkpoints, get_checkpoint


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

    dataset = dataset.cuda()
    dataset = dataset[:, torch.arange(dataset.size(1)) != 6]
    E = dataset[:, 0].unsqueeze(1)
    I_target = dataset[:, 1].unsqueeze(1)
    scan_dir = dataset[:, -1].unsqueeze(1)
    cond_features = dataset[:, 2: -1]

    outs = model(E, scan_dir, cond_features)

    I_pred_all.append(outs["I_pred"])
    I_real_all.append(I_target)
    E_real_all.append(E)
    scan_dir_all.append(scan_dir)


    xs = torch.cat(E_real_all).squeeze().detach().cpu().flatten()
    pred_curve = torch.cat(I_pred_all).squeeze().detach().cpu().flatten()
    real_curve = torch.cat(I_real_all).squeeze().detach().cpu().flatten()
    dir_curve = torch.cat(scan_dir_all).squeeze().detach().cpu().flatten()
    mass = cond_features.squeeze().detach().cpu()[0, 0]

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
    real_area = abs(hysteresis_area(xs, real_curve, dir_curve) / 2)
    pred_area = abs(hysteresis_area(xs,pred_curve, dir_curve) / 2)
    C_sp_real = (real_area * mass_g) / (scanrate_V * (mass_g/2)**2 * delta_V)
    C_sp_pred = (pred_area * mass_g) / (scanrate_V * (mass_g/2)**2 * delta_V)
    grav_txt = f"real: {C_sp_real:.3f}$Fg^{{-1}}$ | pred: {C_sp_pred:.3f}$Fg^{{-1}}$"
    print(f"Hysteresis area: {real_area}")

    plt.figure()
    plt.plot(xs.numpy(), real_curve.numpy(), label="real I")
    plt.plot(xs.numpy(), pred_curve.numpy(), label="pred I")
    plt.legend()
    plt.title(f"{run_name}\n{grav_txt}")
    plt.savefig(f"saved_models/{run_name}/epoch_{epoch}_curve.png")
    

 
if __name__ == "__main__":
    args = sys.argv[1:] 
    if len(args) > 0:
        run_name = args[0]
        peripheral_curves = True
    else:
        assert False, f"Need run name!"

    run_path = Path("saved_models") / run_name
    all_checkpoints, config = get_sorted_checkpoints(run_path)
    save_freq = config["save_freq"]
    best_checkpoint = all_checkpoints[0]
    best_epoch = int(best_checkpoint.stem.split("_")[2])
    prev_epoch = (best_epoch // save_freq) * save_freq
    next_epoch = (1 + (best_epoch // save_freq)) * save_freq

    prev_chkpt, _ = get_checkpoint(run_path, prev_epoch)
    next_chkpt, _ = get_checkpoint(run_path, next_epoch)
    print(f"Loading from checkpoint: {best_checkpoint} and peripherals at epochs {prev_epoch}, {next_epoch}")

    if peripheral_curves:
        checkpoints = [
            (e, c) for (e, c) in [
                (prev_epoch, prev_chkpt),
                (best_epoch, best_checkpoint),
                (next_epoch, next_chkpt)
            ] 
            if c is not None
        ]
    else:
        checkpoints = [
            (best_epoch, best_checkpoint),
        ]
    data_obj = load_data(config)
    val_ds = data_obj["val_ds"]
    ds_handler = data_obj["ds_handler"]

    for e, c in checkpoints:
        model = load_pretrained_model(config, c)
        model = model.cuda()
        make_curve(model, val_ds, ds_handler.get_ds_stats(), run_name, e)
