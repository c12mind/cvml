import sys
import torch
from argparse import ArgumentParser, BooleanOptionalAction
from pathlib import Path
from pprint import pprint
import csv

import matplotlib.pyplot as plt

from dataloading import load_data
from model import CV_PINN

from utils import hysteresis_area, get_sorted_checkpoints, get_checkpoint


def parse_args(args):
    parser = ArgumentParser(prog="cvml.generate_curve")
    parser.add_argument("prefix")
    parser.add_argument("-p", "--peripheral-curves", action=BooleanOptionalAction)
    parsed_args = parser.parse_args(args)
    return parsed_args


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
    real_area = abs(hysteresis_area(xs, real_curve, dir_curve))
    pred_area = abs(hysteresis_area(xs,pred_curve, dir_curve))
    C_sp_real = (real_area) / (2 * scanrate_V * mass_g * delta_V)
    C_sp_pred = (pred_area) / (2 * scanrate_V * mass_g * delta_V)
    grav_txt = f"real: {C_sp_real:.3f}$Fg^{{-1}}$ | pred: {C_sp_pred:.3f}$Fg^{{-1}}$"
    print(f"Hysteresis area: {real_area}")

    plt.figure()
    plt.plot(xs.numpy(), real_curve.numpy(), label="real I")
    plt.plot(xs.numpy(), pred_curve.numpy(), label="pred I")
    plt.legend()
    plt.title(f"{run_name}\n{grav_txt}")
    plt.savefig(f"saved_models/{run_name}/epoch_{epoch}_curve.png")
    return C_sp_pred, C_sp_real
    

 
if __name__ == "__main__":
    args = sys.argv[1:] 
    parsed_args = parse_args(args)

    all_runs = Path("saved_models").glob(f"{parsed_args.prefix}*")
    all_results = {}
    for run_path in all_runs:
        all_checkpoints, config = get_sorted_checkpoints(run_path)
        save_freq = config["save_freq"]
        best_checkpoint = all_checkpoints[0]
        best_epoch = int(best_checkpoint.stem.split("_")[2])
        prev_epoch = (best_epoch // save_freq) * save_freq
        next_epoch = (1 + (best_epoch // save_freq)) * save_freq

        prev_chkpt, _ = get_checkpoint(run_path, prev_epoch)
        next_chkpt, _ = get_checkpoint(run_path, next_epoch)
        print(f"Loading from checkpoint: {best_checkpoint} and peripherals at epochs {prev_epoch}, {next_epoch}")
        chkpt_tags = run_path.name.split("_")
        curve_name = chkpt_tags[-2].split(".")[-1]
        curve_run_num = int(chkpt_tags[-1].split(".")[-1])
        if not curve_name in all_results.keys():
            all_results[curve_name] = {}
        
        if not curve_run_num in all_results[curve_name].keys():
            all_results[curve_name][curve_run_num] = {}

        if parsed_args.peripheral_curves:
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

        real = None
        pred_csp = []
        for e, c in checkpoints:
            model = load_pretrained_model(config, c)
            model = model.cuda()
            pred, real =  make_curve(model, val_ds, ds_handler.get_ds_stats(), run_path.name, e)
            pred_csp.append(pred.item())
        all_results[curve_name][curve_run_num]["real"] = real.item()
        all_results[curve_name][curve_run_num]["preds"] = pred_csp

    csv_rows = []
    for curve in all_results.keys():
        row = [curve, all_results[curve][0]["real"]]
        for run_num in all_results[curve].keys():
            preds = all_results[curve][run_num]["preds"]
            row += "."
            row += preds
            row += "."
        csv_rows.append(row)

    with open(f"{parsed_args.prefix}.csv", "w") as fh:
        writer = csv.writer(fh, delimiter="\t")
        writer.writerows(csv_rows)

