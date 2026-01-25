import json
import sys
from pathlib import Path
from argparse import ArgumentParser

import torch
import wandb

from model import CV_PINN, PINN_Loss, train_step, val_step
from dataloading import load_data


def parse_args(args):
    parser = ArgumentParser(prog="cvml.train")
    parser.add_argument("run_prefix", type=str)
    parser.add_argument("--num-iters", "-n", type=int, default=5)

    parsed_args = parser.parse_args(args)
    return parsed_args


def save_model(
    epoch,
    model,
    opt,
    ds_stats,
    filename
):
    to_save = {
        "epoch": epoch,
        "model_dict": model.state_dict(),
        "optimiser_dict": opt.state_dict(),
        "ds_stats": ds_stats
    }
    torch.save(to_save, filename)


def do_training(
    run_name,
    config,
    loss_func,
):
    use_wandb = config["use_wandb"]
    epochs = config["epochs"]

    save_dir  = Path(config["model_dir"]) / run_name
    data_obj = load_data(config)
    train_ds = data_obj["train_ds"]
    val_ds = data_obj["val_ds"]
    ds_handler = data_obj["ds_handler"]
    
    model = CV_PINN(config).cuda()
    optimiser = torch.optim.AdamW(model.parameters(), betas=(0.9, 0.999), lr=config["learning_rate"])

    if use_wandb:
        wandb.init(project=config["project_name"], name=run_name, config=config)

    if not save_dir.exists():
        save_dir.mkdir()
    
    with open(save_dir / "config.json", "w") as fh:
        json.dump(config, fh)

    best_loss = float('inf')
    best_state = None
    for i in range(epochs):
        train_loss = train_step(model, train_ds, loss_func, optimiser)
        val_loss = val_step(model, val_ds, loss_func)
        print(f"Epoch {i} | Train loss: {train_loss.item()} | Val loss: {val_loss.item()}", end="\r" if i < epochs - 1 else "\n", flush=True)
        if use_wandb:
            wandb.log({
                "train_loss": train_loss,
                "val_loss": val_loss
            }, step=i)

        if val_loss < best_loss:
            best_loss = val_loss.item()
            ds_stats = ds_handler.get_ds_stats()
            best_state = {
                "epoch": i,
                "model": model,
                "optimiser": optimiser,
                "ds_stats": ds_stats
            }

        model_dir = Path(config["model_dir"])
        if i % config["save_freq"] == 0:
            ds_stats = ds_handler.get_ds_stats()
            save_model(
                i,
                model,
                optimiser,
                ds_stats,
                model_dir / f"{run_name}/PINNCV_epoch_{i}_loss_{val_loss.item()}.pth"
            )
        if i == (epochs - 1):
            ds_stats = ds_handler.get_ds_stats()
            save_model(
                i,
                model,
                optimiser,
                ds_stats,
                model_dir / f"{run_name}/PINNCV_epoch_{i}_loss_{val_loss.item()}.pth"
            )
            save_model(
                best_state["epoch"],
                best_state["model"],
                best_state["optimiser"],
                best_state["ds_stats"],
                model_dir / f"{run_name}/PINNCV_epoch_{best_state['epoch']}_loss_{best_loss}.pth"
            )


if __name__ == "__main__":
    args = sys.argv[1:]
    parsed_args = parse_args(args)

    with open("config.json", "r") as fh:
        config = json.load(fh)

    prefix = parsed_args.run_prefix
    exp_names = [
        exp["name"].split(".")[0]
        for exp in config["data_files"]
    ]
    loss_func = PINN_Loss(config)
    for exp_name in exp_names:
        for i in range(parsed_args.num_iters):
            exp_config = config.copy()    
            exp_config["leave_out"] = [exp_name]
            exp_run_name = f"{prefix}_leave.out.{exp_name}_run.{i}"
            do_training(exp_run_name, exp_config, loss_func)


