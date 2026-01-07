import json
import sys
from pathlib import Path

import torch
import wandb

from model import CV_PINN, PINN_Loss, train_step, val_step
from dataloading import load_data


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



if __name__ == "__main__":
    args = sys.argv[1:]

    if len(args) > 0:
        run_name = args[0]
    else:
        assert False, "Need run name!"

    with open("config.json", "r") as fh:
        config = json.load(fh)

    use_wandb = config["use_wandb"]
    train_dl, val_dl, ds_handler = load_data(config)
    
    loss_func = PINN_Loss(config)
    model = CV_PINN(config).cuda()
    optimiser = torch.optim.AdamW(model.parameters(), betas=(0.9, 0.999), lr=config["learning_rate"])

    epochs = config["epochs"]
    if use_wandb:
        wandb.init(project=config["project_name"], name=run_name, config=config)

    save_dir  = Path(f"{config['model_dir']}") / run_name

    if not save_dir.exists():
        save_dir.mkdir()
    
    with open(save_dir / "config.json", "w") as fh:
        json.dump(config, fh)

    best_loss = float('inf')
    best_state = None
    for i in range(epochs):
        train_loss = train_step(model, train_dl, loss_func, optimiser)
        val_loss = val_step(model, val_dl, loss_func)
        print(f"Epoch {i} | Train loss: {train_loss.item()} | Val loss: {val_loss.item()}", end="\r" if i < epochs else "\n", flush=True)
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
