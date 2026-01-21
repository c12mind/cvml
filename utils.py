import json

import torch
import numpy as np
from scipy.stats import t
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 15


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

    
def get_student_t(confidence_required, num_samples):
    alpha = 1 - confidence_required
    utp = 1 - (alpha/2)
    dof = num_samples - 1
    return t.ppf(utp, df=dof)

    
def plot_result(result_obj, offset_param=0.2):
    experiments = result_obj["experiments"]
    xs = 2 * np.arange(len(experiments))
    plt.figure(figsize=(15, 5))
    setting_names = result_obj["settings"].keys()
    n = len(setting_names)
    width = offset_param /  n
    for i, setting_name in enumerate(setting_names):
        offset = (i - (n-1)/2) * width
        setting = result_obj["settings"][setting_name]
        ys = np.array(setting["normed_pred"])
        ci = np.array(setting["ci"])
        plt.errorbar(xs + offset, ys, yerr=ci, fmt='o', capsize=4, label=setting_name)
    plt.plot(xs, np.ones_like(xs), color="black", label="Target", alpha=0.5, linestyle="--")
    plt.legend(ncols=5)
    plt.ylabel("Average Normalised $C_{sp}$")
    plt.xlabel("Experiment")
    plt.xticks(xs, experiments)
    plt.savefig("test.png")



if __name__ == "__main__":
    with open("saved_models/results.json", "r") as fh:
        results = json.load(fh)
    
    plot_result(results, offset_param=0.8)