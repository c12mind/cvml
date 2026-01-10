import json
from pathlib import Path
import math

import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split

# Each data row must have:
# Electrode mass -> float z score normalisation
# sweep direction -> -1 or +1
# Temperature -> float z score normalisation
# Voltage -> float [-1, 1]
# current ->  float [-1, 1]

class DatasetHandler:
    def __init__(self, config, ds_stats=None):
        super().__init__()

        data_files = config["data_files"]
        rows = config["rows"]
        experiments = []
        
        for file in data_files:
            name = Path(file["name"]).resolve()
            df = pd.read_csv(name)[:rows]
            df.columns = ["time", "E_step", "current"] 
            df["mass_mg"] = file["mass_mg"]
            df["temp"] = file["temp"]
            df["time_h"] = file["time_h"]
            df["voltage"] = file["voltage"]
            df["name"] = name.with_suffix("").stem
            experiments.append(df)
        
        self.dataset = pd.concat(experiments, ignore_index=True).drop(columns=["time"])

        if ds_stats is None:
            self.set_voltage_sweep_stats() 
            self.set_current_stats()
            self.set_mass_stats()
            self.set_temp_stats()
            self.set_electrolysis_voltage_stats()
            self.set_time_stats()
        else:
            self.voltage_sweep_stats = ds_stats["voltage_sweep_stats"]
            self.current_stats = ds_stats["current_stats"]
            self.mass_stats = ds_stats["mass_stats"]
            self.temp_stats = ds_stats["temp_stats"]
            self.electrolysis_voltage_stats = ds_stats["electrolysis_voltage_stats"]
            self.time_stats = ds_stats["time_stats"]

        self.add_scan_direction()
        self.normalise_dataset()
        print(self.dataset)

    def add_scan_direction(self):
        self.dataset["scan_dir"] = (self.dataset["E_step"].shift(-1) - self.dataset["E_step"]).fillna(-1)
        self.dataset["scan_dir"] = self.dataset["scan_dir"].apply(lambda x: (x > 0) - (x < 0))
            
    def norm_voltage_sweep(self):
        voltage_mid = self.voltage_sweep_stats["mid"]
        voltage_halfrange = self.voltage_sweep_stats["halfrange"]
        self.dataset["E_step"] = (self.dataset["E_step"] - voltage_mid) / voltage_halfrange

    def norm_continuous_data(self, header):
        mean = self.dataset[header].mean()
        std = self.dataset[header].std()
        self.dataset[header] = (self.dataset[header] - mean) / std
    
    def set_voltage_sweep_stats(self):
        v_min = self.dataset["E_step"].min()
        v_max = self.dataset["E_step"].max()
        voltage_mid = (v_min + v_max) / 2
        voltage_halfrange = (v_max - v_min) / 2
        self.voltage_sweep_stats = {
            "mid": voltage_mid,
            "halfrange": voltage_halfrange
        }

    def set_current_stats(self):
        mean = self.dataset["current"].mean()
        std = self.dataset["current"].std()
        self.current_stats = {
            "mean": mean,
            "std": std
        }

    def set_mass_stats(self):
        mean = self.dataset["mass_mg"].mean()
        std = self.dataset["mass_mg"].std()
        self.mass_stats = {
            "mean": mean,
            "std": std
        }
        
    def set_temp_stats(self):
        mean = self.dataset["temp"].mean()
        std = self.dataset["temp"].std()
        self.temp_stats = {
            "mean": mean,
            "std": std
        }

    def set_electrolysis_voltage_stats(self):
        mean = self.dataset["voltage"].mean()
        std = self.dataset["voltage"].std()
        self.electrolysis_voltage_stats = {
            "mean": mean,
            "std": std
        }

    def set_time_stats(self):
        mean = self.dataset["time_h"].mean()
        std = self.dataset["time_h"].std()
        self.time_stats = {
            "mean": mean,
            "std": std
        }

    def get_ds_stats(self):
        out = {
            "voltage_sweep_stats": self.voltage_sweep_stats,
            "current_stats": self.current_stats,
            "mass_stats": self.mass_stats,
            "temp_stats": self.temp_stats,
            "electrolysis_voltage_stats": self.electrolysis_voltage_stats,
            "time_stats": self.time_stats
        }
        return out

    def normalise_dataset(self):
        self.norm_voltage_sweep()
        self.norm_continuous_data("current")
        self.norm_continuous_data("mass_mg")
        self.norm_continuous_data("time_h")
        self.norm_continuous_data("temp")
        self.norm_continuous_data("voltage")


class CVDataset(Dataset):
    def __init__(self, dataframe, chunksize=8):
        super().__init__()
        self.dataset = dataframe
        self.chunksize = chunksize

    def __len__(self):
        return len(self.dataset) // self.chunksize

    def __getitem__(self, index):
        chunk_start = index * self.chunksize
        chunk_end = chunk_start + self.chunksize

        chunk = self.dataset.loc[
                self.dataset.index[chunk_start:chunk_end],
                ["current", "E_step", "mass_mg", "temp", "time_h", "voltage", "scan_dir"]
            ]
        # chunk = self.dataset.iloc[index][["current", "E_step", "mass_mg", "temp", "time_h", "voltage", "scan_dir"]]
        return chunk

        
def collate_fn(batch):
    target_batch = []
    E_batch = []
    scan_d_batch = []
    conditions_batch = []
    for row in batch:
        target_batch.append(torch.FloatTensor(list(row["current"])).unsqueeze(0))
        E_batch.append(torch.FloatTensor(list(row["E_step"])).unsqueeze(0))
        scan_d_batch.append(torch.FloatTensor(list(row["scan_dir"])).unsqueeze(0))
        features = torch.FloatTensor(row[["mass_mg", "temp", "time_h", "voltage"]].astype(float).to_numpy()).squeeze().unsqueeze(0)
        conditions_batch.append(features)

    target_batch = torch.cat(target_batch)
    E_batch = torch.cat(E_batch)
    scan_d_batch = torch.cat(scan_d_batch)
    conditions_batch = torch.cat(conditions_batch)
    if len(conditions_batch.shape) == 3:
        conditions_batch = conditions_batch[:, 0, :]
    return {
        "E": E_batch,
        "scan_dir": scan_d_batch,
        "cond_features": conditions_batch,
        "I_target": target_batch
    }


def _load_data(config):

    batch_sz = config["batch_sz"]
    leave_out = config["leave_out"]
    filenames = [
        k["name"].split(".")[0] for k in config["data_files"]
    ]
    for name in leave_out:
        if name not in filenames:
            assert False, f"Cannot leave out experiment {name}: does not exist!"
    
    ds_handler = DatasetHandler(config)
    val_subset = ds_handler.dataset["name"].isin(leave_out)
    train_subset = ~val_subset
    train_df = ds_handler.dataset[train_subset]
    val_df = ds_handler.dataset[val_subset]
    train_ds = CVDataset(train_df, config["chunksize"]) 
    val_ds = CVDataset(val_df, config["chunksize"])

    train_dl = DataLoader(
        train_ds,
        batch_size=batch_sz,
        collate_fn=collate_fn,
        num_workers=32,
        pin_memory=True,
        persistent_workers=True,
        shuffle=True
    )

    val_dl = DataLoader(
        val_ds,
        batch_size=batch_sz,
        collate_fn=collate_fn,
        num_workers=32,
        pin_memory=True,
        persistent_workers=True,
        shuffle=False
    )

    for batch in val_dl:
        print(batch["I_target"], batch["I_target"].shape)

    return train_dl, val_dl, ds_handler

    

# TODO: fix the loading, fast loading seems to break the loaded current values
def load_data(config):
    leave_out = config["leave_out"]
    filenames = [
        k["name"].split(".")[0] for k in config["data_files"]
    ]
    for name in leave_out:
        if name not in filenames:
            assert False, f"Cannot leave out experiment {name}: does not exist!"
    
    ds_handler = DatasetHandler(config)
    val_subset = ds_handler.dataset["name"].isin(leave_out)
    train_subset = ~val_subset
    train_df = ds_handler.dataset[train_subset].drop(columns=["name"])
    val_df = ds_handler.dataset[val_subset].drop(columns=["name"])
    train_ds = torch.tensor(train_df.values, dtype=torch.double)
    val_ds = torch.tensor(val_df.values, dtype=torch.double)
    return train_ds, val_ds, ds_handler
