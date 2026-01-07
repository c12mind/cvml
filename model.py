import torch
import torch.nn as nn

class CV_PINN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.scanrate = config["scanrate_V"]
        self.hidden_sz = config["hidden_sz"]
        self.dropout = config["dropout"]
        self.inp_sz = config["chunksize"] + 5 # 4 const. exp conditions: mass, voltage, temp, time + scan dir
        self.out_sz = config["chunksize"]
        self.drop = config["dropout"]

        self.mlp = nn.Sequential(
            nn.Linear(self.inp_sz, self.hidden_sz),
            nn.ReLU(),
            nn.Linear(self.hidden_sz, self.hidden_sz),
            nn.ReLU(),
            nn.Linear(self.hidden_sz, self.hidden_sz),
            nn.ReLU(),
            # nn.Dropout(self.drop),
            nn.Linear(self.hidden_sz, self.out_sz),
        )

    
    def forward(self, E, scan_dir, cond_features):
        E.requires_grad_(True)

        faradaic_features = torch.cat([E, scan_dir, cond_features], dim=1)
        I_pred = self.mlp(faradaic_features)
        dI_dE = torch.autograd.grad(
            outputs=I_pred,
            inputs=E,
            grad_outputs=torch.ones_like(I_pred),
            create_graph=True,
        )[0]

        d2I_dE2 = torch.autograd.grad(
            outputs=dI_dE,
            inputs=E,
            grad_outputs=torch.ones_like(dI_dE),
            create_graph=True,
        )[0] # 2nd derviative of I wrt to E - for smoothness loss 

    
        net_outputs = {
            "I_pred": I_pred,
            "d2I_dE2": d2I_dE2,
        }
        return net_outputs

        
def train_step(model, dataloader, loss_func, opt):
    model.train()
    for batch in dataloader:
        I_target = batch["I_target"].cuda()
        E = batch["E"].cuda()
        scan_dir = batch["scan_dir"].cuda()
        cond_features = batch["cond_features"].cuda()

        net_outs = model(E, scan_dir, cond_features)
        loss = loss_func(
            I_target, net_outs["I_pred"],
            net_outs["d2I_dE2"],
        )
        loss.backward()
        opt.step()
        opt.zero_grad()

    return loss

def val_step(model, dataloader, loss_func):
    model.eval()
    for batch in dataloader:
        I_target = batch["I_target"].cuda()
        E = batch["E"].cuda()
        scan_dir = batch["scan_dir"].cuda()
        cond_features = batch["cond_features"].cuda()

        net_outs = model(E, scan_dir, cond_features)
        loss = loss_func(
            I_target, net_outs["I_pred"],
            net_outs["d2I_dE2"],
        )
    return loss


class PINN_Loss:
    def __init__(self, config):
        self.scanrate = config["scanrate_V"]
        self.lambda_smooth = config["lambda_smooth"]
        self.lambda_dir = config["lambda_dir"]
        self.mse_loss = nn.MSELoss()

    def __call__(
        self,
        I_target, I_pred,
        d2I_dE2,
    ):
        data_loss = self.mse_loss(I_pred, I_target)

        smooth_loss = torch.mean(d2I_dE2 ** 2)
        loss = (
            data_loss + 
            self.lambda_smooth * smooth_loss
        )
        return loss
