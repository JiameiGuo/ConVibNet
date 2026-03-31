import os
import random
import time

import cv2
import numpy as np
import torch
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
import torch.nn.functional as F

from dataset_4loss import SeqDataset
from model.network_seq import SeqNet
# from utils import reverse_all_hough_space, reverse_max_hough_space, vis_result


def setup_seed(seed):
    # random package
    random.seed(seed)

    # torch package
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # numpy package
    np.random.seed(seed)

    # os
    os.environ["PYTHONHASHSEED"] = str(seed)


def modified_focal_loss(pred, gt):
    """Modified focal loss.
    Runs faster and costs a little bit more memory
    Arguments:
    preds (B x c x h x w)
    gt_regr (B x c x h x w)
    """

    gt = gt.unsqueeze(1)  # (B, H, W) -> (B, 1, H, W)

    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()

    neg_weights = torch.pow(1 - gt, 4)

    pos_loss = torch.log(torch.clamp(pred, min=1e-4)) * torch.pow(1 - pred, 2) * pos_inds
    neg_loss = torch.log(torch.clamp(1 - pred, min=1e-4)) * torch.pow(pred, 2) * neg_weights * neg_inds

    # num_pos = pos_inds.float().sum()

    # loss = -(pos_loss + neg_loss).sum() / (num_pos + 1e-4)
    loss = -(pos_loss + neg_loss).mean()
    return loss

# class LBSign(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, input):
#         return torch.sign(input)

#     @staticmethod
#     def backward(ctx, grad_output):
#         return grad_output.clamp_(-1, 1)

def intersection_loss(pred1, pred2, mask1, mask2):
    mask1 = mask1.unsqueeze(1)  # (B, H, W) -> (B, 1, H, W)
    mask2 = mask2.unsqueeze(1)

    intersection_pred = pred1 * pred2
    intersection_mask = mask1.float() * mask2.float()
    loss = F.binary_cross_entropy(intersection_pred.clamp(min=1e-7), intersection_mask)
    return loss

def diff_loss(pred1, pred2, mask1, mask2):
    mask1 = mask1.unsqueeze(1).float()  # (B, H, W) -> (B, 1, H, W)
    mask2 = mask2.unsqueeze(1).float()

    diff_pred = torch.abs(pred1 - pred2) 
    diff_mask = (mask1 - mask2).abs() 
    loss = F.binary_cross_entropy(diff_pred, diff_mask)
    return loss

def get_model_dataset_expname(config):
    win = config.get("model").get("win")
    stride = config.get("model").get("stride")
    enc_init = config.get("model").get("enc_init")
    fic_init = config.get("model").get("fic_init")

    model = SeqNet(
        num_angle=config["model"]["num_angle"],
        num_rho=config["model"]["num_rho"],
        seq_len=config["model"]["seq_length"],
        win=win if win is not None else 10,
        stride=stride if stride is not None else 5,
        enc_init=enc_init if enc_init is not None else True,
        fic_init=fic_init if fic_init is not None else True,
    )

    if config.get("expriment_name") is None:
        expriment_name = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    else:
        expriment_name = config["expriment_name"] + time.strftime(
            "_(%Y-%m-%d-%H-%M-%S)", time.localtime()
        )

    dataset_train = SeqDataset(
        data_path=config["data"]["data_path"],
        split="train_30_15",
        # split="train_without45",
        size=config["data"]["size"],
        mask_size=config["data"]["mask_size"],
        seq_length=config["model"]["seq_length"],
        num_angle=config["model"]["num_angle"],
        num_rho=config["model"]["num_rho"],
        augment=True,
    )

    dataset_val = SeqDataset(
        data_path=config["data"]["data_path"],
        split="val_30_15",
        # split="val_without45",
        size=config["data"]["size"],
        mask_size=config["data"]["mask_size"],
        seq_length=config["model"]["seq_length"],
        num_angle=config["model"]["num_angle"],
        num_rho=config["model"]["num_rho"],
        augment=False,
    )

    return model, dataset_train, dataset_val, expriment_name


def train(config):
    model, dataset_train, dataset_val, expriment_name = get_model_dataset_expname(
        config
    )

    device = config["train"]["device"]

    base_dir = os.path.abspath("path_to_base")
    log_path = os.path.join(base_dir, expriment_name)
    os.makedirs(log_path, exist_ok=True)
    writer = SummaryWriter(log_path)
    figure_path = os.path.join(log_path, "figures_val")
    os.makedirs(figure_path, exist_ok=True)

    val_every_n = config["train"]["val_every_n"]
    print_every_n = config["train"]["print_every_n"]
    early_stop_thres = config["train"]["early_stop"]

    # save config
    config_path = os.path.join(log_path, "config.txt")
    with open(config_path, "w") as f:
        f.write(str(config))

    best_val_loss = 100
    early_stop_cnt = 0
    optimizer = torch.optim.Adam(model.parameters(), lr=config["train"]["lr"])
    if config["model"]["FocalLoss"]:
        loss_fn1 = modified_focal_loss
        loss_fn3 = intersection_loss
        loss_fn4 = diff_loss
    else:
        loss_fn1 = torch.nn.BCELoss()
        loss_fn3 = intersection_loss
        loss_fn4 = diff_loss

    train_loader = DataLoader(
        dataset_train,
        batch_size=config["train"]["batch_size_train"],
        shuffle=True,
    )
    val_loader = DataLoader(
        dataset_val,
        batch_size=config["train"]["batch_size_val"],
        shuffle=False,
    )

    
    model.to(device)
    model.train()
    for epoch in range(config["train"]["epoch"]):
        epoch_loss12 = 0.0
        epoch_loss3 = 0.0
        epoch_loss4 = 0.0
        epoch_total_loss = 0.0
        for i, (img_seq_1, img_seq_2, labels) in enumerate(train_loader):
            # img = img.to(device)
            # label = label.to(device)
            img_seq_1 = img_seq_1.to(device)
            img_seq_2 = img_seq_2.to(device)
            labels = labels.to(device)
            mask1, mask2 = labels[:, 0, :, :], labels[:, 1, :, :]
            # print(f"img_seq_1 shape: {img_seq_1.shape}, img_seq_2 shape: {img_seq_2.shape}")
            # print(f"mask1 shape: {mask1.shape}, mask2 shape: {mask2.shape}")

            optimizer.zero_grad()

            pred1 = model(img_seq_1)
            pred2 = model(img_seq_2)
            # print("Debugging Info:")
            # print(f"Pred1 Min: {pred1.min().item()}, Max: {pred1.max().item()}")
            # print(f"Pred2 Min: {pred2.min().item()}, Max: {pred2.max().item()}")
            # print(f"Mask1 Min: {mask1.min().item()}, Max: {mask1.max().item()}")
            # print(f"Mask2 Min: {mask2.min().item()}, Max: {mask2.max().item()}")
            loss1 = loss_fn1(pred1, mask1)
            loss2 = loss_fn1(pred2, mask2)
            loss3 = 0.50 * loss_fn3(pred1, pred2, mask1, mask2)
            if epoch >= 1:
                loss4 = 0.02 * loss_fn4(pred1, pred2, mask1, mask2)
            else:
                loss4 = torch.tensor(0.0, device=device)
            
            loss12 = loss1 + loss2

            total_loss = loss12 + loss3 + loss4
            total_loss.backward()

            optimizer.step()

            epoch_loss12 += loss12.item()
            epoch_loss3 += loss3.item()
            epoch_loss4 += loss4.item()
            epoch_total_loss += total_loss.item()

            if (i + 1) % print_every_n == 0 or i == len(train_loader) - 1:
                avg_loss12 = epoch_loss12 / (i + 1)
                avg_loss3 = epoch_loss3 / (i + 1)
                avg_loss4 = epoch_loss4 / (i + 1)
                avg_total_loss = epoch_total_loss / (i + 1)
                print(f"Epoch {epoch+1} | Iter {i+1} | Loss12: {avg_loss12:.6f} | Loss3: {avg_loss3:.6f} | "
                      f"Loss4: {avg_loss4:.6f} | Total Loss: {avg_total_loss:.6f}", flush=True)
                writer.add_scalar("loss/train_total", avg_total_loss, epoch * len(train_loader) + i)
                writer.add_scalar("loss/train_loss12", avg_loss12, epoch * len(train_loader) + i)
                writer.add_scalar("loss/train_loss3", avg_loss3, epoch * len(train_loader) + i)
                writer.add_scalar("loss/train_loss4", avg_loss4, epoch * len(train_loader) + i)

            # validation
            if (epoch >= 2) and (
                ((i + 1) % val_every_n == 0) or i == len(train_loader) - 1
            ):
                val_loss = validate(model, config, val_loader, loss_fn1, loss_fn3, loss_fn4, epoch, i, figure_path)

                print("======================================================", flush=True)
                print(
                    f"Epoch {epoch+1} | Iter {i+1} | Val Loss {val_loss:.6f})", flush=True
                )
                print("======================================================", flush=True)
                writer.add_scalar("loss/val", val_loss, epoch * len(train_loader) + i)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(model.state_dict(), f"{log_path}/model.pth")
                    early_stop_cnt = 0
                else:
                    print("No improvement!!", flush=True)
                    early_stop_cnt += 1
                    if early_stop_cnt >= early_stop_thres:
                        print("Early stop!!!", flush=True)
                        return

                model.train()
        torch.save(model.state_dict(), f"{log_path}/model_epoch.pth")

def validate(model, config, val_loader, loss_fn1, loss_fn3, loss_fn4, epoch, i, figure_path):
    device = config["train"]["device"]
    with torch.no_grad():
        model.eval()
        val_loss = 0.0
        # val_loss_shaft = 0
        # val_loss_tip = 0
        # k = 0
        for j, (img_seq_1, img_seq_2, labels) in enumerate(val_loader):
            # img = img.to(device)
            # label = label.to(device)
            # pred = model(img)
            img_seq_1 = img_seq_1.to(device)
            img_seq_2 = img_seq_2.to(device)
            labels = labels.to(device)
            mask1, mask2 = labels[:, 0, :, :], labels[:, 1, :, :]

            pred1 = model(img_seq_1)
            pred2 = model(img_seq_2)
        
            loss1 = loss_fn1(pred1, mask1)
            loss2 = loss_fn1(pred2, mask2)
            loss3 = 0.50 * loss_fn3(pred1, pred2, mask1, mask2)
            loss4 = 0.02 * loss_fn4(pred1, pred2, mask1, mask2)
            loss12 = loss1 + loss2

            total_loss = loss12 + loss3 + loss4

            val_loss += total_loss.item()
            
        val_loss /= len(val_loader)
           
    return val_loss


if __name__ == "__main__":
    from config import config_list

    setup_seed(42)
    for config in config_list:
        train(config)
