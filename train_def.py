from wandb import AlertLevel
from datetime import timedelta
import torch
import os
from loss_functions import au_detection_metric, CCCLoss
from tqdm import tqdm
import wandb
import torch.optim
from torch import nn

from hyperparams import epochs, grad_accumilation
expression_classification_fn = nn.CrossEntropyLoss()


def train(train_loader, model, device, optimizer, epoch):
    step = 0
    loop = tqdm(train_loader, leave=False)
    for index, data in enumerate(loop):
        x = {}
        x['clip'] = data['clip'].to(device)
        result = model(x['clip']).to(device)
        if(torch.isnan(result.cpu()).any()):
            print("nan value for output")
            continue
        expected = data['expressions'].to(device)
        loss_exp = expression_classification_fn(result[:, 0:8], expected)
        au0 = data['au0'].to(device)
        au1 = data['au1'].to(device)
        au2 = data['au2'].to(device)
        au3 = data['au3'].to(device)
        au4 = data['au4'].to(device)
        au5 = data['au5'].to(device)
        au6 = data['au6'].to(device)
        au7 = data['au7'].to(device)
        au8 = data['au8'].to(device)
        au9 = data['au9'].to(device)
        au10 = data['au10'].to(device)
        au11 = data['au11'].to(device)

        loss_exp_0 = au_detection_metric(result[:, 8], au0.float()).to(device)
        loss_exp_1 = au_detection_metric(result[:, 9], au1.float()).to(device)
        loss_exp_2 = au_detection_metric(result[:, 10], au2.float()).to(device)
        loss_exp_3 = au_detection_metric(result[:, 11], au3.float()).to(device)
        loss_exp_4 = au_detection_metric(result[:, 12], au4.float()).to(device)
        loss_exp_5 = au_detection_metric(result[:, 13], au5.float()).to(device)
        loss_exp_6 = au_detection_metric(result[:, 14], au6.float()).to(device)
        loss_exp_7 = au_detection_metric(result[:, 15], au7.float()).to(device)
        loss_exp_8 = au_detection_metric(result[:, 16], au8.float()).to(device)
        loss_exp_9 = au_detection_metric(result[:, 17], au9.float()).to(device)
        loss_exp_10 = au_detection_metric(
            result[:, 18], au10.float()).to(device)
        loss_exp_11 = au_detection_metric(
            result[:, 19], au11.float()).to(device)
        valience = data['valience'].to(device)
        arousal = data['arousal'].to(device)
        valoss = CCCLoss(result[:, 20], valience)
        arloss = CCCLoss(result[:, 21], arousal)
        losses = [loss_exp_0, loss_exp_1, loss_exp_2, loss_exp_3, loss_exp_4, loss_exp_5, loss_exp_6,
                  loss_exp_7, loss_exp_8, loss_exp_9, loss_exp_10, loss_exp_11, loss_exp, valoss, arloss]
        loss = losses[0]
        for l in losses[1:]:
            loss = loss + l
        wandb.log({
            "Before backprop: Total Train Loss": loss.sum().item(),
            "Before backprop: Expression Loss": loss_exp.item(),
            "Before backprop: valience_loss": valoss.sum().item(),
            "Before backprop: arousal_loss": arloss.sum().item(),
            "Before backprop: au_0": loss_exp_0.sum().item(),
            "Before backprop: au_1": loss_exp_1.sum().item(),
            "Before backprop: au_2": loss_exp_2.sum().item(),
            "Before backprop: au_3": loss_exp_3.sum().item(),
            "Before backprop: au_4": loss_exp_4.sum().item(),
            "Before backprop: au_5": loss_exp_5.sum().item(),
            "Before backprop: au_6": loss_exp_6.sum().item(),
            "Before backprop: au_7": loss_exp_7.sum().item(),
            "Before backprop: au_8": loss_exp_7.sum().item(),
            "Before backprop: au_9": loss_exp_7.sum().item(),
            "Before backprop: au_10": loss_exp_7.sum().item(),
            "Before backprop: au_11": loss_exp_7.sum().item(),
        })
        if(torch.isnan(loss).any()):
            continue

        loss = loss/grad_accumilation
        loss.sum().backward()
        if (index+1) % grad_accumilation == 0:
            optimizer.step()
            optimizer.zero_grad()
        loop.set_description(f"Epoch [{epoch+1}/{epochs}]")
        loop.set_postfix(loss=loss.sum().item())
        wandb.log({
            "step": step,
            "epoch": epoch+1,
            "Total Train Loss": loss.sum().item(),
            "Expression Loss": loss_exp.item(),
            "valience_loss": valoss.sum().item(),
            "arousal_loss": arloss.sum().item(),
            "au_0": loss_exp_0.sum().item(),
            "au_1": loss_exp_1.sum().item(),
            "au_2": loss_exp_2.sum().item(),
            "au_3": loss_exp_3.sum().item(),
            "au_4": loss_exp_4.sum().item(),
            "au_5": loss_exp_5.sum().item(),
            "au_6": loss_exp_6.sum().item(),
            "au_7": loss_exp_7.sum().item(),
            "au_8": loss_exp_7.sum().item(),
            "au_9": loss_exp_7.sum().item(),
            "au_10": loss_exp_7.sum().item(),
            "au_11": loss_exp_7.sum().item(),
        })
        step = step + 1


def val(val_loader, model, device, epoch):
    loop = tqdm(val_loader, leave=False)
    i = 0
    for data in loop:
        x = {}
        x['clip'] = data['clip'].to(device)
        result = model(x['clip']).to(device)
        i += 1
        expected = data['expressions'].to(device)
        loss_exp = expression_classification_fn(result[:, 0:8], expected)
        au0 = data['au0'].to(device)
        au1 = data['au1'].to(device)
        au2 = data['au2'].to(device)
        au3 = data['au3'].to(device)
        au4 = data['au4'].to(device)
        au5 = data['au5'].to(device)
        au6 = data['au6'].to(device)
        au7 = data['au7'].to(device)
        au8 = data['au8'].to(device)
        au9 = data['au9'].to(device)
        au10 = data['au10'].to(device)
        au11 = data['au11'].to(device)
        loss_exp_0 = au_detection_metric(result[:, 8], au0.float()).to(device)
        loss_exp_1 = au_detection_metric(result[:, 9], au1.float()).to(device)
        loss_exp_2 = au_detection_metric(result[:, 10], au2.float()).to(device)
        loss_exp_3 = au_detection_metric(result[:, 11], au3.float()).to(device)
        loss_exp_4 = au_detection_metric(result[:, 12], au4.float()).to(device)
        loss_exp_5 = au_detection_metric(result[:, 13], au5.float()).to(device)
        loss_exp_6 = au_detection_metric(result[:, 14], au6.float()).to(device)
        loss_exp_7 = au_detection_metric(result[:, 15], au7.float()).to(device)
        loss_exp_8 = au_detection_metric(result[:, 16], au8.float()).to(device)
        loss_exp_9 = au_detection_metric(result[:, 17], au9.float()).to(device)
        loss_exp_10 = au_detection_metric(
            result[:, 18], au10.float()).to(device)
        loss_exp_11 = au_detection_metric(
            result[:, 19], au11.float()).to(device)
        valience = data['valience'].to(device)
        arousal = data['arousal'].to(device)
        losses = [loss_exp_0, loss_exp_1, loss_exp_2, loss_exp_3, loss_exp_4, loss_exp_5, loss_exp_6,
                  loss_exp_7, loss_exp_8, loss_exp_9, loss_exp_10, loss_exp_11, loss_exp, valience, arousal]
        loss = losses[0]
        for l in losses[1:]:
            loss = loss + l
        loss /= 10
        loop.set_description(f"Epoch [{epoch+1}/{epochs}] validation")
        loop.set_postfix(loss=loss.sum().item(),)
        wandb.log({
            "epoch_val": epoch+1,
            "val_loss_sum": loss.sum().item(),
        })
