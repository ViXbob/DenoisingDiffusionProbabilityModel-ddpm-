
import os
from typing import Dict

import torch
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torchvision.utils import save_image

from Diffusion import GaussianDiffusionSampler, GaussianDiffusionTrainer
from Diffusion.Model import UNet
from Scheduler import GradualWarmupScheduler

from ema import ExponentialMovingAverage
from pathlib import Path

def train(modelConfig: Dict):
    device = torch.device(modelConfig["device"])
    # dataset
    dataset = CIFAR10(
        root='./CIFAR10', train=True, download=True,
        transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]))
    dataloader = DataLoader(
        dataset, batch_size=modelConfig["batch_size"], shuffle=True, num_workers=4, drop_last=True, pin_memory=True)

    # model setup
    net_model = UNet(T=modelConfig["T"], ch=modelConfig["channel"], ch_mult=modelConfig["channel_mult"], attn=modelConfig["attn"],
                     num_res_blocks=modelConfig["num_res_blocks"], dropout=modelConfig["dropout"]).to(device)
    if modelConfig["training_load_weight"] is not None:
        net_model.load_state_dict(torch.load(os.path.join(
            modelConfig["save_weight_dir"], modelConfig["training_load_weight"]), map_location=device))
    optimizer = torch.optim.AdamW(
        net_model.parameters(), lr=modelConfig["lr"], weight_decay=1e-4)
    cosineScheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer, T_max=modelConfig["epoch"], eta_min=0, last_epoch=-1)
    warmUpScheduler = GradualWarmupScheduler(
        optimizer=optimizer, multiplier=modelConfig["multiplier"], warm_epoch=modelConfig["epoch"] // 10, after_scheduler=cosineScheduler)
    trainer = GaussianDiffusionTrainer(
        net_model, modelConfig["beta_1"], modelConfig["beta_T"], modelConfig["T"]).to(device)

    enable_ema =  modelConfig["enable_ema"]
    if enable_ema:
        ema_update_gap = modelConfig["ema_update_gap"]
    else:
        ema_update_gap = 1
    
    ema = ExponentialMovingAverage(net_model.parameters(), decay=0.9999)
    
    start_index = modelConfig["start_index"]
    
    print(f"enable_ema: {enable_ema}, ema_update_gap: {ema_update_gap}")
    
    # start training
    total_epoch = start_index
    for e in range(modelConfig["epoch"]):
        total_epoch += 1
        before_avg_loss = 0
        len = 0
        
        if 'epoch_resume' in modelConfig and e < modelConfig['epoch_resume']:
            warmUpScheduler.step()
            continue
                    
        with tqdm(dataloader, dynamic_ncols=True) as tqdmDataLoader:
            for images, labels in tqdmDataLoader:
                # train
                optimizer.zero_grad()
                x_0 = images.to(device)
                loss = trainer(x_0).sum() / 1000.
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    net_model.parameters(), modelConfig["grad_clip"])
                optimizer.step()
                tqdmDataLoader.set_postfix(ordered_dict={
                    "epoch": e,
                    "loss: ": loss.item(),
                    "img shape: ": x_0.shape,
                    "LR": optimizer.state_dict()['param_groups'][0]["lr"]
                })
                
                before_avg_loss += loss.item()
                len += 1
                
                if enable_ema == True:
                    if ema.num_updates % ema_update_gap == 0:
                        ema.update()
                        ema.copy_to()
                
        print(f"before: {before_avg_loss / len}")
        
        warmUpScheduler.step()
        
        torch.save(net_model.state_dict(), os.path.join(
            modelConfig["save_weight_dir"], 'ckpt_' + str(start_index + e) + "_.pt"))
        
        if total_epoch % (10000 // len) == 0:
            torch.save(net_model.state_dict(), os.path.join(
                modelConfig["save_weight_dir"], 'iterations_ckpt_' + str(total_epoch // (10000 // len)) + "0k_.pt"))


def eval(modelConfig: Dict):
    # load model and evaluate
    with torch.no_grad():
        device = torch.device(modelConfig["device"])
        model = UNet(T=modelConfig["T"], ch=modelConfig["channel"], ch_mult=modelConfig["channel_mult"], attn=modelConfig["attn"],
                     num_res_blocks=modelConfig["num_res_blocks"], dropout=0.)
        ckpt = torch.load(os.path.join(
            modelConfig["save_weight_dir"], modelConfig["test_load_weight"]), map_location=device)
        model.load_state_dict(ckpt)
        print("model load weight done.")
        model.eval()
        sampler = GaussianDiffusionSampler(
            model, modelConfig["beta_1"], modelConfig["beta_T"], modelConfig["T"]).to(device)
        for i in range(modelConfig['sample_number_of_batch']):
            # Sampled from standard normal distribution
            noisyImage = torch.randn(
                size=[modelConfig["batch_size"], 3, 32, 32], device=device)
            if modelConfig['enable_progress_sampling'] == False:
                sampledImgs = sampler(noisyImage, i)
                sampledImgs = sampledImgs * 0.5 + 0.5  # [0 ~ 1]
                for j in range(modelConfig['batch_size']):
                    save_image(sampledImgs[j], os.path.join(
                        modelConfig["sampled_dir"],  str(i * modelConfig["batch_size"] + j).zfill(6)) + ".png")
            else:
                sampler.progressive_sampling_and_save(noisyImage, [str(os.path.join(modelConfig["sampled_dir"],  str(i * modelConfig["batch_size"] + j).zfill(6))) for j in range(modelConfig['batch_size'])], i)