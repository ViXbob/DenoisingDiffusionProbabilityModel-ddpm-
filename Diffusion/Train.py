
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

from ema import ModelEmaV2
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

    ema = ModelEmaV2(net_model, decay=0.9999, device=device)
    
    if modelConfig["training_load_weight"] is not None:
        net_model.load_state_dict(torch.load(os.path.join(
            modelConfig["save_weight_dir"], modelConfig["training_load_weight"]), map_location=device))
        ema.load_state_dict(torch.load(os.path.join(
            modelConfig["save_weight_dir"], "ema_" + modelConfig["training_load_weight"]), map_location=device))
        print(f"Pretrained model weights loaded")
    
    optimizer = torch.optim.AdamW(
        net_model.parameters(), lr=modelConfig["lr"], weight_decay=1e-4)
    cosineScheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer, T_max=modelConfig["epoch"], eta_min=0, last_epoch=-1)
    warmUpScheduler = GradualWarmupScheduler(
        optimizer=optimizer, multiplier=modelConfig["multiplier"], warm_epoch=modelConfig["epoch"] // 10, after_scheduler=cosineScheduler)
    
    trainer = GaussianDiffusionTrainer(
        net_model, modelConfig["beta_1"], modelConfig["beta_T"], modelConfig["T"]).to(device)
    ema_trainer = GaussianDiffusionTrainer(
        ema.module, modelConfig["beta_1"], modelConfig["beta_T"], modelConfig["T"]).to(device)
    
    total_epoch = modelConfig["start_index"]
    
    for e in range(modelConfig["epoch"]):
        total_epoch += 1
        
        if 'epoch_resume' in modelConfig and e < modelConfig['epoch_resume']:
            warmUpScheduler.step()
            continue
        
        with tqdm(dataloader, dynamic_ncols=True) as tqdmDataLoader:
            non_ema_loss = 0
            len = 0
            
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
                
                non_ema_loss += loss.item()
                len += 1
                ema.update(net_model)
            
            if total_epoch % 5 == 0:
                torch.save(net_model.state_dict(), os.path.join(
                    modelConfig["save_weight_dir"], 'ckpt_' + str(total_epoch) + "_.pt"))
                torch.save(ema.module.state_dict(), os.path.join(
                    modelConfig["save_weight_dir"], 'ema_ckpt_' + str(total_epoch) + "_.pt"))
            if total_epoch % 25 == 0:
                torch.save(net_model.state_dict(), os.path.join(
                    modelConfig["save_weight_dir"], 'iterations_ckpt_' + str(total_epoch // (10000 // len)) + "0k_.pt"))
                torch.save(ema.module.state_dict(), os.path.join(
                    modelConfig["save_weight_dir"], 'ema_iterations_ckpt_' + str(total_epoch // (10000 // len)) + "0k_.pt"))
                
                tqdmDataLoader.set_postfix(ordered_dict={})
                
                ema_loss = 0
                non_ema_loss = 0
                
                for images, _ in tqdmDataLoader:
                    x_0 = images.to(device)
                    loss = trainer(x_0).sum() / 1000.
                    non_ema_loss += loss.item()
                    
                    loss = ema_trainer(x_0).sum() / 1000.
                    ema_loss += loss.item()
                print(f"non-ema average loss: {non_ema_loss / len}, ema average loss: {ema_loss / len}")
            else:
                print(f"average loss: {non_ema_loss / len}")
        
        warmUpScheduler.step()

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
                        modelConfig["sampled_dir"],  str(modelConfig['sampled_start_index'] + i * modelConfig["batch_size"] + j).zfill(6)) + ".png")
            else:
                sampler.progressive_sampling_and_save(noisyImage, [str(os.path.join(modelConfig["sampled_dir"],  str(modelConfig['sampled_start_index'] + i * modelConfig["batch_size"] + j).zfill(6))) for j in range(modelConfig['batch_size'])], i)