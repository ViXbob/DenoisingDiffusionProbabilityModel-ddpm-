
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
        ema.module.load_state_dict(torch.load(os.path.join(
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
    
    ema_evaluation_gap = modelConfig['ema_evaluation_gap']
    
    total_epoch = modelConfig["start_index"]
    
    for e in range(modelConfig["epoch"]):
        total_epoch += 1
        
        if 'epoch_resume' in modelConfig and e < modelConfig['epoch_resume']:
            warmUpScheduler.step()
            continue
        
        non_ema_loss = 0
        len = 0
        
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
                
        if total_epoch % ema_evaluation_gap == 0:
            ema_loss = 0
            non_ema_loss = 0
            with tqdm(dataloader, dynamic_ncols=True) as tqdmDataLoader:
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
        # Load model
        if modelConfig['enable_ema'] == False:
            model.load_state_dict(torch.load(os.path.join(
                modelConfig["save_weight_dir"], modelConfig["test_load_weight"]), map_location=device))
            print(f"Pretrained model weights loaded")
        else:
            model.load_state_dict(torch.load(os.path.join(
                modelConfig["save_weight_dir"], "ema_" + modelConfig["test_load_weight"]), map_location=device))
            print(f"Pretrained ema-model weights loaded")
        
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


import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.utils.data.distributed
import torch.multiprocessing as mp

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size) # 这条命令之后，master进程就处于等待状态

def cleanup():
    dist.destroy_process_group()
     
def trainSingleNodeMultiGPU(local_rank: int, world_size: int, model_config):
    print(f"Running DDP training on rank {local_rank}.")
    
    setup(local_rank, world_size)
    
    device = torch.device(f"cuda:{local_rank}")
    
    # dataset
    dataset = CIFAR10(
        root='./CIFAR10', train=True, download=True,
        transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]))
    
    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset, drop_last=True, shuffle=True)
    
    dataloader = DataLoader(
        dataset, batch_size=model_config["batch_size"], num_workers=4, pin_memory=False, sampler=train_sampler)
    
    net_model = UNet(T=model_config["T"], ch=model_config["channel"], ch_mult=model_config["channel_mult"], attn=model_config["attn"],
                     num_res_blocks=model_config["num_res_blocks"], dropout=model_config["dropout"]).to(device)
    
    if local_rank == 1:
        ema = ModelEmaV2(net_model, decay=0.9999, device=device)
        ema_trainer = GaussianDiffusionTrainer(
            ema.module, model_config["beta_1"], model_config["beta_T"], model_config["T"]).to(device)
        ema_evaluation_gap = model_config['ema_evaluation_gap']
    
    if model_config["training_load_weight"] is not None:
        net_model.load_state_dict(torch.load(os.path.join(
            model_config["save_weight_dir"], model_config["training_load_weight"]), map_location=device))
        ema.module.load_state_dict(torch.load(os.path.join(
            model_config["save_weight_dir"], "ema_" + model_config["training_load_weight"]), map_location=device))
        print(f"Pretrained model weights loaded on {local_rank}")
    
    ddp_model = DDP(net_model, device_ids=[local_rank])
    
    trainer = GaussianDiffusionTrainer(
        ddp_model, model_config["beta_1"], model_config["beta_T"], model_config["T"]).to(device)
    
    optimizer = torch.optim.AdamW(
        ddp_model.parameters(), lr=model_config["lr"], weight_decay=1e-4)
    cosineScheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer, T_max=model_config["epoch"], eta_min=0, last_epoch=-1)
    warmUpScheduler = GradualWarmupScheduler(
        optimizer=optimizer, multiplier=model_config["multiplier"], warm_epoch=model_config["epoch"] // 10, after_scheduler=cosineScheduler)
    
    total_epoch = model_config["start_index"]
    
    for e in range(model_config["epoch"]):
        total_epoch += 1
        non_ema_loss = 0
        
        if 'epoch_resume' in model_config and e < model_config['epoch_resume']:
            warmUpScheduler.step()
            continue
            
        train_sampler.set_epoch(e)
        iterator = tqdm(dataloader, dynamic_ncols=True)
        len = 0
        
        for images, labels in iterator:
            # train
            optimizer.zero_grad()
            x_0 = images.to(device)
            loss = trainer(x_0).sum() / 1000.
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(
                ddp_model.parameters(), model_config["grad_clip"])
            
            optimizer.step()
            
            iterator.set_postfix(ordered_dict={
                "epoch": e,
                "loss: ": loss.item(),
                "img shape: ": x_0.shape,
                "LR": optimizer.state_dict()['param_groups'][0]["lr"]
            })
            
            non_ema_loss += loss.item()
            len += 1
            
            if local_rank == 1:
                ema.update(ddp_model.module)
            
        if local_rank == 1:
            if total_epoch % 25 == 0:
                torch.save(ddp_model.module.state_dict(), os.path.join(
                    model_config["save_weight_dir"], 'iterations_ckpt_' + str(total_epoch // (10000 // len // world_size)) + "0k_.pt"))
                torch.save(ema.module.state_dict(), os.path.join(
                    model_config["save_weight_dir"], 'ema_iterations_ckpt_' + str(total_epoch // (10000 // len // world_size)) + "0k_.pt"))
            else:
                if total_epoch % 5 == 0:
                    torch.save(ddp_model.module.state_dict(), os.path.join(
                        model_config["save_weight_dir"], 'ckpt_' + str(total_epoch) + "_.pt"))
                    torch.save(ema.module.state_dict(), os.path.join(
                        model_config["save_weight_dir"], 'ema_ckpt_' + str(total_epoch) + "_.pt"))
            if total_epoch % ema_evaluation_gap == 0:
                ema_loss = 0
                non_ema_loss = 0
                with tqdm(dataloader, dynamic_ncols=True) as tqdmDataLoader:
                    for images, _ in tqdmDataLoader:
                        x_0 = images.to(device)
                        loss = trainer(x_0).sum() / 1000.
                        non_ema_loss += loss.item()
                        
                        loss = ema_trainer(x_0).sum() / 1000.
                        ema_loss += loss.item()
            else:
                print(f"average loss: {non_ema_loss / len}")
        
        warmUpScheduler.step()
    
    cleanup()

def RunMultiGPUTrain(world_size, model_config):
    mp.spawn(trainSingleNodeMultiGPU,
             args=(world_size, model_config),
             nprocs=world_size,
             join=True)

def sampleSingleNodeMultiGPU(local_rank: int, world_size: int, model_config):
    print(f"Running DDP sampling on rank {local_rank}.")
    
    setup(local_rank, world_size)
    
    # load model and evaluate
    with torch.no_grad():
        device = torch.device(f"cuda:{local_rank}")
        net_model = UNet(T=model_config["T"], ch=model_config["channel"], ch_mult=model_config["channel_mult"], attn=model_config["attn"],
                        num_res_blocks=model_config["num_res_blocks"], dropout=model_config["dropout"]).to(device)
        
        if local_rank == 0:
            if model_config['enable_ema'] == False:
                net_model.load_state_dict(torch.load(os.path.join(
                    model_config["save_weight_dir"], model_config["test_load_weight"]), map_location=device))
                print(f"Pretrained model weights loaded")
            else:
                net_model.load_state_dict(torch.load(os.path.join(
                    model_config["save_weight_dir"], "ema_" + model_config["test_load_weight"]), map_location=device))
                print(f"Pretrained ema-model weights loaded")
        
        ddp_model = DDP(net_model, device_ids=[local_rank])

        ddp_model.eval()
        
        sampler = GaussianDiffusionSampler(
            ddp_model, model_config["beta_1"], model_config["beta_T"], model_config["T"]).to(device)
        
        for i in range(model_config['sample_number_of_batch']):
            # Sampled from standard normal distribution
            noisyImage = torch.randn(
                size=[model_config["batch_size"], 3, 32, 32], device=device)
            if model_config['enable_progress_sampling'] == False:
                sampledImgs = sampler(noisyImage, i)
                sampledImgs = sampledImgs * 0.5 + 0.5  # [0 ~ 1]
                for j in range(model_config['batch_size']):
                    save_image(sampledImgs[j], os.path.join(
                        model_config["sampled_dir"],  str(model_config['sampled_start_index'] + local_rank *  model_config["batch_size"] * model_config['sample_number_of_batch'] + i * model_config["batch_size"] + j).zfill(6)) + ".png")
            else:
                sampler.progressive_sampling_and_save(noisyImage, [str(os.path.join(model_config["sampled_dir"],  str(model_config['sampled_start_index'] + local_rank *  model_config["batch_size"] * model_config['sample_number_of_batch'] + i * model_config["batch_size"] + j).zfill(6))) for j in range(model_config['batch_size'])], i)
    
    cleanup()
    
def RunMultiGPUSample(world_size, model_config):
    mp.spawn(sampleSingleNodeMultiGPU,
             args=(world_size, model_config),
             nprocs=world_size,
             join=True)