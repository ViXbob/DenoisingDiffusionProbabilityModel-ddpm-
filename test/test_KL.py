import os
import sys
from typing import Dict

# Get the directory of the current script
current_directory = os.path.dirname(os.path.abspath(__file__))

# Get the parent directory of the current directory
parent_directory = os.path.dirname(current_directory)

print(parent_directory)

# Add the parent directory to sys.path
sys.path.append(parent_directory)


import torch
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torchvision.utils import save_image

from Diffusion import GaussianDiffusionSampler, GaussianDiffusionTrainer, GaussianDiffusion, create_gaussian_diffusion
from Diffusion.Model import UNet
from Scheduler import GradualWarmupScheduler

from ema import ModelEmaV2
from pathlib import Path

import torch.nn.functional as F


import numpy as np


if __name__ == "__main__":
    device = torch.device("cuda:0")
    
    dataset = CIFAR10(
        root='./CIFAR10', train=False, download=True,
        transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]))
    
    print(dataset.data.shape)
    
    batch_size = 125
    
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=True, pin_memory=True)
    
    # model setup
    net_model = UNet(T=1000, ch=128, ch_mult=[1, 2, 2, 2], attn=[1],
                     num_res_blocks=2, dropout=0.1).to(device)
    
    net_model.load_state_dict(torch.load(os.path.join(
        ".\Checkpoints_official_cifar10", "ema_iterations_ckpt_460k_.pt"), map_location=device))
    
    net_model.eval()

    diffusion = create_gaussian_diffusion()
    
    with torch.no_grad():
        import json
        results_list = {
            "total_bpd": np.zeros((1)),
            "prior_bpd": np.zeros((1)),
            "vb": np.zeros((1000)),
            "xstart_mse": np.zeros((1000)),
            "mse": np.zeros((1000)),
        }
        size = 0
        len = 0
        for images, labels in dataloader:
            x_0 = images.to(device)
            result = diffusion.calc_bpd_loop(model = net_model, x_start=x_0)

            for key, value in result.items():
                results_list[key] = results_list[key] + np.sum(result[key].cpu().numpy(), axis=0)
            
            size += batch_size
            len += 1
            
            if len >= 10:
                break
            
        with open('result.json', "w") as output:
            json.dump({key: (value / size).tolist() for key, value in results_list.items()}, output, indent=4)
    