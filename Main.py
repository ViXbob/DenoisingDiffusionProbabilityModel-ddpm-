from Diffusion.Train import train, eval
import argparse
import json
from pathlib import Path

def main(model_config = None):
    modelConfig = {
        "state": "train", # or eval
        "epoch": 50,
        "batch_size": 128,
        "T": 1000,
        "channel": 128,
        "channel_mult": [1, 2, 3, 4],
        "attn": [2],
        "num_res_blocks": 2,
        "dropout": 0.15,
        "lr": 5e-5,
        "multiplier": 2.,
        "beta_1": 1e-4,
        "beta_T": 0.02,
        "img_size": 32,
        "grad_clip": 1.,
        "device": "cuda:0", ### MAKE SURE YOU HAVE A GPU !!!
        "training_load_weight": "saved_ckpt_349_.pt",
        "save_weight_dir": "./Checkpoints/",
        "test_load_weight": "ckpt_199_.pt",
        "sampled_dir": "./SampledImgs/",
        "enable_ema": True,
        "ema_update_gap": 30,
        "start_index": 350
    }
    if model_config is not None:
        modelConfig = model_config
    Path(model_config['save_weight_dir']).mkdir(parents=True, exist_ok=True)
    Path(model_config['sampled_dir']).mkdir(parents=True, exist_ok=True)
    if modelConfig["state"] == "train":
        train(modelConfig)
    else:
        eval(modelConfig)

if __name__ == '__main__':
    # Arguments parser
    parser = argparse.ArgumentParser(description = "Train or eval using DDPM")
    parser.add_argument('--input_setting', type = str, help = 'The json file to the input setting', required=True)
    parser.add_argument('--state', type = str, help = 'train or eval', required=True, choices=["train", "eval"])
    parser.add_argument('--device', type = str, help = 'The device to use', required=False)
    parser.add_argument('--save_weight_dir', type = str, help = 'The directory to save checkpoints', required=True)
    
    # Training arguments
    parser.add_argument('--start_index', type = int, help = 'The start index you store your checkpoints', required=False)
    parser.add_argument('--enable_load_training_weight', action = 'store_true', help = 'Whether enable pre-trained training weights', required=False)
    parser.add_argument('--training_load_weight', type = str, help = 'The name of pretrained weights file', required=False)
    parser.add_argument('--enable_ema', action = 'store_true', help = 'Whether enable EMA', required=False)
    parser.add_argument('--ema_update_gap', type = int, help = 'EMA update gap', required=False)
    parser.add_argument('--epoch', type = int, help = 'Total epoch to train', required=False)
    # End
    
    # Eval arguments
    parser.add_argument('--sample_number_of_batch', type = int, help = 'The number of batch when sampling', required=False)
    parser.add_argument('--enable_progress_sampling', action = 'store_true', help = 'Whether enable progressive sampling', required=False)
    parser.add_argument('--test_load_weight', type = str, help = 'File name of eval pre-trained weight', required=False)
    parser.add_argument('--sampled_dir', type = str, help = 'Directory of sampled imgs', required=False)
    args = parser.parse_args()
    # End
    
    print(args)
    
    with open(args.input_setting, 'r') as input:
        model_config = json.load(input)
    
    if args.device != None:
        model_config['device'] = args.device

    model_config['state'] = args.state
    
    model_config['save_weight_dir'] = args.save_weight_dir
    
    if args.state == 'eval':
        if args.sample_number_of_batch == None:
            raise ValueError('Missing sample_number_of_batch when in eval mode')
        else:
            model_config['sample_number_of_batch'] = args.sample_number_of_batch
        
        if args.enable_progress_sampling == None:
            raise ValueError('Missing enable_progress_sampling when in eval mode')
        else:
            model_config['enable_progress_sampling'] = args.enable_progress_sampling
        
        if args.test_load_weight == None:
            raise ValueError('Missing test_load_weight')
        else:
            model_config['test_load_weight'] = args.test_load_weight
        
        if args.sampled_dir == None:
            raise ValueError('Missing sampled_dir')
        else:
            model_config['sampled_dir'] = args.sampled_dir
    else:
        if args.start_index == None:
            raise ValueError('Missing start_index')
        if args.enable_load_training_weight == None:
            raise ValueError('Missing enable_load_training_weight')
        if args.epoch == None:
            raise ValueError('Missing epoch')

        model_config['start_index'] = args.start_index
        
        if args.enable_ema != None:
            model_config['enable_ema'] = args.enable_ema
            if args.enable_ema == True:
                if args.ema_update_gap != None or 'ema_update_gap' in model_config:
                    if args.ema_update_gap != None:
                        model_config['ema_update_gap'] = args.ema_update_gap
                else:
                    raise ValueError('Missing ema_update_gap when ema enabled')
        
        if args.enable_load_training_weight == True:
            if args.training_load_weight == None:
                raise ValueError('Missing ema_update_gap when ema enabled')
            else:
                model_config['training_load_weight'] = args.training_load_weight
        else:
            model_config['training_load_weight'] = None

        model_config['epoch'] = args.epoch

    print(json.dumps(model_config, indent=4))
    
    main(model_config)
