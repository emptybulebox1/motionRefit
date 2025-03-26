# generate samples for evaluation & visualization
import torch
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser
from omegaconf import OmegaConf
import pickle

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
print(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

from inference import inference
from utils.inference_utils import set_all_seeds, fix_state_dict, load_hint_texts_from_file, load_mask_from_file, load_file_names, gen_prog_ind
from model.gaussian_diffusion import GaussianDiffusion
from model.unet import Unet
from utils.normalize import set_up_normalization
from utils.constants import TO_24


set_all_seeds(135)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import clip
text_embedder, _ = clip.load("ViT-B/32", device=device)
text_embedder.eval()

def print_config(config):
    print(OmegaConf.to_yaml(config))

def getmodel(model_used, device, model_root, use_step=False, is_disc=False, config=None):

    model = Unet(
        dim_model=config.dim_model,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        dropout_p=config.dropout_p,
        dim_input=config.dim_input,
        dim_output=config.dim_output,
        text_emb=config.text_emb,
        device=device,
        Disc = is_disc,
    ).to(device)
    
    model_path = os.path.join(model_root, f'model_h3d_epoch{model_used}.pth')
    if use_step:
        model_path = os.path.join(model_root, f'model_h3d_step{model_used}.pth')
    print("==>", model_path)
    if torch.cuda.is_available():
        state_dict = torch.load(model_path)
    else:
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    
    fixed_state_dict = fix_state_dict(state_dict)['model_state_dict']
    fixed_state_dict = fix_state_dict(fixed_state_dict)
    model.load_state_dict(fixed_state_dict)
    model.eval()
    return model

if __name__ == '__main__':
    """
    args:
        - task: "regen", "style_transfer", "adjustment"
    """
    parser = ArgumentParser()
    parser.add_argument('--task', type=str, default='regen')
    args = parser.parse_args()
    task_config = OmegaConf.load(f"configs/inference/{args.task}.yaml")
    base_config = OmegaConf.load("configs/base.yaml")
    config = OmegaConf.merge(base_config, task_config)

    text_path = os.path.join(project_root, config.test_data_path, config.text_path)
    mask_path = os.path.join(project_root, config.test_data_path, config.mask_path)
    joints_src_path = os.path.join(project_root, config.test_data_path, config.joints_src_path)
    gen_file_names_path = os.path.join(project_root, config.test_data_path, config.gen_file_names_path)

    hint_text_all = load_hint_texts_from_file(text_path)
    mask_all = load_mask_from_file(mask_path)
    gen_file_names = load_file_names(gen_file_names_path)
    joints_orig_all = torch.tensor(np.load(joints_src_path), dtype=torch.float32, device=device)
    prog_ind_all = gen_prog_ind(num_cases=len(hint_text_all), sublist_length = 4)#sublist_length=config.sublist_length)

    models = {
        'model': getmodel(config.model_used, 
                          device=device, 
                          model_root=os.path.join(project_root, config.model_path, config.task), 
                          use_step=False, 
                          is_disc=False,
                          config = config.unet,
                          ),
        'disc_model': getmodel(config.disc_model_used, 
                               device=device, 
                               model_root=os.path.join(project_root, config.disc_model_path, config.task), 
                               use_step=True,
                               is_disc=True,
                               config = config.unet,
                               ),
    }
    
    diffuser = GaussianDiffusion(device=device, 
                                fix_mode=config.diffusion.fix_mode, 
                                text_emb=config.diffusion.text_emb, 
                                fixed_frames=config.diffusion.fixed_frames,
                                seq_len=config.diffusion.seq_len,
                                timesteps=config.diffusion.timesteps, 
                                beta_schedule=config.diffusion.beta_schedule)

    normalize, denormalize = set_up_normalization(device=device, seq_len=config.seq_len, scale=3)
    joints_orig = normalize(joints_orig_all)


    test_configs = {
        'batch_size': config.batch_size,
        'seq_len': config.seq_len,
        'channels': config.channels,
        'fixed_frame': config.fixed_frame,
        'use_cfg': config.use_cfg,
        'cfg_alpha': config.cfg_alpha,
        'cg_alpha': config.cg_alpha,
        'cg_diffusion_steps': config.cg_diffusion_steps,
    }
    for i in tqdm(range(len(hint_text_all))):

        generated_samples, orig = inference.test_model(
                                                    models=models, 
                                                    diffuser=diffuser, 
                                                    normalizer=(normalize, denormalize), 
                                                    configs=test_configs, 
                                                    text_embedder=text_embedder, 
                                                    hint_text=hint_text_all[i], 
                                                    prog_ind=prog_ind_all[i], 
                                                    joint_orig=joints_orig[i]
                                                )
        
        # only consider 24 joints instaed of 28
        generated_samples = generated_samples.reshape(1, -1, config.joints_num, 3)[..., TO_24, :].reshape(1, -1, 72)
        orig = orig.reshape(1, -1, config.joints_num, 3)[..., TO_24, :].reshape(1, -1, 72)

        combined_dict = {
            'generated_samples': generated_samples,
            'original_samples': orig, 
            'text' : hint_text_all[i][0] + f"{i}",
            'mask' : mask_all[i]
        }

        save_pth = os.path.join(project_root, config.save_path)
        if not os.path.exists(save_pth):
            os.makedirs(save_pth)

        with open(os.path.join(save_pth, f'{gen_file_names[i]}.pkl'), 'wb') as file:
            pickle.dump(combined_dict, file)