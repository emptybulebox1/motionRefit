import torch
from omegaconf import OmegaConf
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
print(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
src_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
project_root = os.path.abspath(os.path.join(src_root, '..'))


from utils.inference_utils import set_all_seeds, fix_state_dict
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


base_config = OmegaConf.load(os.path.join(src_root, "configs/base.yaml"))
regen_config = OmegaConf.load(os.path.join(src_root, "configs/inference/regen.yaml"))
regen_config = OmegaConf.merge(base_config, regen_config)
style_transfer_config = OmegaConf.load(os.path.join(src_root, "configs/inference/style_transfer.yaml"))
style_transfer_config = OmegaConf.merge(base_config, style_transfer_config)
adjustment_config = OmegaConf.load(os.path.join(src_root, "configs/inference/adjustment.yaml"))
adjustment_config = OmegaConf.merge(base_config, adjustment_config)

models = {
    'regen': getmodel(regen_config.model_used, 
                        device=device, 
                        model_root=os.path.join(project_root, regen_config.model_path, regen_config.task), 
                        use_step=False, 
                        is_disc=False,
                        config = regen_config.unet,
                        ),
    'regen_disc': getmodel(regen_config.disc_model_used, 
                            device=device, 
                            model_root=os.path.join(project_root, regen_config.disc_model_path, regen_config.task), 
                            use_step=True,
                            is_disc=True,
                            config = regen_config.unet,
                            ),
    'style_transfer': getmodel(style_transfer_config.model_used,
                                    device=device,
                                    model_root=os.path.join(project_root, style_transfer_config.model_path, style_transfer_config.task),
                                    use_step=False,
                                    is_disc=False,
                                    config = style_transfer_config.unet,
                                    ),
    'style_transfer_disc': getmodel(style_transfer_config.disc_model_used,
                                    device=device,
                                    model_root=os.path.join(project_root, style_transfer_config.disc_model_path, style_transfer_config.task),
                                    use_step=True,
                                    is_disc=True,
                                    config = style_transfer_config.unet,
                                    ),
    'adjustment': getmodel(adjustment_config.model_used,
                            device=device,
                            model_root=os.path.join(project_root, adjustment_config.model_path, adjustment_config.task),
                            use_step=False,
                            is_disc=False,
                            config = adjustment_config.unet,
                            ),
    'adjustment_disc': getmodel(adjustment_config.disc_model_used,
                                device=device,
                                model_root=os.path.join(project_root, adjustment_config.disc_model_path, adjustment_config.task),
                                use_step=True,
                                is_disc=True,
                                config = adjustment_config.unet,
                                ),
}

diffuser = GaussianDiffusion(device=device, 
                            fix_mode=base_config.diffusion.fix_mode, 
                            text_emb=base_config.diffusion.text_emb, 
                            fixed_frames=base_config.diffusion.fixed_frames,
                            seq_len=base_config.diffusion.seq_len,
                            timesteps=base_config.diffusion.timesteps, 
                            beta_schedule=base_config.diffusion.beta_schedule)

normalize, denormalize = set_up_normalization(device=device, seq_len=base_config.seq_len, scale=3, 
                                              norm_path=os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/norm_scaled.npy')))


test_configs = {
    'batch_size': 1,
    'seq_len': base_config.seq_len,
    'channels': base_config.channels,
    'fixed_frame': base_config.fixed_frame,
    'use_cfg': base_config.use_cfg,
    'cfg_alpha': regen_config.cfg_alpha,
    'cg_alpha': regen_config.cg_alpha,
    'cg_diffusion_steps': regen_config.cg_diffusion_steps,
}