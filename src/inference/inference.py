import clip
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

import os
import sys
utils_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'utils'))
sys.path.append(utils_dir)
from utils.transforms import rigid_transform_3D, transform_points_numpy
from utils.constants import rest_pelvis


def test_model(models, diffuser, normalizer, configs, text_embedder, hint_text, prog_ind, joint_orig=None, All_one_model=True, **kwargs):
    # set up
    if All_one_model:
        model= models['model']
        try:
            disc_model = models['disc_model']
        except:
            print("disc_model is not provided!", flush=True)
            disc_model = None
    else:
        assert len(kwargs['model_type']) == len(hint_text), "model_type should have the same length as hint_text"

    device = joint_orig.device
    normalize, denormalize = normalizer
    text_embedder = text_embedder
    batch_size = configs['batch_size']
    seq_len = configs['seq_len']
    channels = configs['channels']
    fixed_frame = configs['fixed_frame']
    use_cfg = configs['use_cfg']
    cfg_alpha = configs['cfg_alpha']

    # for classifier guidance
    cg_alpha = configs['cg_alpha']
    cg_diffusion_steps = configs['cg_diffusion_steps']
    
    # select the prog_ind and hint embedding
    def get_prog_hint(i, prog_ind, hint_emb, model_type=None):
        get_hint_idx = i
        remains = 0
        task_i = None
        for j in range(len(prog_ind)+1):
            if(get_hint_idx>=0):
                get_hint_idx -= len(prog_ind[j])
            else:
                remains = get_hint_idx + len(prog_ind[j-1])
                get_hint_idx = j-1
                break
        prog_ind_i = torch.tensor(prog_ind[get_hint_idx][remains]).unsqueeze(0).to(device)
        if model_type is not None:
            task_i = model_type[get_hint_idx][remains]
        else:
            task_i = None
        hint_emb_i = hint_emb[get_hint_idx].unsqueeze(0)
        return prog_ind_i, hint_emb_i, task_i
    

    epochs_num = 0
    begining_frame = joint_orig[0,:fixed_frame,...].reshape(-1, fixed_frame, channels)
    samples_total = [] 
    orig_samples_total = []

    if hint_text:
        hint_token = clip.tokenize(hint_text).to(device)
        hint_emb = text_embedder.encode_text(hint_token).to(device=device, dtype=torch.float32)
        for i in range(len(prog_ind)):
            epochs_num += len(prog_ind[i])

    ################################################################################
    # autogregresive diffusion
    trans_mats = np.repeat(np.eye(4)[np.newaxis, :, :], batch_size, axis=0)
    trans_mats_orig = np.repeat(np.eye(4)[np.newaxis, :, :], batch_size, axis=0)
    
    for i in range(epochs_num):
        if All_one_model:
            prog_ind_i, hint_emb_i, _ = get_prog_hint(i, prog_ind, hint_emb)
        else:
            prog_ind_i, hint_emb_i, task_model = get_prog_hint(i, prog_ind, hint_emb, kwargs['model_type'])
        joint_orig_i = joint_orig[i].reshape(-1, seq_len, channels)
        
        if not All_one_model:
            model = models[task_model]
            disc_model = models[task_model+'_disc']
        samples = diffuser.sample(model, 
                                batch_size=batch_size, 
                                seq_len=seq_len, 
                                channels=channels,
                                fixed_points=begining_frame, 
                                text=hint_emb_i, 
                                prog_ind=prog_ind_i,
                                joints_orig=joint_orig_i,
                                use_cfg=use_cfg,
                                cfg_alpha=cfg_alpha,
                                disc_model=disc_model,
                                cg_alpha = cg_alpha,
                                cg_diffusion_steps = cg_diffusion_steps,
                                )   
        
        samples = samples[-1] # only consider the last timestep
        samples = denormalize(samples)
        samples = samples.detach().cpu().numpy()
        # for original motion
        orig_samples = denormalize(joint_orig_i).detach().cpu().numpy()
    
        if i==0:
            samples_total.append(samples)
            orig_samples_total.append(orig_samples)
        else:
            samples = samples[:, fixed_frame:, :]
            samples = transform_points_numpy(samples, trans_mats)
            samples_total.append(samples)

            orig_samples = orig_samples[:, fixed_frame:, :]
            orig_samples = transform_points_numpy(orig_samples, trans_mats_orig)
            orig_samples_total.append(orig_samples)
        

        begining_frame = samples[:, -fixed_frame:, :]
        pelvis_new = begining_frame[:, -fixed_frame, :9].reshape(batch_size, 3, 3)
        trans_mats = np.repeat(np.eye(4)[np.newaxis, :, :], batch_size, axis=0)
        for ip, pn in enumerate(pelvis_new):
            _, ret_R, ret_t = rigid_transform_3D(np.matrix(pn), rest_pelvis, False)
            ret_t[1] = 0.0
            rot_euler = R.from_matrix(ret_R).as_euler('zxy')
            shift_euler = np.array([0, 0, rot_euler[2]])
            shift_rot_matrix2 = R.from_euler('zxy', shift_euler).as_matrix()
            trans_mats[ip, :3, :3] = shift_rot_matrix2
            trans_mats[ip, :3, 3] = ret_t.reshape(-1)
        begining_frame = normalize(torch.tensor(transform_points_numpy(begining_frame, np.linalg.inv(trans_mats)), device=device, dtype=torch.float32))
        
        begining_frame_orig = orig_samples[:, -fixed_frame:, :]
        pelvis_new_orig = begining_frame_orig[:, -fixed_frame, :9].reshape(batch_size, 3, 3)
        trans_mats_orig = np.repeat(np.eye(4)[np.newaxis, :, :], batch_size, axis=0)
        for ip, pn in enumerate(pelvis_new_orig):
            _, ret_R, ret_t = rigid_transform_3D(np.matrix(pn), rest_pelvis, False)
            ret_t[1] = 0.0
            rot_euler = R.from_matrix(ret_R).as_euler('zxy')
            shift_euler = np.array([0, 0, rot_euler[2]])
            shift_rot_matrix2 = R.from_euler('zxy', shift_euler).as_matrix()
            trans_mats_orig[ip, :3, :3] = shift_rot_matrix2
            trans_mats_orig[ip, :3, 3] = ret_t.reshape(-1)
        begining_frame_orig = normalize(torch.tensor(transform_points_numpy(begining_frame_orig, np.linalg.inv(trans_mats_orig)), device=device, dtype=torch.float32))

    samples_total = np.concatenate(samples_total, axis=1) 
    orig_samples_total = np.concatenate(orig_samples_total, axis=1)

    
    return samples_total, orig_samples_total