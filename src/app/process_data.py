import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.constants import SELECTED_JOINT28

local_smplx_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..', 'deps/smplx'))
sys.path.insert(0, local_smplx_path)
import smplx

def get_smplx_model(bs, smplx_pth):
    smpl_model = smplx.create(model_path=smplx_pth, 
                                model_type='smplx',
                                gender='male', ext='npz',
                                batch_size=bs,
                                )
    smpl_model.eval()
    return smpl_model

def get_a_sample(mo_data, motion_len=6, SEQLEN=16, smplx_pth=None):
    SEQLENTIMES2 = SEQLEN*2

    transl_all = []
    global_orient_all = []
    body_pose_all = []

    transl = mo_data['transl']                 # L,3
    global_orient = mo_data['global_orient']   # L,3
    body_pose = mo_data['body_pose']           # L,63 -> L,21,3
    length = transl.shape[0]
    print("Get a sample")
     
    if (length - (SEQLENTIMES2-2)*motion_len) <= 0:
        return None
    
    indices = np.arange(0, (SEQLENTIMES2-1)*motion_len, SEQLENTIMES2-1)
        
    for idx in indices:
        transl_i = transl[idx:idx+SEQLENTIMES2:2]
        global_orient_i = global_orient[idx:idx+SEQLENTIMES2:2]
        body_pose_i = body_pose[idx:idx+SEQLENTIMES2:2]

        b_shape = body_pose_i.shape
        body_pose_i = body_pose_i.reshape(-1, 3)

        transl_i = transl_i - np.array([transl_i[0, 0], 0., transl_i[0, 2]])
        first_frame_euler = R.from_rotvec(global_orient_i[0]).as_euler('zxy')
        first_frame_euler = np.array([0, 0, -first_frame_euler[2]])
        first_frame_matrix = R.from_euler('zxy', first_frame_euler).as_matrix()
        global_orient_i = (
                R.from_matrix(first_frame_matrix) * R.from_rotvec(global_orient_i)
            ).as_rotvec()
        transl_i = transl_i @ first_frame_matrix.T

        transl_all.append(transl_i)
        global_orient_all.append(global_orient_i)
        body_pose_all.append(body_pose_i.reshape(b_shape))
    
    transl_all = np.stack(transl_all).reshape(-1, 3)
    global_orient_all = np.stack(global_orient_all).reshape(-1, 3)
    body_pose_all = np.stack(body_pose_all).reshape(-1, 63)

    assert (motion_len*SEQLEN)==transl_all.shape[0]
    batch_size=(motion_len*SEQLEN)
    smpl_model = get_smplx_model(batch_size, smplx_pth=smplx_pth)

    with torch.no_grad():
        joints = smpl_model(
                            body_pose=torch.tensor(body_pose_all, dtype=torch.float32),
                            global_orient=torch.tensor(global_orient_all, dtype=torch.float32),
                            transl=torch.tensor(transl_all, dtype=torch.float32),
                            ).joints[:, SELECTED_JOINT28]
    print("Get a sample returns successfully!")
    return joints.reshape(motion_len, SEQLEN, 28, 3) # a Tensor of size (6, 16, 28, 3)
