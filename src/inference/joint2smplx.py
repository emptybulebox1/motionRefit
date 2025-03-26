import os
import sys
import numpy as np
import torch
from torch import nn
import pickle
from scipy.interpolate import interp1d

#############Import fast smplx(modified from original ver)
local_smplx_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..', 'deps/smplx'))
sys.path.insert(0, local_smplx_path)
import smplx_fast

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.transforms import matrix_to_axis_angle, rotation_6d_to_matrix
from utils.constants import pelvis_shift, relaxed_hand_pose, SELECTED_JOINTS24


###########This model is used to predict the initial pose for the optimization###########
class JointsToSMPLX(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, **kwargs):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.layers(x)

def get_j2s_model(ckpt_path, 
                input_dim=72, 
                output_dim=132, 
                hidden_dim=64,
                device='cpu'):
    model_joints_to_smplx = JointsToSMPLX(input_dim=input_dim, 
                                            output_dim=output_dim, 
                                            hidden_dim=hidden_dim
                                            )
    if device == 'cpu':
        map_location = torch.device('cpu')
    else:
        map_location = device

    model_joints_to_smplx.load_state_dict(torch.load(ckpt_path, map_location=map_location))
    model_joints_to_smplx.eval()
    return model_joints_to_smplx

###########This model is used to predict the initial pose for the optimization###########


def optimize_smpl(pose_pred, joints, joints_ind, smplx_path, print_loss=True):
    device = joints.device
    len = joints.shape[0]

    smpl_model = smplx_fast.create(smplx_path, 
                              model_type='smplx_joint_only',
                              gender='male', ext='npz',
                              num_betas=10,
                              use_pca=False,
                              create_global_orient=True,
                              create_body_pose=True,
                              create_betas=True,
                              create_left_hand_pose=True,
                              create_right_hand_pose=True,
                              create_expression=True,
                              create_jaw_pose=True,
                              create_leye_pose=True,
                              create_reye_pose=True,
                              create_transl=True,
                              batch_size=len,
                              ).to(device)
    smpl_model.eval()

    joints = joints.reshape(len, -1, 3) + torch.tensor(pelvis_shift).to(device)
    pose_input = torch.nn.Parameter(pose_pred.detach(), requires_grad=True)
    transl = torch.nn.Parameter(torch.zeros(pose_pred.shape[0], 3).to(device), requires_grad=True)
    left_hand = torch.from_numpy(relaxed_hand_pose[:45].reshape(1, -1).repeat(pose_pred.shape[0], axis=0)).to(device)
    right_hand = torch.from_numpy(relaxed_hand_pose[45:].reshape(1, -1).repeat(pose_pred.shape[0], axis=0)).to(device)
    optimizer = torch.optim.Adam(params=[pose_input, transl], lr=0.05)
    loss_fn = nn.MSELoss()
    vertices_output = None
    
    for step in range(120):
        smpl_output = smpl_model(transl=transl, 
                                 body_pose=pose_input[:, 3:], 
                                 global_orient=pose_input[:, :3], 
                                 return_verts=True,
                                 left_hand_pose=left_hand,# @ left_hand_components[:hand_pca],
                                 right_hand_pose=right_hand,# @ right_hand_components[:hand_pca],
                                 )
        joints_output = smpl_output[:, joints_ind].reshape(len, -1, 3)
        loss = loss_fn(joints[:, :], joints_output[:, :])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if print_loss:
        print(loss.item(), flush=True)

    return pose_input.detach().cpu().numpy(), \
            transl.detach().cpu().numpy(), \
            left_hand.detach().cpu().numpy(), \
            right_hand.detach().cpu().numpy(), \
            vertices_output


def joints_to_smpl(model, joints, joints_ind, interp_s, smplx_path, print_loss=True):
    joints = interpolate_joints(joints, scale=interp_s)
    input_len = joints.shape[0]
    joints = joints.reshape(input_len, -1, 3)
    joints = joints.permute(1, 0, 2)
    trans_np = joints[0].detach().cpu().numpy()
    joints = joints - joints[0]
    joints = joints.permute(1, 0, 2)
    joints = joints.reshape(input_len, -1)
    pose_pred = model(joints)

    pose_pred = pose_pred.reshape(-1, 6)
    pose_pred = matrix_to_axis_angle(rotation_6d_to_matrix(pose_pred)).reshape(input_len, -1)
    pose_output, transl, left_hand, right_hand, vertices = optimize_smpl(pose_pred, 
                                                                         joints, 
                                                                         joints_ind, 
                                                                         smplx_path,
                                                                         print_loss=print_loss)
    transl = trans_np - np.array(pelvis_shift) + transl
    return pose_output, transl, left_hand, right_hand, vertices

def interpolate_joints(joints, scale):
    if scale == 1:
        return joints
    device = joints.device
    joints = joints.detach().cpu().numpy()
    in_len = joints.shape[0]
    out_len = int(in_len * scale)
    joints = joints.reshape(in_len, -1)
    x = np.array(range(in_len))
    xnew = np.linspace(0, in_len - 1, out_len)
    f = interp1d(x, joints, axis=0)
    joints_new = f(xnew)
    joints_new = torch.from_numpy(joints_new).to(device).float()

    return  joints_new




def process_file(file_path,                     # input dir
                file_name,                      # input file
                save_path,                      # output dir
                JointsToSMPLX_model_path,       # JointsToSMPLX weight
                smplx_path,                     # smplx weight
                key_list = ['generated_samples', 'original_samples'],
                joints_ind = SELECTED_JOINTS24,
                interp_s=2,                     # 2*10=20 fps
                ):


    data = np.load(os.path.join(file_path, file_name), allow_pickle=True)
    model = get_j2s_model(ckpt_path=JointsToSMPLX_model_path, device='cpu')

    for key in key_list: # original_samples, generated_samples, GT
        if key in data:
            joints = torch.tensor(data[key], dtype=torch.float32).reshape(-1, 72)

            print_loss=False
            if key == 'generated_samples':
                print_loss=True
            
            pose, transl, left_hand, right_hand, vertices = joints_to_smpl(model,
                                                                        joints, 
                                                                        joints_ind, 
                                                                        interp_s,
                                                                        smplx_path,
                                                                        print_loss=print_loss)
            try:
                data_text = data['text']
            except:
                data_text = None

            output_data = {
                'body_pose': pose[:, 3:],
                'global_orient': pose[:, :3],
                'transl': transl,
                'left_hand': left_hand,
                'right_hand': right_hand,
                'vertices': vertices,
                'text': data_text,
            }

            if key == 'generated_samples':
                try:
                    output_data['mask'] = data['mask']
                except:
                    output_data['mask'] = None
               
            if not os.path.exists(os.path.join(save_path, key)):
                os.makedirs(os.path.join(save_path, key))

            output_file = os.path.join(os.path.join(save_path, key), file_name)
            with open(output_file, 'wb') as file:
                pickle.dump(output_data, file)