import os
import torch.distributed as dist
import torch
import sys

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '1253'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def create_smplx_model(fast_smplx_path,
                        model_path,
                        model_type,
                        gender,
                        ext,
                        batch_size,
                        device):
    
    sys.path.insert(0, fast_smplx_path)
    import smplx
    smpl_model = smplx.create(model_path=model_path, 
                              model_type=model_type,
                              gender=gender, 
                              ext=ext,
                              batch_size=batch_size).to(device)
    smpl_model.eval()
    return smpl_model


def load_checkpoint(model, optimizer, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    