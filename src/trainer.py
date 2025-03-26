import argparse
import torch
import os
import logging
from omegaconf import OmegaConf
from train import train_model

os.environ['NCCL_P2P_DISABLE'] = '0'
os.environ['NCCL_IB_DISABLE'] = '0'


if __name__ == "__main__":
    """
    python train.py \
    --task regen/style_transfer/adjustment \
    --start 0 \ # 0 from scratch, n from checkpoint n
    --end 4000 \ # total epochs, default 4000
    --start_from_folder ../models/regen \ # path to checkpoint
    --save_folder ../models/regen \ # path to save model
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='regen')
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=4000)
    parser.add_argument('--start_from_folder', type=str, default=None)
    parser.add_argument('--save_folder', type=str, default=None)

    args = parser.parse_args()

    world_size = torch.cuda.device_count()
    
    logger_name = f'{args.task}_'
    checkpoint_path = None
    if args.start == 0:
        logger_name += ''
        start_epoch = 0
    else:
        checkpoint_path = os.path.join(args.start_from_folder, f'model_h3d_epoch{args.start}.pth')
        assert os.path.exists(checkpoint_path), f'Checkpoint file {checkpoint_path} not found!'
        logger_name += f'continue_from_epoch_{args.start}_'
        start_epoch = args.start
    
    import datetime
    now = datetime.datetime.now()
    logger_name += f'{now.strftime("%m-%d_%H-%M")}'
    logger_name += '.log'

    base_config = OmegaConf.load("src/configs/train/base_config.yaml")
    task_config = OmegaConf.load(f"src/configs/train/tasks/{args.task}.yaml")
    config = OmegaConf.merge(base_config, task_config)

    logger_name = os.path.join(config.train.logger_pth, logger_name)
    if not os.path.exists(config.train.logger_pth):
        os.makedirs(config.train.logger_pth)
    logging.basicConfig(filename=logger_name,
                    level=logging.INFO,
                    format='%(asctime)s:%(levelname)s:%(message)s')

    torch.multiprocessing.spawn(train_model, 
                                args=(world_size, 
                                      start_epoch,
                                      args.end, 
                                      checkpoint_path, 
                                      config,
                                      logging.getLogger(),), 
                                nprocs=world_size, 
                                join=True)
