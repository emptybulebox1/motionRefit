from argparse import ArgumentParser
from omegaconf import OmegaConf

import os
import sys
from inference.joint2smplx import process_file

if __name__ == '__main__':
    """
    args:
        - input_folder
        - output_folder
    """
    parser = ArgumentParser()
    parser.add_argument('--input_folder', type=str, default=None)
    parser.add_argument('--output_folder', type=str, default=None)
    args = parser.parse_args()

    root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) # motionReFit dir
    config = OmegaConf.load(os.path.join(root, "src", "configs/j2s.yaml"))

    for file_name in os.listdir(args.input_folder):
        if file_name.endswith('.pkl'):
            process_file(file_path=args.input_folder, 
                        file_name=file_name,
                        save_path=args.output_folder,
                        JointsToSMPLX_model_path=os.path.join(root, config.JointsToSMPLX_model_path),
                        smplx_path=os.path.join(root, config.smplx_path),
                        key_list = ['generated_samples', 'original_samples'],
                        interp_s=config.interp_s, 
                        )