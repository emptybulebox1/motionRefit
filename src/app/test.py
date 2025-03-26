from setup_models import *
from pipeline import pipeline

import os
import pickle
import sys

text = [
        'The person is jogging with their upper body.',
        'Replace a person with a proud look using their upper body.',
        'The person is playing violin.', 
        ]


task = [
        ['regen', 'regen'],
        ['style_transfer', 'style_transfer'], 
        ['regen', 'regen'],        
        ]

progress_indicator = [
                    [0.2, 0.5], 
                    [0.2, 0.75],
                    [0.5, 0.5], 
                      ]

with open('/home/ziye/cvpr/motion_editing/app/input/000135.pkl', 'rb') as f:
    motion = pickle.load(f)


data = {
    'source': motion,
    'task': task,
    'text': text,
    'prog_ind': progress_indicator,
    'All_one_model': False,
    'model_type': task,
}
device = torch.device('cuda')

pipeline(data, models, device ,diffuser, SEQLEN=16, smplx_pth='/home/ziye/cvpr/motion_editing/app/deps/smplx')