import os
import sys
import pickle
import shutil
from threading import Timer
import subprocess

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
app_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

from utils.inference_utils import gen_prog_ind
from utils.constants import TO_24
from inference import inference

from omegaconf import OmegaConf
from inference.joint2smplx import process_file

def delete_folder(folder_path):
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)

def rand_folder_name():
    import time
    return str(time.time()).replace('.', '')

def pipeline(data, models, device, diffuser, **kwargs):
    from src.app.process_data import get_a_sample
    from src.app.setup_models import text_embedder, test_configs, normalize, denormalize

    len_data = min(data['source']['transl'].shape[0]//((kwargs['SEQLEN']-2)*2), 4)
    if len_data < 4:
        return None # not enough data
    
    joints_orig = get_a_sample(data['source'],
                                len_data,
                                kwargs['SEQLEN'],
                                smplx_pth=os.path.abspath(os.path.join(app_root, '../deps/smplx/models'))
                                ).to(device)
    
    joints_orig = normalize(joints_orig)
    
    hint_text = data['text']

    if data['prog_ind'] is None:
        prog_ind = gen_prog_ind(num_cases=1, sublist_length=len_data)[0]
    else:
        prog_ind = data['prog_ind']

    print("***Begin inference!***")
    generated_samples, orig = inference.test_model(
                                                models=models, 
                                                diffuser=diffuser, 
                                                normalizer=(normalize, denormalize), 
                                                configs=test_configs, 
                                                text_embedder=text_embedder, 
                                                hint_text=hint_text, 
                                                prog_ind=prog_ind, 
                                                joint_orig=joints_orig,
                                                All_one_model=data['All_one_model'],
                                                model_type=data['model_type']
                                            )
    
    generated_samples = generated_samples.reshape(1, -1, 28, 3)[..., TO_24, :].reshape(1, -1, 72)
    orig = orig.reshape(1, -1, 28, 3)[..., TO_24, :].reshape(1, -1, 72)

    combined_dict = {
        'generated_samples': generated_samples,
        'original_samples': orig, 
        'text' : hint_text,
    }
    # return combined_dict

    input_folder = os.path.join(app_root, rand_folder_name())
    output_folder = os.path.join(app_root, rand_folder_name())

    if not os.path.exists(input_folder):
        os.makedirs(input_folder)
    with open(os.path.join(input_folder, 'temp.pkl'), 'wb') as file:
        pickle.dump(combined_dict, file)
    
    
    j2s_config = OmegaConf.load(os.path.join(app_root, "configs/j2s.yaml"))

    print("Joint2smplx")
    for file_name in os.listdir(input_folder):
        if file_name.endswith('.pkl'):
            process_file(file_path=input_folder, 
                        file_name=file_name,
                        save_path=output_folder,
                        JointsToSMPLX_model_path=os.path.abspath(os.path.join(app_root, '..', j2s_config.JointsToSMPLX_model_path)),
                        smplx_path=os.path.abspath(os.path.join(app_root, '..', j2s_config.smplx_path)),
                        key_list = ['generated_samples'],
                        # remenber to remove original samples when using app
                        interp_s=j2s_config.interp_s, 
                        )
    
    # run render process:
    render_script_path = os.path.join(app_root, 'app/render.py')
    input_file_path = os.path.join(output_folder, 'generated_samples/temp.pkl')
    
    try:
        result = subprocess.run(
            [sys.executable, render_script_path, '--motion_path', input_file_path, '--title', hint_text],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        print("渲染完成，输出:", result.stdout)
        if result.stderr:
            print("渲染错误:", result.stderr)
    except subprocess.CalledProcessError as e:
        print(f"渲染失败，错误代码: {e.returncode}")
        print(f"错误输出: {e.stderr}")

    Timer(100, delete_folder, [input_folder]).start()
    Timer(100, delete_folder, [output_folder]).start()
    
    return os.path.join(output_folder, 'generated_samples/temp.mp4')
    # return os.path.join(output_folder, 'generated_samples/temp.pkl')