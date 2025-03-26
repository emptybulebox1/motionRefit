import gradio as gr
import os
import pickle
import requests
import time
import re
from huggingface_hub import hf_hub_download, list_repo_files
from huggingface_hub.errors import LocalEntryNotFoundError

app_root = os.path.dirname(os.path.abspath(__file__))
app_root = os.path.join(app_root, "src/app")

repo_id = "Yzy00518/motionReFit"


selected_videos = {
    '000021': 'walking',
    '000472': 'writing something',
    '001454': 'walking backward',
    '002093': 'cleaning the window',
    '002550': 'walking in a Zig-Zag pattern',
    '003111': 'dancing',
    '003712': 'kneeling down and crawling',
    '004163': 'flying like a bird',
    '004455': 'running in place',
    '004912': 'swimming',
    '005458': 'running forward',
    '005869': 'picking up something',
    '006662': 'falling down',
    '006979': 'punching',
    '007354': 'crawling',
    '007822': 'jumping on both sides',
    '008162': 'dancing like a robot',
    '009768': 'looking back',
    '010193': 'lifting something heavy',
    '013449': 'punching with fists',
    '013659': 'jumping jacks',
    '014920': 'walking in place',
    '015249': 'putting something on their face',
    '015729': 'cleaning the table',
}

def is_six_digit_filename(file_path):
    basename = os.path.basename(file_path)
    return bool(re.match(r'^\d{6}\.[a-zA-Z0-9]+$', basename))

def extract_six_digit_code(file_path):
    basename = os.path.basename(file_path)
    match = re.search(r'(\d{6})', basename)
    return match.group(1) if match else None

def rename_files_with_six_digit_code(file_path):
    if not is_six_digit_filename(file_path):
        return file_path
    six_digit_code = extract_six_digit_code(file_path)
    if six_digit_code in selected_videos:
        new_name = selected_videos[six_digit_code]
        if '.mp4' in file_path:
            new_path = os.path.join(os.path.dirname(file_path), f"{new_name}.mp4")
        elif '.pkl' in file_path:
            new_path = os.path.join(os.path.dirname(file_path), f"{new_name}.pkl")
        else:
            raise ValueError(f"Invalid file extension: {file_path}")
        os.rename(file_path, new_path)
        print(f"Renamed: {file_path} -> {new_path}")
        return new_path
    return None


def download_files_from_huggingface(repo_id, repo_type, max_retries=10):
    file_list = list_repo_files(repo_id=repo_id, repo_type=repo_type)
    for file in file_list:
        relative_path = os.path.dirname(file)
        local_path = os.path.join(os.getcwd(), relative_path)
        if not os.path.exists(local_path):
            os.makedirs(local_path)

        # if is_six_digit_filename(file) and (extract_six_digit_code(file) not in selected_videos):
        #     print(f"Skipping: {file}")
        #     continue
        # if "app_base_motion" in file:
        #     print(f"Skipping:{file}")
        #     continue

        for attempt in range(max_retries):
            try:
                hf_hub_download(repo_id=repo_id, 
                                filename=file, 
                                local_dir=os.getcwd(), 
                                local_dir_use_symlinks=False,
                                resume_download=True,)
                print(f"Successfully Download: {file}")
                # rename_files_with_six_digit_code(os.path.join(local_path, os.path.basename(file)))
                break
            except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectionError, LocalEntryNotFoundError) as e:
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1)*40
                    print(f"Download failed, retrying in {wait_time} seconds")
                    time.sleep(wait_time)
            except FileExistsError:
                print(f"{file} Exists")
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1)*100
                    print(f"Unknown exception, retrying in {wait_time} seconds")
                    time.sleep(wait_time)
                

download_files_from_huggingface(repo_id, 'model')

from src.app.setup_models import *
from src.app.pipeline import pipeline

def select_and_show(data_id):
    video_path = os.path.join(app_root, f"app_base_motion_mp4/{data_id}.mp4")
    return video_path if os.path.exists(video_path) else None


from openai import OpenAI
def __translate(text_raw, temperature=1.5):
    MODEL_NAME = 'gpt-3.5-turbo-0125'
    MAX_TOKENS = 400

    os.environ['OPENAI_API_KEY'] = 'sk-gCaeFye5rjfnhdvDDaE24205108b4bB1Bf1497D6A9EeB704'
    client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY", "<your OpenAI API key if not set as env var>"),
        base_url='https://api.xty.app/v1'
    )

    content = [
        {
            "type": "text",
            "text": F"""
                    You will be given a sentence about a person's action or description about their motion. 
                    First, determine which of the following three tasks best fits the input:
                        regen (regeneration) (Use this option very often.) – Modify or generate a new version of the given action (e.g., 'a person is waving his hands' or 'a person kicks with his left foot').
                        style_transfer – Add an emotional or stylistic tone to the given action. For example, 'Replace a person with a proud look using their upper body'. In this task, the output should include one of these adjectives (angry, sad, proud, old, or sexy) and should ONLY change the action style, NOT the action itself.
                        adjustment – (Use this option very rarely.) Only choose this when the input clearly specifies explicit adjustments such as increasing or decreasing the motion amplitude or switching from clockwise to counterclockwise.
                    Next, simplify and normalize the input text by retaining only the parts that emphasize how the action changes. Remove any details about the person's appearance, gender, body type, scenery, or objects they hold.
                    Only for style_transfer tasks, format the output as:   Replace a person with a X look using their Y body. Here, X must be chosen from [angry, sad, proud, old, sexy] and Y is limited to either 'upper' or 'lower'.
                    For other tasks, the format should be 'The person is doing X with their Y'. Here, X is the action and Y is the body part[upper body, lower body, left arm, right arm, both arms] used to perform the action.
                    Finally, your output should follow the format:   task#text where 'task' is one of regen, style_transfer, or adjustment, and 'text' is the simplified and formatted result.
                    REMEMBER TO USE THE '#' SYMBOL TO SEPARATE THE TASK AND TEXT. For example, if the task is regen and the text is 'a person is waving his hands', the output should be: regen#a person is waving his hands with their both arms
                    REMEMBER YOU CAN ONLY CHOOSE ONE TASK. If the input contains multiple tasks, choose the one that best fits the input. If the input contains no tasks, choose regen.
                    """
        },
    ]
    content.append({
        "type": "text",
        "text": F"""The sentence is: {text_raw}"""
    })

    # print(content)
    messages = [{"role": "user", "content": content}]
    params = {"model": MODEL_NAME, "messages": messages, "max_tokens": MAX_TOKENS, "temperature": temperature}
    result = client.chat.completions.create(**params)
    
    translated_text = result.choices[0].message.content
    print("Translated Text:", translated_text)

    return translated_text


def translate(text_raw):
    tasks = ['regen', 'style_transfer', 'adjustment']
    split = '#'
    tempreture = 1.5
    timer = 0
    while timer <= 5:
        result = __translate(text_raw, tempreture)
        if split in result:
            task, text = result.split(split)[0], result.split(split)[1]
            if task in tasks:
                print("=========GPT-3.5 Turbo successfully translated the text.===========")
                print("Task:", task)
                print("Text:", text)
                print("===================================================================")
                return task, text
            
        tempreture -= 0.2
        timer += 1
    return None, None
    

def inference_warpper(data_id, change_prompt):
    def select_model(task):
        if task == 'regen':
            return {'model': models['regen'], 'disc_model': models['regen_disc']}
        elif task == 'style_transfer':
            return {'model': models['style_transfer'], 'disc_model': models['style_transfer_disc']}
        elif task == 'adjustment':
            return {'model': models['adjustment'], 'disc_model': models['adjustment_disc']}
        else:
            raise ValueError(f"Invalid task: {task}")

    with open(os.path.join(app_root, f"app_base_motion/{data_id}.pkl"), 'rb') as f:
        motion = pickle.load(f)

    task, text_ = translate(change_prompt)
    if task is None:
        print("GPT-3.5 Turbo failed to translate the text.")
        return None
    
    model = select_model(task)
    data = {
        'source': motion,
        'text': text_,
        'prog_ind': None,
        'All_one_model': True,
        'model_type': task,
    }
    ret = pipeline(data, model, device ,diffuser, SEQLEN=16, smplx_pth='deps/smplx')
    return ret

def get_all_videos():
    video_dir = os.path.join(app_root, "app_base_motion_mp4")
    all_files = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]
    all_files.sort()
    return [os.path.splitext(f)[0] for f in all_files]

def bar_length(data_id):
    return 4

with gr.Blocks() as demo:
    gr.Markdown("### Edit a motion with your own instruction")
    all_videos = get_all_videos()

    #with gr.Row():
    data_id_input = gr.Dropdown(
            label=f"Step 1: Select the motion to edit（{len(all_videos)} in total）",
            choices=all_videos,
            multiselect=False,
            allow_custom_value=True,
            #scale=5,
        )
    show_video_button = gr.Button("Step 2: Display the selected motion")
    video_output = gr.Video(label="Selected Motion")

    change_prompt_textbox = gr.Textbox(visible=True, label="Step 3: Write your editing instruction here")
    inference_button = gr.Button("Step4: Start inference")
    output_file = gr.Video(label="Edited Motion (It takes 40 sec to render. The video is truncated to 6sec due to the constraint of computational resources)")
   
    show_video_button.click(
        fn=select_and_show,
        inputs=[data_id_input],
        outputs=[video_output]
    )

    inference_button.click(
        fn=inference_warpper,
        inputs=[data_id_input, change_prompt_textbox],
        outputs=[output_file]
    )

demo.launch()