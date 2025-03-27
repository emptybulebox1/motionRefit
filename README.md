# Dynamic Motion Blending for Versatile Motion Editing

![Teaser](assets/teaser.jpeg)

This is the code repository of **Dynamic Motion Blending for Versatile Motion Editing** at **CVPR 2025**.

📝 [**arXiv**](https://arxiv.org) | 🌐 [**Project Page**](https://awfuact.github.io/motionrefit) | 🤗 [**Hugging Face Space**](https://huggingface.co/spaces/Yzy00518/motionReFit)  

## Getting Started  
### Prerequisites  
To run the application, you need to have the following installed:  
- Python 3.10
- Required Python packages (specified in `requirements.txt`)
- git-lfs

### Installation

1. **Clone the Repository**:
    ```sh
    git clone https://github.com/emptybulebox1/motionReFit.git
    cd motionRefit
    ```

2. **Download Checkpoints and SMPL-X Models**:
    - You can find them in https://huggingface.co/Yzy00518/motionReFit, where contains all checkpoints and data required for demo.
    - You can also run app.py, which can automatically download all weights from huggingface repo and put them to correct position.

3. **Install Python Packages**:
    ```sh
    pip install -r requirements.txt
    ```

### Running the Application
  - Before running the application, use git-lfs to pull data from this repo
 ```sh
  git-lfs install
  git-lfs pull
  ```
  - Then start the Gradio application:
    
  ```sh
  python app.py
  ```

## Training
  - Comming soon!



---

# STANCE Dataset

STANCE (Style Transfer, Fine-Grained Adjustment, and Body Part Replacement) is a comprehensive motion editing benchmark that encompasses three common types of motion editing tasks

---

## Download
[**Google Drive**]()

## Tasks

### 1. Regeneration (Body Part Replacement)
This task focuses on precise modifications of specific body part movements. For example, _“replace a waving right hand with a swinging motion”_ or _“change a walking leg movement to running”_. To support this task, we have annotated 13,000 motion sequences. Each sequence is tagged with precise body part masks (UPPER_BODY, LOWER_BODY, BOTH_ARMS, LEFT_ARM, RIGHT_ARM) along with corresponding motion descriptions.

### 2. Style Transfer
This task is designed to change the stylistic expression of a motion while preserving its semantic content. For example, _“convert normal walking to tired walking”_ or _“transform a calm gesture into an angry one”_. We collaborated with experienced motion capture actors to perform multiple emotional and stylistic variations of the same motion. In total, we collected approximately 2 hours of high-quality footage, covering styles such as sexy, angry, and old.

### 3. Fine-Grained Adjustment
This task involves subtle modifications of motion characteristics such as amplitude and force — for example, _“walk faster”_ or _“increase the swing of a hand wave”_. To support this, we have constructed 4,500 motion pairs along with editing descriptions. These triples were partially generated using large language models and later validated by experts to ensure quality.


## File Structure
```plaintext
dataset
    - base_motion
        - humanml3d_pkl_all: SMPLX format files of 24,000 motions from HumanML3D
        - test.txt: Official test set of HumanML3D (https://github.com/EricGuo5513/HumanML3D/blob/main/HumanML3D/test.txt)
    - regen (pkl data is in base_motion)
        - mask_all.json: Annotated body parts. The key of the dictionary is the HumanML3D ID, and the value is the annotated body part.
        - part_annotations.json
    - style_transfer
        - humanml3d_stylized: Stylized motions, with IDs corresponding to those in base_motion
    - adjustment
        - paired_pickle
        - adjustment_annotations_part.json
        - transfer.py: Conversion between paired_pickle and annotations
    - README.md
```

## Data struture
All SMPLX format data is sampled at 20 fps. 

### SMPLX format data(humanml3d_pkl_all, humanml3d_stylized)
```plaintext
    {
        'transl': numpy.ndarray (L, 3),
        'global_orient': numpy.ndarray (L, 3),
        'body_pose': numpy.ndarray (L, 63),
    }
```

### paired_pickle in adjustment
```plaintext
    {
        'src': SMPLX format data,
        'tgt': SMPLX format data,
    }
```
