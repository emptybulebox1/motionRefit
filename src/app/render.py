import smplx
import torch
import pickle
import numpy as np
import os
import trimesh
import cv2
os.environ['PYOPENGL_PLATFORM'] = 'egl'
import pyrender

root_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..'))
def get_smplx_model(bs, deivce):
    smpl_model = smplx.create(os.path.join(root_dir, 'deps/smplx/models'),
                                    model_type='smplx',
                                    gender='male',
                                    ext='npz',
                                    batch_size=bs,
                                    ).to(device)
    return smpl_model.eval()


def render_motion_to_video(motion, device, output_path, title, step=2):
    length = motion['transl'].shape[0]
    model = get_smplx_model(length, device)
    vertices = model(
        body_pose = torch.tensor(motion['body_pose'], dtype=torch.float32).to(device),
        transl = torch.tensor(motion['transl'], dtype=torch.float32).to(device),
        global_orient = torch.tensor(motion['global_orient'], dtype=torch.float32).to(device)
    ).vertices.detach().cpu().numpy()
    faces = model.faces
    
    temp_dir = "temp_render_images"
    os.makedirs(temp_dir, exist_ok=True)
    
    width, height = 480, 320
    images = []
    for i in range(0, length, step):
        trimesh_mesh = trimesh.Trimesh(vertices=vertices[i], faces=faces)
        mesh = pyrender.Mesh.from_trimesh(trimesh_mesh)
        
        scene = pyrender.Scene(ambient_light=[0.5, 0.5, 0.5])
        scene.add(mesh)
        
        camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
        angle = -np.pi / 8
        camera_pose = np.array([
            [1, 0, 0, 0],
            [0, np.cos(angle), -np.sin(angle), 2.0],
            [0, np.sin(angle), np.cos(angle), 4.0],
            [0, 0, 0, 1]
        ])
        scene.add(camera, pose=camera_pose)
        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=2.0)
        scene.add(light, pose=camera_pose)
        
        r = pyrender.OffscreenRenderer(width, height)
        color, _ = r.render(scene)
        r.delete()
        
        image = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
        
        img_path = os.path.join(temp_dir, f"frame_{i:04d}.png")
        cv2.imwrite(img_path, image)
        images.append(image)
    
    if images:
        height, width, _ = images[0].shape
        sampled_fps = int(20 / step)
        temp_video_path = output_path.replace('.mp4', '_temp.mp4')

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(temp_video_path, fourcc, sampled_fps, (width, height))
        for image in images:
            video.write(image)
        video.release()

        try:
            import subprocess
            cmd = [
                'ffmpeg',
                '-y',
                '-i', temp_video_path,
                '-filter:v', f'minterpolate=fps={24}:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1',
                '-c:v', 'libx264',
                '-preset', 'medium',
                '-crf', '18',
                output_path
            ]
            
            subprocess.run(cmd, check=True)
            print(f"帧插值完成，最终视频保存至: {output_path}")
            os.remove(temp_video_path)
        except Exception as e:
            print(f"帧插值失败: {e}")
            os.rename(temp_video_path, output_path)
            print(f"使用原始视频: {output_path}")
    else:
        print("没有图像可用于创建视频")
    
    for file in os.listdir(temp_dir):
        os.remove(os.path.join(temp_dir, file))
    os.rmdir(temp_dir)


from argparse import ArgumentParser
import time
if __name__ == "__main__":
    parser = ArgumentParser()
    print('******Start rendering...******')
    start_time = time.time()
    
    parser.add_argument("--motion_path", type=str, required=True)
    parser.add_argument("--title", type=str, default="rendered_motion.mp4")

    args = parser.parse_args()
    with open(args.motion_path, 'rb') as f:
        motion = pickle.load(f)
    output_path = args.motion_path.replace(".pkl", ".mp4")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(args.title)
    render_motion_to_video(motion, device, output_path, args.title)

    print(f"******Rendering finished in {time.time() - start_time:.2f} seconds.******")
