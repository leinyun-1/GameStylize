import torch 
import os 
from diffsynth import WanVideoPipeline, ModelManager, save_video
from fps_dataset import VideoClipDataset
from thuman_dataset import VideoDataset,TextDataset,LatentDataset
from csgo_dataset import CsgoImageDataset,CsgoVideoDataset
from doom_dataset import DoomH5Dataset
from utils import save_tensor_to_video
from PIL import Image
import numpy as np
import imageio

def main():
    #root = '/root/leinyu/data/ngr/ngr_clip_latent'
    #root = '/root/leinyu/data/doom/freedoom_random_128_u2_rs_fix_v6_1280_720_hdf5'
    #root = '/root/leinyu/data/round/latent/'
    root = '/root/leinyu/data/thuman2.1/Thuman2.1_norm_render_1/latent_11'
    #root = '/root/leinyu/data/fps/latent_d2v/pubg/erangel'
    dest = '/root/leinyu/code/results/wan/vae/thuman2.1'
    os.makedirs(dest,exist_ok=True)

    model_path = ['/dockerdata/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth']
    model_manager = ModelManager(torch_dtype=torch.bfloat16, device="cuda:1")
    model_manager.load_models(model_path)
    pipe = WanVideoPipeline.from_model_manager(model_manager)

    #clips = os.listdir(root)
    clips = sorted(os.listdir(root), 
              key=lambda x: os.path.getmtime(os.path.join(root, x)),
              reverse=True)
    eval_num = 2
    
    # 随机选择eval_num个clips
    selected_clips = np.random.choice(clips, size=eval_num, replace=False)
    for clip in selected_clips:
        latent_path = os.path.join(root,clip)
        data = torch.load(latent_path)
        for k,v in data.items():
            print(k,v.shape)    
        latent = data['latents'].unsqueeze(0) # 1 21 c 60 104 

        frames = pipe.decode_video(latent,tiled=False)
        video = pipe.tensor2video(frames[0])

        save_path = os.path.join(dest,clip+'.mp4')
        save_video(video, save_path, fps=15, quality=5)

        if 'control_latents' in data:
            depth_latent = data['control_latents'].unsqueeze(0)
            depth_frames = pipe.decode_video(depth_latent,tiled=False)
            depth_video = pipe.tensor2video(depth_frames[0])
            save_path = os.path.join(dest,clip+'_depth.mp4')
            save_video(depth_video, save_path, fps=15, quality=5)
        
        if "normal_latents" in data:
            depth_latent = data['normal_latents'].unsqueeze(0)
            depth_frames = pipe.decode_video(depth_latent,tiled=False)
            depth_video = pipe.tensor2video(depth_frames[0])
            save_path = os.path.join(dest,clip+'_normal.mp4')
            save_video(depth_video, save_path, fps=15, quality=5)


def test_fps_dataset():
    dataset = VideoClipDataset(root='/root/leinyu/data/fps/varolent')
    print(len(dataset))
    data = dataset[0]
    print(data['pixel_values'].shape)

    save_path = 'data/examples/fps/varolent.mp4'
    save_tensor_to_video(data['pixel_values'],save_path)

def test_thuman_dataset():
    dataset = VideoDataset()
    print(len(dataset))
    data = dataset[0]
    print(data['pixel_values'].shape)

    save_path = 'data/examples/thuman/round.mp4'
    #save_tensor_to_video(data['pixel_values'],save_path)

    Image.fromarray(data['first_frame'].numpy().astype('uint8')).save(save_path.replace('.mp4','.png'))

def test_thuman_latent_dataset():
    dataset = LatentDataset()
    print(len(dataset))
    data = dataset[100]
    for k,v in data.items():
        print(k,v.shape)

def test_text_dataset():
    dataset = TextDataset()
    print(len(dataset))
    data = dataset[0]
    print(data['text']) 
    print(data['path'])

def test_prompt_emb():
    file_path = '/root/leinyu/data/round/round_video_1/0000/caption_qwen_vltensors.pth'
    data = torch.load(file_path,map_location='cpu')
    print(data['prompt_emb'].shape)

def make_data_examples():
    dest_root = 'data/examples/demo'
    dataset = CsgoVideoDataset(image_size=(480,832),num_frames=900)
    dataset_len = len(dataset)
    for i in range(10):
        data = dataset[np.random.randint(0,dataset_len)]
        img = data['pixel_values'] # 3 f h w (-1~1)
        img = (img + 1) / 2 * 255
        img = img.permute(1,2,3,0).numpy().astype('uint8')
        save_path = os.path.join(dest_root,f'csgo_{i}.mp4')
        writer = imageio.get_writer(save_path, fps=15)
        for frame in img:
            writer.append_data(frame)
        writer.close()
    
    # dataset = DoomH5Dataset(size=(4,480,832))
    # dataset_len = len(dataset)
    # for i in range(10):
    #     data = dataset[np.random.randint(0,dataset_len)]
    #     img = data['first_frame'] # h w 3 (0-255)
    #     img = img.numpy().astype('uint8')
    #     save_path = os.path.join(dest_root,f'doom_{i}_first_frame.png')
    #     Image.fromarray(img).save(save_path)

def test_ngr_latent():
    file_path = '/mnt/aigc_cq/shared/game_videos/NGR_data/NGR_dataset_v2_v3_0621_sub2_walk_F16_WANVAE/167673.h5'
    import h5py
    with h5py.File(file_path, 'r') as f:
        print(f.keys())
        print(f['latent'].shape) # (1, 16, 617, 64, 112)

def test_ngr_dataset():
    from tqdm import tqdm 
    from ngr_dataset import NgrVideoClipDataset
    dataset = NgrVideoClipDataset()
    print(len(dataset))
    # for i in range(10):
    #     data = dataset[i+np.random.randint(len(dataset))]
    #     print(data['pixel_values'].shape)
    
    #     save_path = f'data/examples/ngr/ngr_video_{i}.mp4'
    #     save_tensor_to_video(data['pixel_values'],save_path)
    for i in tqdm(range(1000)):
        data = dataset[i]
        continue

def test_csgo_d2v_dataset():
    from csgo_dataset import CsgoD2VDataset
    dataset = CsgoD2VDataset()
    print(len(dataset))
    # for i in range(10):
    #     data = dataset[i+np.random.randint(len(dataset))]
    #     print(data['pixel_values'].shape)
    
    #     save_path = f'data/examples/csgo/csgo_d2v_video_{i}.mp4'
    #     save_tensor_to_video(data['pixel_values'],save_path)

    #     save_path = f'data/examples/csgo/csgo_d2v_depth_video_{i}.mp4'
    #     save_tensor_to_video(data['depth_values'],save_path)

    data = dataset[0]
    depth = data['depth_values']  # (1, 81, H, W)
    name = data['name']
    print(name)
    print(depth.shape)  # (1, 81, H, W)

    model_path = ['/dockerdata/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth']
    model_manager = ModelManager(torch_dtype=torch.bfloat16, device="cuda:1")
    model_manager.load_models(model_path)
    pipe = WanVideoPipeline.from_model_manager(model_manager)

    depth = depth.to(dtype=pipe.torch_dtype, device=pipe.device)  # (1, 81, H, W)
    depth_latent = pipe.encode_video(depth.unsqueeze(0), tiled=False)  # (1, 21, C, H, W)
    print(depth_latent.shape)  # (1, 21, C, H, W)

def test_doom_dataset():
    from doom_dataset import DoomH5Dataset
    dataset = DoomH5Dataset(size=(81,480,832))
    print(len(dataset))
    for i in range(10):
        data = dataset[i+np.random.randint(len(dataset))]
        print(data['pixel_values'].shape)
    
        save_path = f'data/examples/doom/doom_video_{i}.mp4'
        save_tensor_to_video(data['pixel_values'],save_path)


if __name__ == "__main__":
    main()
    #test_fps_dataset()
    #test_thuman_dataset()
    #test_thuman_latent_dataset()
    #test_text_dataset()
    #test_prompt_emb()
    #make_data_examples()
    #test_ngr_latent()
    #test_ngr_dataset()
    #test_csgo_d2v_dataset()
    #test_doom_dataset()



