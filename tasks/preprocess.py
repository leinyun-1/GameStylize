import torch 
from controlnet_aux.processor import ZoeDetector, HEDdetector, LineartDetector  # HEDdetector 需根据实际包名调整
from diffusers import ControlNetModel
import imageio
import numpy as np 
from transformers import AutoModelForDepthEstimation, AutoImageProcessor
from ngr_dataset import NgrVideoClipDataset  # 你自己的数据集类
from csgo_dataset import CsgoVideoDataset
from fps_dataset import VideoClipDataset
from doom_dataset import DoomH5Dataset
import os 
from tqdm import tqdm
DATASET_MAP = {
    'csgo': CsgoVideoDataset,
    'doom': DoomH5Dataset,
    'pubg': VideoClipDataset,  
    'varolent': VideoClipDataset,
    'delta': VideoClipDataset,
    'ngr': NgrVideoClipDataset
}
DATASET_ROOT_MAP = {
    'csgo': '/dockerdata/hdf5_dm_july2021/hdf5_dm_july2021_tars',
    'doom': '/mnt/kaiwu-user-carlfang/doom_dataset/freedoom_random_128_u2_rs_fix_v6_1280_720_hdf5',
    'pubg': '/root/leinyu/data/fps/pubg/erangel',  
    'varolent': '/root/leinyu/data/fps/varolent',
    'delta': '/root/leinyu/data/fps/delta',
    'ngr': '/root/leinyu/data/ngr/ngr_clip'
    
}


def get_controlnet_and_detector(controlnet_type):
    if controlnet_type == "depth":
        detector = ZoeDetector.from_pretrained("/root/leinyu/model/Annotators")
        controlnet = ControlNetModel.from_pretrained("/root/leinyu/model/sd-controlnet-depth", torch_dtype=torch.float16)
    elif controlnet_type == "softedge":
        detector = HEDdetector.from_pretrained("/root/leinyu/model/Annotators")
        controlnet = ControlNetModel.from_pretrained("/root/leinyu/model/control_v11p_sd15_softedge", torch_dtype=torch.float16)
    elif controlnet_type == "lineart":
        detector = LineartDetector.from_pretrained("/root/leinyu/model/Annotators")
        controlnet = ControlNetModel.from_pretrained("/root/leinyu/model/control_v11p_sd15_lineart", torch_dtype=torch.float16)
    else:
        raise ValueError(f"Unknown controlnet_type: {controlnet_type}")
    return controlnet, detector

def turn_rgb_to_depth():
    import imageio
    import numpy as np 
    from PIL import Image

    controlnet, detector = get_controlnet_and_detector('depth')
    detector = detector.to('cuda')
    video_path = '/mnt/aigc_cq/private/leinyu/code/GameVideoControl/validation_data/doom/test/gt.mp4'
    reader = imageio.get_reader(video_path)
    num_frames = 81
    frames = []
    for i in range(num_frames):
        frame = reader.get_data(i)
        frames.append(Image.fromarray(frame))
    reader.close()

    conditioning_frames = []
    for frame in frames:
        conditioning_frames.append(detector(frame))

    save_path = 'data/examples/wan/vace_0722/doom_depth_zoe.mp4'
    writer = imageio.get_writer(save_path)
    for depth in conditioning_frames:
        writer.append_data(np.array(depth))
    writer.close()

    

def turn_rgb_to_softedge():
    pass

def turn_rgb_to_lineart():
    pass

def depth_anything_v2():
    from transformers import pipeline
    from PIL import Image
    import requests
    import imageio
    import numpy as np 

    # load pipe
    pipe = pipeline(task="depth-estimation", model="/root/leinyu/model/Depth-Anything-V2-Large-hf",device="cuda:1")

    # load image
    video_root = 'data/examples/demo'
    for video in os.listdir(video_root):
        if video.endswith('.mp4'):
            video_path = os.path.join(video_root, video)
            print(f"Processing video: {video_path}")
        else:
            continue
        reader = imageio.get_reader(video_path)
        num_frames = 810
        start_frame_id = 0
        frames = []
        for i in range(start_frame_id, start_frame_id + num_frames):
            frame = reader.get_data(i)
            frames.append(Image.fromarray(frame))
        reader.close()
        
        # inference
        depths = []
        for image in frames:
            depth = pipe(image)["depth"]    
            depths.append(depth)
        
        # save depth video
        save_path = video_path.replace('.mp4','_depth.mp4')
        writer = imageio.get_writer(save_path,fps=15)
        for depth in depths:
            writer.append_data(np.array(depth))
        writer.close()
    


def cut_video():
    import imageio 
    video_path = 'data/examples/wan/vace_0722/doom_depth_dav2_0+1.mp4' 
    reader = imageio.get_reader(video_path)
    sd_id = 0
    num_frames = 149
    writer = imageio.get_writer('data/examples/wan/vace_0722/doom_depth_dav2_0-149.mp4')
    for i in range(sd_id, sd_id + num_frames):
        frame = reader.get_data(i)
        writer.append_data(frame)
    reader.close()
    writer.close()


def move_csgo_video():
    dataset = CsgoVideoDataset(image_size=(480,832),num_frames=81,num_prefix_frames=0)
    for i in range(10):
        data = dataset[i+np.random.randint(len(dataset))]
        video = data['pixel_values'] # 3 f h w (-1,1)
        video = (video + 1) / 2  # 转换到 [0,1]
        video = (video * 255).clamp(0, 255).byte()  # 转换到 [0,255]

        # 2. 调整维度顺序为 (f, h, w, 3) 并转换为 numpy 数组
        video_np = video.permute(1, 2, 3, 0).cpu().numpy()  # f h w 3

        # 3. 写入视频文件
        output_path = f'data/examples/csgo/csgo_video_{i}.mp4'
        with imageio.get_writer(output_path, fps=15) as writer:
            for frame in video_np:
                writer.append_data(frame)


def make_depth_caption():
    from transformers import pipeline
    from PIL import Image
    import requests
    import imageio
    import numpy as np 
    import os 
    from tqdm import tqdm
    from csgo_dataset import CsgoVideoDataset
    from ngr_dataset import NgrVideoClipDataset

    pipe = pipeline(task="depth-estimation", model="/root/leinyu/model/Depth-Anything-V2-Large-hf",device="cuda:1")
    #dataset = CsgoVideoDataset(image_size=(480,832),num_frames=81,num_prefix_frames=0,data_root='/dockerdata/hdf5_dm_july2021/hdf5_dm_july2021_tars')
    dataset = NgrVideoClipDataset()
    dest_root = '/root/leinyu/data/ngr/ngr_d2v/'
    os.makedirs(dest_root+'depth', exist_ok=True)
    os.makedirs(dest_root+'rgb', exist_ok=True)


    for i in tqdm(range(len(dataset))):
        data = dataset[i]
        name = data['name']
        save_path = os.path.join(dest_root, 'rgb', name + '.mp4')
        if os.path.exists(save_path):
            print(f"跳过已存在的视频: {save_path}")
            continue
        video = data['pixel_values'] # c f h w (-1,1)
        video = video.permute(1, 2, 3, 0).cpu().numpy()  # f h w c
        video = (video + 1) / 2  # 转换到 [0,1]
        video = (video * 255).clip(0, 255).astype(np.uint8)
        depths = []
        # 将视频帧按每27帧打包成一个列表
        batch_size = 27
        for i in range(0, len(video), batch_size):
            batch_frames = video[i:i + batch_size]
            
            # 将帧转换为PIL.Image对象并打包成列表
            batch_images = [Image.fromarray(frame) for frame in batch_frames]
            
            # 将打包后的帧列表输入到pipe中
            batch_depths = pipe(batch_images)
            
            # 将输出深度图保存到depths列表中
            for depth in batch_depths:
                depths.append(depth["depth"])

        
        name = data['name']
        save_path = os.path.join(dest_root, 'depth', name + '.mp4')
        writer = imageio.get_writer(save_path,fps=15)
        for depth in depths:
            writer.append_data(np.array(depth))
        writer.close()

        save_path = os.path.join(dest_root, 'rgb', name + '.mp4')
        writer = imageio.get_writer(save_path,fps=15)
        for frame in video:
            writer.append_data(frame)
        writer.close()

def make_depth_caption_1():
    from transformers import pipeline
    from PIL import Image
    import imageio
    import numpy as np 
    import os
    from tqdm import tqdm
    from ngr_dataset import NgrVideoClipDataset

    # 初始化 pipeline，直接指定 batch_size
    pipe = pipeline(
        task="depth-estimation",
        model="/root/leinyu/model/Depth-Anything-V2-Large-hf",
        device="cuda:1",
        batch_size=81  # 尽量大一点，取决于显存
    )

    dataset = NgrVideoClipDataset()
    dest_root = '/root/leinyu/data/ngr/ngr_d2v/'
    os.makedirs(dest_root + 'depth', exist_ok=True)
    os.makedirs(dest_root + 'rgb', exist_ok=True)

    for idx in tqdm(range(len(dataset)), desc="Processing videos"):
        data = dataset[idx]
        name = data['name']

        save_rgb_path = os.path.join(dest_root, 'rgb', name + '.mp4')
        save_depth_path = os.path.join(dest_root, 'depth', name + '.mp4')

        # 跳过已存在的
        if os.path.exists(save_rgb_path) and os.path.exists(save_depth_path):
            continue

        
        # 获取视频帧并转到 [0, 255]
        video = data['pixel_values']  # c f h w
        video = (video.permute(1, 2, 3, 0).cpu().numpy() + 1) / 2
        video = (video * 255).clip(0, 255).astype(np.uint8)

        # 初始化视频写入器
        rgb_writer = imageio.get_writer(save_rgb_path, fps=15)
        depth_writer = imageio.get_writer(save_depth_path, fps=15)

        batch_size = 81
        num_frames = video.shape[0]

        for start in range(0, num_frames, batch_size):
            batch_frames = video[start:start + batch_size]
            batch_images = [Image.fromarray(f) for f in batch_frames]

            # 直接批推理
            depth_results = pipe(batch_images)

            for frame, depth_data in zip(batch_frames, depth_results):
                rgb_writer.append_data(frame)
                depth_writer.append_data(np.array(depth_data["depth"]))

        rgb_writer.close()
        depth_writer.close()


def get_src_dest(dataset_type='pubg'):
    dataset_class = DATASET_MAP.get(dataset_type)
    root = DATASET_ROOT_MAP.get(dataset_type)
    dataset = dataset_class(root=root)



def make_depth_caption_fast():
    device = "cuda:1"
    
    # 加载模型和预处理器
    model_path = "/root/leinyu/model/Depth-Anything-V2-Large-hf"
    model = AutoModelForDepthEstimation.from_pretrained(model_path).to(device)
    processor = AutoImageProcessor.from_pretrained(model_path)


    dataset = VideoClipDataset(root='/root/leinyu/data/fps/delta')
    dest_root = '/root/leinyu/data/fps/delta_d2v/'
    os.makedirs(dest_root + 'depth', exist_ok=True)
    os.makedirs(dest_root + 'rgb', exist_ok=True)

    batch_size = 16

    for idx in tqdm(range(len(dataset)), desc="Videos"):
        data = dataset[idx]
        name = data['name']

        rgb_path = os.path.join(dest_root, 'rgb', name + '.mp4')
        if os.path.exists(rgb_path):
            print(f"跳过已存在的视频: {rgb_path}")
            continue

        # [c, f, h, w] -> [f, h, w, c]
        video = data['pixel_values'].permute(1, 2, 3, 0).cpu().numpy()
        video = ((video + 1) / 2 * 255).clip(0, 255).astype(np.uint8)

        depths_all = []

        # 按 batch_size 批推理
        for i in range(0, len(video), batch_size):
            frames_np = video[i:i + batch_size]  # [B, H, W, 3]
            
            # 直接批量送入 processor
            inputs = processor(images=list(frames_np), return_tensors="pt").to(device)
            
            with torch.no_grad():
                outputs = model(**inputs)
                depth_preds = outputs.predicted_depth  # [B, H, W]
            
            # 后处理到 uint8
            depth_preds = depth_preds.cpu().numpy()
            for depth in depth_preds:
                # 归一化到 0~255
                d_min, d_max = depth.min(), depth.max()
                depth_norm = (depth - d_min) / (d_max - d_min + 1e-8)
                depth_img = (depth_norm * 255).astype(np.uint8)
                depths_all.append(depth_img)

        # 保存深度视频
        depth_path = os.path.join(dest_root, 'depth', name + '.mp4')
        with imageio.get_writer(depth_path, fps=15) as writer:
            for depth_img in depths_all:
                writer.append_data(depth_img)

        # 保存原视频
        with imageio.get_writer(rgb_path, fps=15) as writer:
            for frame in video:
                writer.append_data(frame)

if __name__ == "__main__":
    #depth_anything_v2()
    #turn_rgb_to_depth() 
    #cut_video()
    #move_csgo_video()
    #make_depth_caption()
    #make_depth_caption_1()
    make_depth_caption_fast()

