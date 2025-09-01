import os
import random

import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as transforms_f
from torch.utils.data import Dataset

import h5py
import imageio

class CsgoVideoDataset(Dataset):
    def __init__(
            self,
            image_size=(128,256),
            crop_size=(128,256),
            data_root="/dockerdata/hdf5_dm_july2021",
            num_frames=64,
            num_prefix_frames=8,
            repeat_first_ratio=0.1,
    ):
        super().__init__()
        self.num_frames = num_frames
        self.num_prefix_frames = num_prefix_frames
        self.repeat_first_ratio = repeat_first_ratio
        self.crop_size = crop_size
        self.image_size = image_size
        self.data_paths = []
        for filename in os.listdir(data_root):
            if filename.endswith('hdf5'):
                if filename != "hdf5_dm_july2021_1080.hdf5":
                    self.data_paths.append(os.path.join(data_root, filename))

        self.img_transform = transforms.Compose([
            #transforms.CenterCrop(self.crop_size),
            transforms.Resize(image_size),
        ])

    def process_single_image(self, image, rand_state=None):
        image = transforms_f.to_tensor(image)
        image = self.augmentation(image, self.img_transform, rand_state)
        image = transforms_f.normalize(image, mean=[.5], std=[.5])
        return image

    @staticmethod
    def augmentation(image, transform, state=None):
        if state is not None:
            torch.set_rng_state(state)
        return transform(image)

    def __getitem__(self, index):
        while True:
            if not os.path.exists(self.data_paths[index]):
                index = index * random.randint(3, 1000) % len(self.data_paths)
                continue
            name = self.data_paths[index].split('/')[-1].split('.')[0]
            data_info = h5py.File(self.data_paths[index])
            total_frames = len(data_info.keys()) // 4
            st_frame = random.randint(0, total_frames - self.num_frames)
            pixel_values = []
            game_actions = []
            use_repeat_prefix = random.random() < self.repeat_first_ratio
            for i in range(st_frame, st_frame+self.num_prefix_frames):
                if use_repeat_prefix:
                    pixel_values.append(self.process_single_image(np.array(data_info[f"frame_{st_frame+self.num_prefix_frames-1}_x"])).flip(0))
                else:
                    pixel_values.append(self.process_single_image(np.array(data_info[f"frame_{i}_x"])).flip(0)) # bgr to rgb
                game_actions.append(np.zeros_like(np.array(data_info[f"frame_{i}_y"])))
                
            for i in range(st_frame+self.num_prefix_frames, st_frame+self.num_frames):
                pixel_values.append(self.process_single_image(np.array(data_info[f"frame_{i}_x"])).flip(0)) # bgr to rgb
                # 用上一帧的action来预测当前帧的图像？？？？，但是也有当前的鼠标位置，所以这个不确定
                game_actions.append(np.array(data_info[f"frame_{i}_y"]))
            game_actions = np.stack(game_actions) # nf, 50 (0~1)
            pixel_values = torch.stack(pixel_values, dim=1) # c, nf, h, w (-1~1)
            first_frame = (pixel_values[:,0].permute(1,2,0) * 0.5 + 0.5)*255 # h w 3 

            return {
                "pixel_values": pixel_values,
                "game_actions": game_actions,
                "first_frame": first_frame,
                "name": name,
            }            
            

    def __len__(self):
        return len(self.data_paths)


class CsgoD2VDataset(Dataset):
    """
    从root目录下读取depth和rgb视频对的数据集类
    root目录结构：
    root/
    ├── depth/
    │   ├── video1.mp4
    │   ├── video2.mp4
    │   └── ...
    └── rgb/
        ├── video1.mp4
        ├── video2.mp4
        └── ...
    """
    
    def __init__(
        self,
        root: str = '/root/leinyu/data/csgo/hdf5_dm_july2021_d2v',  # 根目录
        image_size: tuple = (480, 832),
        num_frames: int = 81,
    ):
        super().__init__()
        self.root = root
        self.image_size = image_size
        self.num_frames = num_frames
        
        # 设置RGB和深度视频目录路径
        self.rgb_dir = os.path.join(root, "rgb")
        self.depth_dir = os.path.join(root, "depth")
        
        # 获取所有视频文件名（确保RGB和深度目录都存在对应的文件）
        self.video_names = []
        rgb_files = set(f for f in os.listdir(self.rgb_dir) if f.endswith('.mp4'))
        depth_files = set(f for f in os.listdir(self.depth_dir) if f.endswith('.mp4'))
        
        # 只保留两个目录都存在的视频文件
        common_files = rgb_files.intersection(depth_files)
        self.video_names = sorted(list(common_files))
        
        # 图像变换：调整大小
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(image_size),
        ])
        
    def _load_video(self, video_path: str) -> torch.Tensor:
        """
        使用imageio加载视频并返回张量格式
        Args:
            video_path: 视频文件路径
        Returns:
            视频张量: (T, C, H, W) 范围 [0, 1]
        """
        # 使用imageio读取视频
        reader = imageio.get_reader(video_path)
        
        # 收集所有帧
        frames = []
        for frame in reader:
            # 转换为PIL Image格式
            frame_pil = transforms_f.to_pil_image(frame)
            # 应用变换：转换为张量并调整大小
            frame_tensor = self.transform(frame_pil)
            frames.append(frame_tensor)
        
        reader.close()
        
        # 堆叠所有帧: (T, C, H, W)
        video = torch.stack(frames, dim=0)
        
        return video
    
    def _normalize_to_neg1_1(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        将张量从 [0, 1] 归一化到 [-1, 1]
        """
        return tensor * 2.0 - 1.0
    
    def __getitem__(self, index: int) -> dict:
        """
        获取指定索引的视频对
        Args:
            index: 视频索引
        Returns:
            dict: 包含 'pixel_values' 和 'depth_values' 的字典
        """
        video_name = self.video_names[index]
        
        # 构建RGB和深度视频路径
        rgb_path = os.path.join(self.rgb_dir, video_name)
        depth_path = os.path.join(self.depth_dir, video_name)
        
        # 加载RGB视频 (T, C, H, W)
        rgb_video = self._load_video(rgb_path)
        
        # 加载深度视频 (T, C, H, W)
        depth_video = self._load_video(depth_path)
        
        # 确保帧数正确
        if rgb_video.shape[0] != self.num_frames:
            # 如果帧数不匹配，进行插值或裁剪
            if rgb_video.shape[0] > self.num_frames:
                # 裁剪多余的帧
                rgb_video = rgb_video[:self.num_frames]
                depth_video = depth_video[:self.num_frames]
            else:
                # 重复最后一帧直到达到所需帧数
                last_rgb = rgb_video[-1:]
                last_depth = depth_video[-1:]
                while rgb_video.shape[0] < self.num_frames:
                    rgb_video = torch.cat([rgb_video, last_rgb], dim=0)
                    depth_video = torch.cat([depth_video, last_depth], dim=0)
        
        # 调整维度顺序: (T, C, H, W) -> (C, T, H, W)
        rgb_video = rgb_video.permute(1, 0, 2, 3)
        depth_video = depth_video.permute(1, 0, 2, 3)
        
        # 归一化到 [-1, 1]
        rgb_video = self._normalize_to_neg1_1(rgb_video)
        depth_video = self._normalize_to_neg1_1(depth_video)
        
        return {
            "pixel_values": rgb_video,      # (4, 81, H, W)
            "depth_values": depth_video,    # (1, 81, H, W)
            "name": video_name,             # 视频文件名
        }
    
    def __len__(self) -> int:
        """
        返回数据集大小
        """
        return len(self.video_names) 


class CsgoLatentDataset(Dataset):
    def __init__(self,root='/root/leinyu/data/csgo/hdf5_dm_july2021'):
        super().__init__()
        self.root = root
        self.subs = os.listdir(self.root)
    
    def __getitem__(self, index):
        sub = self.subs[index]
        sub_path = os.path.join(self.root,sub)
        data  = torch.load(sub_path,map_location='cpu',weights_only=True)
        return data 

    def __len__(self):
        return len(self.subs)



class CsgoImageFromVideoDataset(Dataset):
    def __init__(
            self,
            image_size=(512,512),
            data_root='/mnt/aigc_sh/shared/game_videos/csgo/CounterStrike_Deathmatch/hdf5_dm_july2021',
    ):
        super().__init__()
        self.image_size = image_size
        self.video_length = 1000
        self.data_paths = []
        for filename in os.listdir(data_root):
            if filename.endswith('hdf5'):
                if filename != "hdf5_dm_july2021_1080.hdf5":
                    self.data_paths.append(os.path.join(data_root, filename))

        self.img_transform = transforms.Compose([
            #transforms.CenterCrop(self.crop_size),
            transforms.Resize(image_size),
        ])

    def process_single_image(self, image, rand_state=None):
        image = transforms_f.to_tensor(image)
        image = self.augmentation(image, self.img_transform, rand_state)
        image = transforms_f.normalize(image, mean=[.5], std=[.5])
        return image

    @staticmethod
    def augmentation(image, transform, state=None):
        if state is not None:
            torch.set_rng_state(state)
        return transform(image)

    def __getitem__(self, index):
        image_id = index // self.video_length % len(self.data_paths)
        frame_id = index % self.video_length
        while True:
            if not os.path.exists(self.data_paths[image_id]):
                image_id = image_id * random.randint(3, 1000) % len(self.data_paths)
                continue
            data_info = h5py.File(self.data_paths[image_id])
            total_frames = len(data_info.keys()) // 4
            pixel_values = self.process_single_image(np.array(data_info[f"frame_{frame_id}_x"])).flip(0) # bgr to rgb
            return {
                "pixel_values": pixel_values,
            }            
            

    def __len__(self):
        return len(self.data_paths) * self.video_length



from PIL import Image
class CsgoImageDataset(Dataset):
    def __init__(
            self,
            image_size=(512,512),
            data_root='/dockerdata/csgo/images/',
    ):
        super().__init__()
        self.image_size = image_size
        self.data_paths = []
        for filename in os.listdir(data_root):
            if filename.endswith('.png'):
                self.data_paths.append(os.path.join(data_root, filename))

        self.img_transform = transforms.Compose([
            #transforms.CenterCrop(self.crop_size),
            transforms.Resize(image_size),
        ])

    def process_single_image(self, image, rand_state=None):
        image = transforms_f.to_tensor(image)
        image = self.augmentation(image, self.img_transform, rand_state)
        image = transforms_f.normalize(image, mean=[.5], std=[.5])
        return image

    @staticmethod
    def augmentation(image, transform, state=None):
        if state is not None:
            torch.set_rng_state(state)
        return transform(image)

    def __getitem__(self, index):
        image_id = index 
        
        while True:
            try:
                img = Image.open(self.data_paths[image_id])
                img = img.convert('RGB')  # 确保是RGB格式
                img = self.process_single_image(img) # bgr to rgb
                pixel_values = img
                return {
                    "pixel_values": pixel_values
                } 
            except Exception as e:
                print(f"Error loading image {self.data_paths[image_id]}: {e}")
                image_id = (image_id + 1) % len(self.data_paths)           
            

    def __len__(self):
        return len(self.data_paths[:6])

