### 数据集目录为 root，root下有文件夹 0000、0001、、、、0525 , 0000文件夹下有5个mp4文件，0001文件夹下有5个mp4文件，以此类推 
### 现在需要你写一个dataset，从root目录下读取视频片段，返回视频片段的帧数据（3 81 h w ），处理到 -1到1 值域。  加入resize功能，根据输入的hw尺寸对每帧
### 进行resize，hw在dataset的init中传入。

# FILEPATH: /root/leinyu/code/DiffSynth-Studio/tasks/thuman_dataset_81.py
import os
import glob
import numpy as np
import torch
import torchvision.transforms as T
import imageio
from torch.utils.data import Dataset


class VideoDataset(Dataset):
    """
    数据集里所有视频固定 81 帧。
    返回: [3, 81, H, W]，值域 [-1, 1]
    """

    def __init__(self, root: str="/root/leinyu/data/round/round_video_1", h: int=832, w: int=832):
        super().__init__()
        self.h, self.w = h, w

        # 收集所有 mp4 路径
        self.video_paths = sorted(glob.glob(os.path.join(root, "*", "*.mp4")))
        if len(self.video_paths) == 0:
            raise RuntimeError(f"未找到任何视频文件，请检查 {root}")

        # 仅做 resize + 归一化
        self.transform = T.Compose([
            T.Resize((h, w)),
            T.Lambda(lambda x: x / 127.5 - 1.0)
        ])

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        path = self.video_paths[idx]

        # imageio 一次性读 81 帧 (T, H, W, C) uint8
        frames_np = np.stack(imageio.mimread(path), axis=0)   # (81, H0, W0, 3)
        frames = torch.from_numpy(frames_np).permute(3, 0, 1, 2).float()  # (3, 81, H0, W0)
        frames = self.transform(frames)                       # (3, 81, H, W)
        first_frame = (frames[:,0].permute(1,2,0) * 0.5 + 0.5)*255 # h w 3

        res = {
            'pixel_values': frames,
            'first_frame': first_frame,
            'name': ('_').join(path.split('/')[-2:])
        }
        return res

class TextDataset(Dataset):
    """
    根目录 root 下有 0000、0001、...、0525 共 526 个文件夹，
    每个文件夹里有一个 .txt 文件，读取该文件并以字符串形式返回。
    """

    def __init__(self, root: str='/root/leinyu/data/round/round_video_1'):
        super().__init__()
        self.root = root

        # 收集所有 txt 文件路径
        self.txt_paths = sorted(glob.glob(os.path.join(root, "*", "*.txt")))
        if len(self.txt_paths) == 0:
            raise RuntimeError(f"未找到任何 txt 文件，请检查 {root}")

    def __len__(self):
        return len(self.txt_paths)

    def __getitem__(self, idx):
        txt_path = self.txt_paths[idx]
        with open(txt_path, "r", encoding="utf-8") as f:
            # 去掉首尾空白后，再把所有换行/制表等空白字符统一替换成单个空格，变成一行
            text = " ".join(f.read().strip().split())
        return {"text": text, "path": txt_path}

class VideoTextNormalDataset(Dataset):
    def __init__(self, root: str="/root/leinyu/data/thuman2.1/Thuman2.1_norm_render_1/video", h: int=832, w: int=832, interval: int=1):
        super().__init__()
        self.h, self.w = h, w
        self.frame_interval = interval

        # 收集所有 mp4 路径
        self.video_paths = sorted(glob.glob(os.path.join(root, "*", "*.mp4")))
        if len(self.video_paths) == 0:
            raise RuntimeError(f"未找到任何视频文件，请检查 {root}")

        # 仅做 resize + 归一化
        self.transform = T.Compose([
            T.Resize((h, w)),
            T.Lambda(lambda x: x / 127.5 - 1.0)
        ])

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        path = self.video_paths[idx]

        # imageio 一次性读 81 帧 (T, H, W, C) uint8
        frames_np = np.stack(imageio.mimread(path,memtest=False), axis=0)   # (81, H0, W0, 3) 这种读取方式存在文件最大体积限制
        frames_np = frames_np[::self.frame_interval]  # 按间隔取帧
        frames = torch.from_numpy(frames_np).permute(3, 0, 1, 2).float()  # (3, 81, H0, W0)
        frames = self.transform(frames)                       # (3, 81, H, W)
        first_frame = (frames[:,0].permute(1,2,0) * 0.5 + 0.5)*255 # h w 3

        normal_video_path = path.replace('video','video_normal')
        n_frames_np = np.stack(imageio.mimread(normal_video_path,memtest=False), axis=0)   # (81, H0, W0, 3)
        n_frames_np = n_frames_np[::self.frame_interval]  # 按间隔取帧
        n_frames = torch.from_numpy(n_frames_np).permute(3, 0, 1, 2).float()  # (3, 81, H0, W0)
        n_frames = self.transform(n_frames)  

        text_path = ('/').join(path.split('/')[:-1]) + '/caption_qwen_vl.txt'
        with open(text_path, "r", encoding="utf-8") as f:
            # 去掉首尾空白后，再把所有换行/制表等空白字符统一替换成单个空格，变成一行
            text = " ".join(f.read().strip().split())
        


        res = {
            'pixel_values': frames,
            'first_frame': first_frame,
            'normal_pixel_values': n_frames,
            'text': text,
            'name': ('_').join(path.split('/')[-2:])
        }
        return res


class LatentDataset(Dataset):
    def __init__(self,root='/root/leinyu/data/round/latent'):
        super().__init__()
        self.root = root
        self.subs = os.listdir(self.root)
        
    
    def __getitem__(self, index):
        sub = self.subs[index]
        sub_path = os.path.join(self.root,sub)
        data  = torch.load(sub_path,map_location='cpu',weights_only=True)

        sub = sub.split('_')[0]
        sub_path = os.path.join(self.root,'../round_video_1',sub,'caption_qwen_vltensors.pth')
        prompt_emb = torch.load(sub_path,map_location='cpu',weights_only=True)

        data.update(prompt_emb)
        return data 

    def __len__(self):
        return len(self.subs)
    