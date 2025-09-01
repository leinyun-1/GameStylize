import os
import imageio
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as transforms_f
from torch.utils.data import Dataset
from PIL import Image
import glob

class NgrVideoClipDataset(Dataset):
    """读取视频片段数据集，返回处理后的帧数据
    
    Args:
        root (str): 数据集根目录，包含cut_0000.mp4等视频文件
        image_size (tuple): 目标尺寸 (height, width)
        num_frames (int): 每个视频片段的帧数，默认为81
    """
    def __init__(self, root='/root/leinyu/data/ngr/ngr_clip/', image_size=(480,832), num_frames=81):
        self.root = root
        self.image_size = image_size
        self.num_frames = num_frames
        self.video_files = sorted(glob.glob(os.path.join(self.root, "**", "cut_*.mp4"), recursive=True))
        
        # 定义图像转换流程
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 转换到[-1,1]
        ])

    def __len__(self):
        return len(self.video_files)
    
    def _load_and_process_video(self, video_path):
        """读取并处理单个视频文件
        
        Args:
            video_path (str): 视频文件路径
            
        Returns:
            list: 处理后的帧张量列表
        """
        reader = imageio.get_reader(video_path)
        frames = []
        
        try:
            # 读取指定数量的帧
            for fid in range(self.num_frames):
                frame = reader.get_data(fid)
                frame_tensor = self.transform(frame)
                frames.append(frame_tensor)
        finally:
            reader.close()
    
        # 如果视频帧数不足，用最后一帧填充
        while len(frames) < self.num_frames:
            frames.append(frames[-1])
            
        return frames

    def __getitem__(self, idx):
        """读取并处理单个视频片段
        
        Returns:
            torch.Tensor: 处理后的视频数据，形状为(3, num_frames, h, w)，值域[-1,1]
        """
        video_path = self.video_files[idx]
        
        # 读取视频
        frames = self._load_and_process_video(video_path)
            
        # 堆叠帧并调整维度顺序为 (C, T, H, W)
        video_tensor = torch.stack(frames, dim=1)  # (3, num_frames, h, w)
        first_frame = (frames[0].permute(1,2,0) * 0.5 + 0.5)*255 # h w 3 

        res = { "pixel_values" : video_tensor,
                "first_frame": first_frame,
                "name": ('_').join(self.video_files[idx].split('/')[-2:])}
        
        return res


class NgrD2VDataset(Dataset):
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
        root: str = '/root/leinyu/data/ngr/ngr_d2v',  # 根目录
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
        try:
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
        except Exception as e:
            print(f"Error loading video at index {index}: {e}. Skipping to the next video.")
            # 如果发生错误，尝试读取下一个视频
            if index + 1 < len(self.video_names):
                return self.__getitem__(index + 1)
            else:
                # 如果已经是最后一个视频，抛出异常
                raise IndexError("No more videos to load.")
    
    def __len__(self) -> int:
        """
        返回数据集大小
        """
        return len(self.video_names) 



class LatentDataset(Dataset):
    def __init__(self,root='/root/leinyu/data/ngr/ngr_clip_latent'):
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