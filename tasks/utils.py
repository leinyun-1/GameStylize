import imageio
import numpy as np
import torch
from PIL import Image

def load_video_as_tensor(video_path, resize=None, max_frames=None):
    reader = imageio.get_reader(video_path)
    frames = []
    for i, frame in enumerate(reader):
        if max_frames and i >= max_frames:
            break
        img = frame  # RGB, shape (H, W, 3)
        if resize:
            img = np.array(Image.fromarray(img).resize(resize))
        frames.append(img)

    video_np = np.stack(frames, axis=0)  # [F, H, W, 3]
    video_np = video_np.transpose(3, 0, 1, 2)  # [3, F, H, W]
    video_tensor = torch.from_numpy(video_np).float() / 255.0 * 2 - 1
    return video_tensor  # shape: [3, F, H, W]

def save_tensor_to_video(video,video_path):
    video = (video + 1) / 2  # 转换到 [0,1]
    video = (video * 255).clamp(0, 255).byte()  # 转换到 [0,255]

    # 2. 调整维度顺序为 (f, h, w, 3) 并转换为 numpy 数组
    video_np = video.permute(1, 2, 3, 0).cpu().numpy()  # f h w 3

    # 3. 写入视频文件
    with imageio.get_writer(video_path, fps=15) as writer:
        for frame in video_np:
            writer.append_data(frame)