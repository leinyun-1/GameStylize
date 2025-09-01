import os 
import imageio
import sys 
import numpy as np
from tqdm import tqdm 

def cut_videos(src_video_path, output_dir):
    """
    将src_video_path每81帧切割为一个独立mp4片段，帧率为15fps，输出到output/cut_xxx.mp4。
    """
    os.makedirs(output_dir, exist_ok=True)
    segment_length = 81
    fps = 15

    reader = imageio.get_reader(src_video_path)
    meta = reader.get_meta_data()
    total_frames = meta.get('nframes', None)
    if total_frames == np.inf:
        # 兼容部分格式无法直接获取帧数
        total_frames = 0
        try:
            while True:
                reader.get_data(total_frames)
                total_frames += 1
        except IndexError:
            pass
    
    print(f"视频总帧数: {total_frames}")
    segment_idx = 0
    for start in range(0, total_frames, segment_length):
        end = min(start + segment_length, total_frames)
        frames = []
        for i in range(start, end):
            try:
                frame = reader.get_data(i)
                frames.append(frame)
            except Exception as e:
                print(f"读取第{i}帧出错: {e}")
                break
        if frames:
            output_path = os.path.join(output_dir, f"cut_{segment_idx:04d}.mp4")
            imageio.mimsave(output_path, frames, fps=fps)
            print(f"保存片段: {output_path}，帧数: {len(frames)}")
        segment_idx += 1
    reader.close()



def make_ngr_clip_dataset():
    root = '/mnt/aigc_cq/shared/game_videos/NGR_data/NGR_dataset_v2_v3_0710_dense_plot_sub2'
    dest = '/root/leinyu/data/ngr/ngr_clip'
    os.makedirs(dest, exist_ok=True)

    for folder in tqdm(os.listdir(root),desc='处理文件夹'):
        folder_path = os.path.join(root, folder)
        temp = os.listdir(folder_path)[0]
        video_path = os.path.join(folder_path, temp, 'result.mp4')
        print(f"处理视频: {video_path}")
        output_dir = os.path.join(dest, folder+'_'+temp)
        cut_videos(video_path, output_dir)


if __name__ == "__main__":
    make_ngr_clip_dataset()