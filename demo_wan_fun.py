import numpy as np 
import os 
import shutil 
import torch
import torch.nn as nn
from safetensors.torch import load_file


def test_wan_fun_1_3b(video_path=None,prompt=None,save_path=None):
    import torch
    from diffsynth import ModelManager, WanVideoPipeline, save_video, VideoData
    from PIL import Image

    # Load models
    model_manager = ModelManager(device="cpu")
    model_manager.load_models(
        [
            "c:/Users/test/Desktop/ckpt/Wan2.1-Fun-1.3B-Control/diffusion_pytorch_model.safetensors",
            "c:/Users/test/Desktop/ckpt/Wan2.1-Fun-1.3B-Control/models_t5_umt5-xxl-enc-bf16.pth",
            "c:/Users/test/Desktop/ckpt/Wan2.1-Fun-1.3B-Control/Wan2.1_VAE.pth",
            "c:/Users/test/Desktop/ckpt/Wan2.1-Fun-1.3B-Control/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
        ],
        torch_dtype=torch.bfloat16, # You can set `torch_dtype=torch.float8_e4m3fn` to enable FP8 quantization.
    )
    lora_path = "c:/Users/test/Desktop/ckpt/wan_fun_1.3b_i2n_lora_e2.ckpt/lora-epoch=02-step=002000.ckpt"
    model_manager.load_lora(lora_path, lora_alpha=1)
    pipe = WanVideoPipeline.from_model_manager(model_manager, torch_dtype=torch.bfloat16, device="cuda:0")
    pipe.enable_vram_management(num_persistent_param_in_dit=None)

    # Control-to-video
    #prompt="第一人称，战士在战斗，csgo风格。"
    #prompt = "一位女孩，双马尾发型，穿着白色衬衣、黄色领带、黑色吊带裤和黑色皮鞋，笔直站立，双手自然下垂"
    control_video = VideoData(video_path, height=832, width=832, length=41)
    video = pipe(
        prompt=prompt,
        negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
        num_inference_steps=30,sigma_shift=8,
        control_video=control_video, height=832, width=832, num_frames=41,
        seed=1, tiled=False
    )
    save_video(video, save_path, fps=15, quality=5)

def demo_wan_fun_1_3b():
    video_root = '/root/leinyu/code/skyreels_v2/result/eval/i2v_14b_lora_perspective'
    prompts_path = '/root/leinyu/code/skyreels_v2/assets/eval_examples/prompts_qwen.txt'
    dest_root = '/root/leinyu/code/skyreels_v2/result/eval/i2v_14b_lora_perspective_normal'
    os.makedirs(dest_root,exist_ok=True)
    def load_prompts(file_path: str) -> dict:
        '''
        将txt文件中每行按空格分成key和value
        
        Args:
            file_path (str): txt文件的路径
            
        Returns:
            dict: 包含所有键值对的字典
        '''
        if not os.path.exists(file_path):
            raise ValueError(f"文件不存在: {file_path}")
            
        if not file_path.endswith('.txt'):
            raise ValueError(f"文件必须是txt格式: {file_path}")
            
        prompts_dict = {}
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                # 去除每行首尾的空白字符
                line = line.strip()
                # 跳过空行
                if line:
                    # 按第一个空格分割
                    parts = line.split(' ', 1)
                    if len(parts) == 2:
                        key, value = parts
                        prompts_dict[key] = value.strip()
                    else:
                        print(f"警告: 行 '{line}' 格式不正确，已跳过")
                    
        return prompts_dict
    prompts = load_prompts(prompts_path)
    videos = os.listdir(video_root)
    for video in videos:
        video_path = os.path.join(video_root,video)
        prompt = prompts['_'.join(video.split('_')[:-6])]
        dest_path = os.path.join(dest_root,video)
        test_wan_fun_1_3b(video_path,prompt,dest_path)


if __name__ == "__main__":
    test_wan_fun_1_3b(video_path='E:/winshare_1/code/skyreels_v2/result/eval/i2v_1.3b_lora_ortho/zzh_front_376139513_2025-12-16_18-20-53_30_3.0_6.0.mp4', \
        prompt='穿着一件黑色羽绒服，黑色长裤和黑色运动鞋。站立姿态，双手自然下垂。 目光正视前方，面无表情。 带着黑框眼镜。',save_path='E:/winshare_1/code/skyreels_v2/result/eval/i2v_1.3b_lora_ortho/zzh_front_376139513_2025-12-16_18-20-53_30_3.0_6.0_normal.mp4')