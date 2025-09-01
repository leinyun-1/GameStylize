from train_wan_t2v_ymzx import TextVideoYmzxDataset,YmzxTensorDataset
import numpy as np 
import os 
import shutil 
import torch
import torch.nn as nn
from safetensors.torch import load_file
from modelscope import snapshot_download, dataset_snapshot_download
import cv2
import torch
from diffsynth import ModelManager, WanVideoPipelineNonunipc, WanVideoPipeline, save_video, VideoData
from diffsynth.pipelines.wan_video import model_fn_wan_video,model_fn_wan_video_df
import sys 
import time 
from tqdm import tqdm 
from tasks.utils import * 
from model import load_lora

def test_wan1_3_inversion():
    device = 'cuda:0'
    device = torch.device(device)
    dtype = torch.bfloat16
    num_inference_steps = 50
    shift = 5 
    cfg = 5
    denoise_strength = 0.7
    input_video_path = 'data/examples/wan/vace_input/60180_614733430200000123_6_src.mp4'
    #input_video_path = '/mnt/aigc_cq/private/leinyu/code/DiffSynth-Studio/exp_out/infer_exp/others/video1.mp4'
    name = input_video_path.split('/')[-1].split('.')[0]
    #prompt = '第一人称视角，fps游戏风格，csgo游戏风格，在建筑物环境中'
    prompt = '一只小狗在奔跑，带着墨镜'
    save_path = f'data/examples/wan/inversion/{name}Source_strength{denoise_strength}.mp4'

    # Load models
    model_manager = ModelManager(device=device)
    model_manager.load_models(
        [
            "models/Wan-AI/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors",
            "models/Wan-AI/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth",
            "models/Wan-AI/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth",
        ],
        torch_dtype=torch.bfloat16, # You can set `torch_dtype=torch.float8_e4m3fn` to enable FP8 quantization.
    )
    #lora_path = "exp_output/train_exp/0721_wan_lora_csgo/lightning_lora_ckpts/lora-epoch=28-step=001500.ckpt"
    #model_manager.load_lora(lora_path, lora_alpha=1)
    pipe = WanVideoPipelineNonunipc.from_model_manager(model_manager, torch_dtype=torch.bfloat16, device=device)
    #pipe.enable_vram_management(num_persistent_param_in_dit=None)
    pipe.scheduler.set_timesteps(num_inference_steps, shift=shift,denoising_strength=1)

    
    pipe.load_models_to_device(['vae'])

    
    input_video_tensor = load_video_as_tensor(input_video_path,resize=(832,480)).to(device,dtype) # 3 f h w 
    input_latent_tensor = pipe.encode_video(input_video_tensor.unsqueeze(0)) # b c f//4 h w
    print(input_latent_tensor.shape)



    noise = torch.randn_like(input_latent_tensor)
    #latents = pipe.scheduler.add_noise(input_latent_tensor, noise, pipe.scheduler.timesteps[0])
    latents = pipe.scheduler.add_noise(input_latent_tensor, noise, pipe.scheduler.timesteps[int(num_inference_steps*(1-denoise_strength))]) 
    print(latents.shape)

    pipe.load_models_to_device(["text_encoder"])
    neg_prompt = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
    prompt_ebd = pipe.encode_prompt(prompt)
    neg_prompt_ebd = pipe.encode_prompt(neg_prompt,positive=False)


    pipe.load_models_to_device(["dit"])
    denoise_timelist = pipe.scheduler.timesteps[int(num_inference_steps*(1-denoise_strength)):]
    pipe.dit.requires_grad_(False)
    with torch.no_grad():
        for timestep in tqdm(denoise_timelist):
            #timestep = timestep.unsqueeze(0).repeat(latents.shape[0]).to(device,dtype)
            timestep = torch.stack([timestep] * latents.shape[0], dim=0).to(dtype=dtype, device=device)

            cond_pred = pipe.dit(latents,timestep,**prompt_ebd,use_gradient_checkpointing=False)
            uncond_pred = pipe.dit(latents,timestep,**neg_prompt_ebd,use_gradient_checkpointing=False)
            pred = uncond_pred + cfg * (cond_pred - uncond_pred)

            latents = pipe.scheduler.step(pred, timestep[0], latents)
    
    
    pipe.load_models_to_device(['vae'])
    frames = pipe.decode_video(latents,tiled=False)
    pipe.load_models_to_device([])
    frames = pipe.tensor2video(frames[0])

    save_video(frames, save_path, fps=15, quality=5)
    

def test_wan1_3_inversion_1():
    from diffsynth import WanVideoPipelineNonunipc as WanVideoPipeline
    # Download models
    #snapshot_download("Wan-AI/Wan2.1-T2V-1.3B", local_dir="models/Wan-AI/Wan2.1-T2V-1.3B")
    device = 'cuda:0'
    device = torch.device(device)

    # Load models
    model_manager = ModelManager(device="cpu")
    model_manager.load_models(
        [
            "models/Wan-AI/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors",
            "models/Wan-AI/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth",
            "models/Wan-AI/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth",
        ],
        torch_dtype=torch.bfloat16, # You can set `torch_dtype=torch.float8_e4m3fn` to enable FP8 quantization.
    )
    lora_path = "exp_output/train_exp/0721_wan_lora_csgo/lightning_lora_ckpts/lora-epoch=28-step=001500.ckpt"
    model_manager.load_lora(lora_path, lora_alpha=1)
    pipe = WanVideoPipeline.from_model_manager(model_manager, torch_dtype=torch.bfloat16, device=device)
    pipe.enable_vram_management(num_persistent_param_in_dit=None)

    # Text-to-video
    # s_t = time.time()
    # video = pipe(
    #     prompt="纪实摄影风格画面，一只活泼的小狗在绿茵茵的草地上迅速奔跑。小狗毛色棕黄，两只耳朵立起，神情专注而欢快。阳光洒在它身上，使得毛发看上去格外柔软而闪亮。背景是一片开阔的草地，偶尔点缀着几朵野花，远处隐约可见蓝天和几片白云。透视感鲜明，捕捉小狗奔跑时的动感和四周草地的生机。中景侧面移动视角。",
    #     negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
    #     num_inference_steps=50,
    #     seed=0, tiled=True
    # )
    # print("text2video time:", time.time() - s_t)
    # save_video(video, "video1.mp4", fps=15, quality=5)

    # Video-to-video
    #video = VideoData("/mnt/aigc_cq/private/leinyu/code/DiffSynth-Studio/exp_out/infer_exp/others/video1.mp4", height=480, width=832)
    video = VideoData("data/examples/wan/vace_input/60180_614733430200000123_6_src.mp4", height=480, width=832)
    video = pipe(
        prompt="一只小狗在奔跑，戴着墨镜, csgo style",
        negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
        input_video=video, denoising_strength=0.9,
        num_inference_steps=50,
        seed=1, tiled=False
    )
    save_video(video, "data/examples/wan/inversion/ymSource_csgoLora&Prompt_strength0.9.mp4", fps=15, quality=5)


def test_wan1_3_varyT():
    device = 'cuda:0'
    device = torch.device(device)
    dtype = torch.bfloat16
    num_inference_steps = 30
    shift = 8
    cfg = 5
    zero_t_length = 1
    history_noise_level_id = 0
    num_frames = 29

    # Load models
    model_manager = ModelManager(device=device)
    model_manager.load_models(
        [
            "/dockerdata/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors",
            "/dockerdata/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth",
            "/dockerdata/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth",
        ],
        torch_dtype=torch.bfloat16, # You can set `torch_dtype=torch.float8_e4m3fn` to enable FP8 quantization.
    )
    lora_path = "/mnt/aigc_cq/private/leinyu/code/DiffSynth-Studio/exp_out/train_exp/0810_wan_lora_doom_ll8_hg1/lightning_lora_ckpts/lora-epoch=07-step=002000.ckpt"
    #lora_path_1 = "/mnt/aigc_cq/private/leinyu/code/DiffSynth-Studio/exp_out/train_exp/0728_wan_lora_csgo_df/lightning_lora_ckpts/lora-epoch=18-step=001000.ckpt"
    model_manager.load_lora(lora_path, lora_alpha=1)
    #model_manager.load_lora(lora_path_1, lora_alpha=1)
    pipe = WanVideoPipelineNonunipc.from_model_manager(model_manager, torch_dtype=torch.bfloat16, device=device)
    pipe.scheduler.set_timesteps(num_inference_steps, shift=shift,denoising_strength=1)

    #### 
    for input_video_path in os.listdir('data/examples/doom'):
        input_video_path = f'data/examples/doom/{input_video_path}'
        name = input_video_path.split('/')[-1].split('.')[0]
        prompt = '第一人称视角，csgo游戏风格'
        save_path = f'/root/leinyu/code/results/wan/varyT/doomHgLora/0810_ll8_hg1_t2000/{name}_T0*{zero_t_length}_csgoHgLora_{prompt}_noisy{history_noise_level_id}History.mp4'
        eval(pipe,input_video_path,zero_t_length,prompt,cfg,save_path,device,dtype,num_frames)
    
def eval(pipe,input_video_path,zero_t_length,prompt,cfg,save_path,device,dtype,num_frames):
    pipe.load_models_to_device(['vae'])
    input_video_tensor = load_video_as_tensor(input_video_path,resize=(832,480)).to(device,dtype) # 3 f h w 
    input_latent_tensor = pipe.encode_video(input_video_tensor[:,:num_frames].unsqueeze(0)) # b c f//4 h w
    latents = input_latent_tensor
    print(input_latent_tensor.shape)


    #history_noise_level = pipe.scheduler.timesteps[history_noise_level_id] # 0
    history_noise_level = torch.tensor(0,dtype=dtype,device=device)
    noise = torch.randn_like(input_latent_tensor)
    latents[:,:,zero_t_length:] = pipe.scheduler.add_noise(input_latent_tensor[:,:,zero_t_length:], noise[:,:,zero_t_length:], pipe.scheduler.timesteps[0]) 
    clean_history = latents[:,:,:zero_t_length].clone()
    #latents[:,:,:zero_t_length] = pipe.scheduler.add_noise(input_latent_tensor[:,:,:zero_t_length], noise[:,:,:zero_t_length], history_noise_level) # 历史帧加入一定程度的噪声
    print(latents.shape)

    pipe.load_models_to_device(["text_encoder"])
    neg_prompt = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
    prompt_ebd = pipe.encode_prompt(prompt)
    neg_prompt_ebd = pipe.encode_prompt(neg_prompt,positive=False)


    pipe.load_models_to_device(["dit"])
    denoise_timelist = pipe.scheduler.timesteps
    pipe.dit.requires_grad_(False)
    with torch.no_grad():
        for timestep in tqdm(denoise_timelist):
            #timestep = timestep.unsqueeze(0).repeat(latents.shape[0]).to(device,dtype)
            timestep = torch.stack([timestep] * latents.shape[0], dim=0).to(dtype=dtype, device=device)
            input_timestep = timestep 
            # timestep = timestep.unsqueeze(1).repeat(1,21-zero_t_length)
            # input_timestep = torch.cat([torch.ones(timestep.shape[0],zero_t_length,dtype=dtype,device=device) * history_noise_level,timestep],dim=1)

            cond_pred = pipe.dit(latents,input_timestep,**prompt_ebd,use_gradient_checkpointing=True)
            uncond_pred = pipe.dit(latents,input_timestep,**neg_prompt_ebd,use_gradient_checkpointing=True)
            pred = uncond_pred + cfg * (cond_pred - uncond_pred)

            latents[:,:,zero_t_length:] = pipe.scheduler.step(pred[:,:,zero_t_length:], timestep[0] if timestep.dim()==1 else timestep[0,0], latents[:,:,zero_t_length:])
            #latents = pipe.scheduler.step(pred, timestep[0] if timestep.dim()==1 else timestep[0,0], latents)
    
    latents[:,:,:zero_t_length] = clean_history 
    pipe.load_models_to_device(['vae'])
    frames = pipe.decode_video(latents,tiled=False)
    pipe.load_models_to_device([])
    frames = pipe.tensor2video(frames[0])

    save_video(frames, save_path, fps=15, quality=5)


def test_fuse_latent():
    device = 'cuda:0'
    device = torch.device(device)
    dtype = torch.bfloat16
    ref_depth_video_path = 'data/examples/wan/vace_0722/doom_depth_dav2.mp4'
    ref_depth_video_path_1 = 'data/examples/wan/vace_0722/doom_depth_dav2_68-149.mp4'


    # Load models
    model_manager = ModelManager(device=device)
    model_manager.load_models(
        [
            "models/iic/VACE-Wan2.1-1.3B-Preview/diffusion_pytorch_model.safetensors",
            "models/iic/VACE-Wan2.1-1.3B-Preview/models_t5_umt5-xxl-enc-bf16.pth",
            "models/iic/VACE-Wan2.1-1.3B-Preview/Wan2.1_VAE.pth",
        ],
        torch_dtype=torch.bfloat16, # You can set `torch_dtype=torch.float8_e4m3fn` to enable FP8 quantization.
    )
    lora_path = "exp_output/train_exp/0721_wan_lora_csgo/lightning_lora_ckpts/lora-epoch=28-step=001500.ckpt"
    model_manager.load_lora(lora_path, lora_alpha=1)
    pipe = WanVideoPipelineNonunipc.from_model_manager(model_manager, torch_dtype=torch.bfloat16, device=device)
    

    control_video = VideoData(ref_depth_video_path,height=480,width=832)
    video,latents = pipe(
        prompt="第一人称视角，fps游戏风格，csgo游戏风格，在建筑物环境中",
        negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
        num_inference_steps=50,
        height=480, width=832, num_frames=81,
        vace_video=control_video,
        vace_video_mask=None,
        seed=1, tiled=False
    )


    control_video_1 = VideoData(ref_depth_video_path_1,height=480,width=832)
    video_1,latents_1 = pipe(
        prompt="第一人称视角，fps游戏风格，csgo游戏风格，在建筑物环境中",
        negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
        num_inference_steps=50,
        height=480, width=832, num_frames=81,
        vace_video=control_video_1,
        vace_video_mask=None,
        seed=1, tiled=False
    )

    fusion_latents = latents[:,:,-3:] * 0.5  + latents_1[:,:,1:4] * 0.5
    all_latents = torch.cat([latents[:,:,:-3],fusion_latents,latents_1[:,:,4:]], dim=2)

    pipe.load_models_to_device(['vae'])
    frames = pipe.decode_video(all_latents,tiled=False)
    pipe.load_models_to_device([])
    frames = pipe.tensor2video(frames[0])

    save_video(frames, 'data/examples/wan/vace_0722/doomDAV2+csgoLora+latentsFusion.mp4', fps=15, quality=5)


    save_video(video_1, 'data/examples/wan/vace_0722/doomDAV2+csgoLora_68-149.mp4', fps=15, quality=5)

   
def test_fuse_latent_each_step():
    device = 'cuda:0'
    device = torch.device(device)
    dtype = torch.bfloat16
    ref_depth_video_path = 'data/examples/wan/vace_0722/doom_depth_dav2_0+1.mp4'
    num_inference_steps = 50
    denoising_strength = 1 
    sigma_shift = 5
    cfg_scale = 5
    num_frames = 161
    height = 480
    width = 832
    seed = 0
    rand_device = 'cpu'
    prompt = '第一人称视角，fps游戏风格，csgo游戏风格，在建筑物环境中'
    negative_prompt = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
    vace_scale = 1
    stride = 10
    window_size = 21
    overlap_size = window_size - stride


    control_video = VideoData(ref_depth_video_path,height=480,width=832,length=num_frames) # 输入深度视频帧数应大于 目标重绘帧数

    # Load models
    model_manager = ModelManager(device=device)
    model_manager.load_models(
        [
            "/dockerdata/VACE-Wan2.1-1.3B-Preview/diffusion_pytorch_model.safetensors",
            "/dockerdata/VACE-Wan2.1-1.3B-Preview/models_t5_umt5-xxl-enc-bf16.pth",
            "/dockerdata/VACE-Wan2.1-1.3B-Preview/Wan2.1_VAE.pth",
        ],
        torch_dtype=torch.bfloat16, # You can set `torch_dtype=torch.float8_e4m3fn` to enable FP8 quantization.
    )
    lora_path = "exp_output/train_exp/0721_wan_lora_csgo/lightning_lora_ckpts/lora-epoch=28-step=001500.ckpt"
    model_manager.load_lora(lora_path, lora_alpha=1)
    pipe = WanVideoPipelineNonunipc.from_model_manager(model_manager, torch_dtype=torch.bfloat16, device=device)
    pipe.scheduler.set_timesteps(num_inference_steps, denoising_strength=denoising_strength, shift=sigma_shift)

    # Initialize noise
    with torch.no_grad():
        noise = pipe.generate_noise((1, 16, (num_frames - 1) // 4 + 1, height//8, width//8), seed=seed, device=device, dtype=dtype)
        latents = noise
        
        # Encode prompts
        pipe.load_models_to_device(["text_encoder"])
        prompt_emb_posi = pipe.encode_prompt(prompt, positive=True)
        if cfg_scale != 1.0:
            prompt_emb_nega = pipe.encode_prompt(negative_prompt, positive=False)

        
        # VACE
        latents, vace_kwargs = pipe.prepare_vace_kwargs(
            latents, control_video,
            height=height, width=width, num_frames=num_frames, seed=seed, rand_device=rand_device, tiled=False
        ) # latents 的 t维度 + 1 , 为了主路x和支路x’在t维度上相等，c维度可以用mlp处理x’来对齐
        print(latents.shape,vace_kwargs["vace_context"].shape) # b 16 t h w , b 96 t h w 

        num_latent_frames = latents.shape[2]    
        assert num_latent_frames == (num_frames - 1) // 4 + 1
        #assert num_latent_frames%stride == overlap_size


        # Denoise
        pipe.load_models_to_device(["dit", "vace"])
        for progress_id, timestep in enumerate(tqdm(pipe.scheduler.timesteps)):
            timestep = timestep.unsqueeze(0).to(dtype=pipe.torch_dtype, device=pipe.device)
            #timestep = torch.stack([timestep] * latents.shape[0], dim=0).to(dtype=pipe.torch_dtype, device=pipe.device)  # (b,)


            for i in range(0,num_latent_frames-window_size+1,stride):
                current_latents = latents[:,:,i:i+window_size]
                if i > 0:
                    current_latents[:,:,:overlap_size] = current_latents[:,:,:overlap_size] * 0.5 + last_window_latents[:,:,-overlap_size:] * 0.5
                current_vace_kwargs = {"vace_scale": vace_scale}
                current_vace_kwargs["vace_context"] =  vace_kwargs["vace_context"][:,:,i:i+window_size]

                noise_pred_posi = model_fn_wan_video(
                    pipe.dit, motion_controller=pipe.motion_controller, vace=pipe.vace,
                    x=current_latents, timestep=timestep,
                    **prompt_emb_posi, **current_vace_kwargs,
                )
                if cfg_scale != 1.0:
                    noise_pred_nega = model_fn_wan_video(
                        pipe.dit, motion_controller=pipe.motion_controller, vace=pipe.vace,
                        x=current_latents, timestep=timestep,
                        **prompt_emb_nega, **current_vace_kwargs,
                    )
                    noise_pred = noise_pred_nega + cfg_scale * (noise_pred_posi - noise_pred_nega)
                else:
                    noise_pred = noise_pred_posi

                # Scheduler
                current_latents = pipe.scheduler.step(noise_pred, pipe.scheduler.timesteps[progress_id],current_latents)

                last_window_latents = latents[:,:,i:i+window_size].clone()
                latents[:,:,i+overlap_size:i+window_size] = current_latents[:,:,overlap_size:]
                if i > 0:
                    latents[:,:,i:i+overlap_size] = latents[:,:,i:i+overlap_size] * 0.5 + current_latents[:,:,:overlap_size] * 0.5
                else:
                    latents[:,:,i:i+overlap_size] = current_latents[:,:,:overlap_size]
                


    # Decode
    pipe.load_models_to_device(['vae'])
    frames = pipe.decode_video(latents,tiled=False)
    pipe.load_models_to_device([])
    frames = pipe.tensor2video(frames[0])

    save_video(frames, f"/root/leinyu/code/DiffSynth-Studio/data/examples/wan/vace_0722/doomDAV2+csgoLora+swin_stride{stride}_0728.mp4", fps=15, quality=5)


def test_fuse_latent_each_step_1(ref_depth_video_path = 'data/examples/wan/vace_0818/ft_local/demo_depth.mp4',lora_path = "exp_out/train_exp/0820_wan_lora_vace_csgo/lightning_lora_ckpts/lora-epoch=03-step=002500.ckpt"):
    device = 'cuda:0'
    device = torch.device(device)
    dtype = torch.bfloat16
    num_inference_steps = 4
    denoising_strength = 1 
    sigma_shift = 5
    cfg_scale = 1
    num_frames = 600
    height = 480
    width = 832
    seed = 0
    rand_device = 'cpu'
    prompt = '第一人称视角，fps游戏风格，csgo游戏风格，在建筑物环境中'
    negative_prompt = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
    vace_scale = 1
    sliding_window_stride = 10
    sliding_window_size = 21


    control_video = VideoData(ref_depth_video_path,height=480,width=832,length=num_frames) # 输入深度视频帧数应大于 目标重绘帧数

    # Load models
    model_manager = ModelManager(device=device)
    model_manager.load_models(
        [
            "models/iic/VACE-Wan2.1-1.3B-Preview/diffusion_pytorch_model.safetensors",
            "models/iic/VACE-Wan2.1-1.3B-Preview/models_t5_umt5-xxl-enc-bf16.pth",
            "models/iic/VACE-Wan2.1-1.3B-Preview/Wan2.1_VAE.pth",
        ],
        torch_dtype=torch.bfloat16, # You can set `torch_dtype=torch.float8_e4m3fn` to enable FP8 quantization.
    )
    #lora_path = "exp_output/train_exp/0721_wan_lora_csgo/lightning_lora_ckpts/lora-epoch=28-step=001500.ckpt"
    #lora_path = "/mnt/aigc_cq/private/leinyu/code/DiffSynth-Studio/exp_out/train_exp/0722_wan_lora_doom/lightning_lora_ckpts/lora-epoch=02-step=002000.ckpt"
    
    #lora_path = "/mnt/aigc_cq/private/leinyu/code/DiffSynth-Studio/exp_out/train_exp/0731_wan_lora_varolent_history_guide/lightning_lora_ckpts/lora-epoch=187-step=001500.ckpt"
    lora_path_1 = '/root/leinyu/model/Wan_lora/Wan21_CausVid_bidirect2_T2V_1_3B_lora_rank32.safetensors'
    model_manager.load_lora(lora_path, lora_alpha=1)
    pipe = WanVideoPipelineNonunipc.from_model_manager(model_manager, torch_dtype=torch.bfloat16, device=device)
    pipe.scheduler.set_timesteps(num_inference_steps, denoising_strength=denoising_strength, shift=sigma_shift)

    lora_ckpt = load_file(lora_path_1)
    from test import apply_lora_and_diff
    apply_lora_and_diff(pipe.dit,lora_ckpt,device=device, dtype=dtype)

    


    with torch.no_grad():
        noise = pipe.generate_noise((1, 16, (num_frames - 1) // 4 + 1, height//8, width//8), seed=seed, device=device, dtype=dtype)
        latents = noise
        
        # Encode prompts
        pipe.load_models_to_device(["text_encoder"])
        prompt_emb_posi = pipe.encode_prompt(prompt, positive=True)
        if cfg_scale != 1.0:
            prompt_emb_nega = pipe.encode_prompt(negative_prompt, positive=False)

        
        # VACE
        latents, vace_kwargs = pipe.prepare_vace_kwargs(
            latents, control_video,
            height=height, width=width, num_frames=num_frames, seed=seed, rand_device=rand_device, tiled=False
        ) # latents 的 t维度 + 1 , 为了主路x和支路x’在t维度上相等，c维度可以用mlp处理x’来对齐
        print(latents.shape,vace_kwargs["vace_context"].shape) # b 16 t h w , b 96 t h w 
        assert latents.shape[2] == vace_kwargs["vace_context"].shape[2] # t维度相等


        # Denoise
        pipe.load_models_to_device(["dit", "vace"])
        for progress_id, timestep in enumerate(tqdm(pipe.scheduler.timesteps)):
            timestep = timestep.unsqueeze(0).to(dtype=pipe.torch_dtype, device=pipe.device)
            noise_pred_posi = model_fn_wan_video(
                pipe.dit, motion_controller=pipe.motion_controller, vace=pipe.vace,
                x=latents, timestep=timestep,
                **prompt_emb_posi, **vace_kwargs, sliding_window_stride=sliding_window_stride, sliding_window_size=sliding_window_size
            )
            if cfg_scale != 1.0:
                noise_pred_nega = model_fn_wan_video(
                    pipe.dit, motion_controller=pipe.motion_controller, vace=pipe.vace,
                    x=latents, timestep=timestep,
                    **prompt_emb_nega, **vace_kwargs,sliding_window_stride=sliding_window_stride, sliding_window_size=sliding_window_size
                )
                noise_pred = noise_pred_nega + cfg_scale * (noise_pred_posi - noise_pred_nega)
            else:
                noise_pred = noise_pred_posi

            # Scheduler
            latents = pipe.scheduler.step(noise_pred, pipe.scheduler.timesteps[progress_id],latents)

    pipe.load_models_to_device(['vae'])
    frames = pipe.decode_video(latents,tiled=False)
    pipe.load_models_to_device([])
    frames = pipe.tensor2video(frames[0])

    save_video(frames, f"/root/leinyu/code/results/wan/diff_swin/{ref_depth_video_path.split('/')[-1]}DAV2+{lora_path.split('/')[2]}Lora+causLora+stride{sliding_window_stride}_frames{num_frames}.mp4", fps=15, quality=5)


def demo():
    depth_video_list =[i for i in  os.listdir('data/examples/demo') if i.endswith('_depth.mp4')]
    #depth_video_list = ['data/examples/wan/vace_0818/ft_local/demo_depth.mp4']
    # lora_paths = ["/mnt/aigc_cq/private/leinyu/code/DiffSynth-Studio/exp_out/train_exp/0531/checkpoints/epoch=0-step=400.ckpt",
    #               "/mnt/aigc_cq/private/leinyu/code/DiffSynth-Studio/exp_out/train_exp/0811_wan_lora_ngr/lightning_lora_ckpts/lora-epoch=03-step=002500.ckpt",
    #               "/mnt/aigc_cq/private/leinyu/code/DiffSynth-Studio/exp_out/train_exp/0730_wan_lora_pubg/lightning_lora_ckpts/lora-epoch=39-step=002000.ckpt",
    #               "/mnt/aigc_cq/private/leinyu/code/DiffSynth-Studio/exp_out/train_exp/0722_wan_lora_doom/lightning_lora_ckpts/lora-epoch=03-step=002500.ckpt",
    #               "/mnt/aigc_cq/private/leinyu/code/DiffSynth-Studio/exp_output/train_exp/0721_wan_lora_csgo/lightning_lora_ckpts/lora-epoch=37-step=002000.ckpt",
    #               "/mnt/aigc_cq/private/leinyu/code/DiffSynth-Studio/exp_out/train_exp/0731_wan_lora_varolent_history_guide/lightning_lora_ckpts/lora-epoch=187-step=001500.ckpt",
    #               "/mnt/aigc_cq/private/leinyu/code/DiffSynth-Studio/exp_out/train_exp/0804_wan_lora_delta_history_guide/lightning_lora_ckpts/lora-epoch=104-step=002500.ckpt"]
    
    lora_paths = ['exp_out/train_exp/0820_wan_lora_vace_ngr/lightning_lora_ckpts/lora-epoch=01-step=003000.ckpt']
    for depth_video in depth_video_list:
        for lora_path in lora_paths:
            test_fuse_latent_each_step_1(ref_depth_video_path = f'data/examples/demo/{depth_video}',lora_path = lora_path)
            #test_fuse_latent_each_step_1(ref_depth_video_path = depth_video,lora_path = lora_path)


def test_fuse_latent_each_step_base():
    device = 'cuda:0'
    device = torch.device(device)
    dtype = torch.bfloat16
    ref_depth_video_path = 'data/examples/wan/vace_0722/doom_depth_dav2_0+1.mp4'
    num_inference_steps = 50
    denoising_strength = 1 
    sigma_shift = 5
    cfg_scale = 5
    num_frames = 241
    height = 480
    width = 832
    seed = 0
    rand_device = 'cpu'
    prompt = '第一人称视角，fps游戏风格，csgo游戏风格，在建筑物环境中'
    negative_prompt = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
    vace_scale = 1
    stride = 10
    window_size = 21
    overlap_size = window_size - stride

    # Load models
    model_manager = ModelManager(device=device)
    model_manager.load_models(
        [
            "models/Wan-AI/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors",
            "models/Wan-AI/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth",
            "models/Wan-AI/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth",
        ],
        torch_dtype=torch.bfloat16, # You can set `torch_dtype=torch.float8_e4m3fn` to enable FP8 quantization.
    )
    lora_path = "exp_output/train_exp/0721_wan_lora_csgo/lightning_lora_ckpts/lora-epoch=28-step=001500.ckpt"
    model_manager.load_lora(lora_path, lora_alpha=1)
    pipe = WanVideoPipelineNonunipc.from_model_manager(model_manager, torch_dtype=torch.bfloat16, device=device)
    pipe.scheduler.set_timesteps(num_inference_steps, denoising_strength=denoising_strength, shift=sigma_shift)

    # Initialize noise
    with torch.no_grad():
        noise = pipe.generate_noise((1, 16, (num_frames - 1) // 4 + 1, height//8, width//8), seed=seed, device=device, dtype=dtype)
        latents = noise
        
        # Encode prompts
        pipe.load_models_to_device(["text_encoder"])
        prompt_emb_posi = pipe.encode_prompt(prompt, positive=True)
        if cfg_scale != 1.0:
            prompt_emb_nega = pipe.encode_prompt(negative_prompt, positive=False)


        num_latent_frames = latents.shape[2]    
        assert num_latent_frames == (num_frames - 1) // 4 + 1
        #assert num_latent_frames%stride == overlap_size


        # Denoise
        pipe.load_models_to_device(["dit"])
        for progress_id, timestep in enumerate(tqdm(pipe.scheduler.timesteps)):
            timestep = timestep.unsqueeze(0).to(dtype=pipe.torch_dtype, device=pipe.device)
            #timestep = torch.stack([timestep] * latents.shape[0], dim=0).to(dtype=pipe.torch_dtype, device=pipe.device)  # (b,)


            for i in range(0,num_latent_frames-window_size+1,stride):
                current_latents = latents[:,:,i:i+window_size]
                if i > 0:
                    current_latents[:,:,:overlap_size] = current_latents[:,:,:overlap_size] * 0 + last_window_latents[:,:,-overlap_size:] * 1


                noise_pred_posi = model_fn_wan_video(
                    pipe.dit, motion_controller=pipe.motion_controller, vace=pipe.vace,
                    x=current_latents, timestep=timestep,
                    **prompt_emb_posi,
                )
                if cfg_scale != 1.0:
                    noise_pred_nega = model_fn_wan_video(
                        pipe.dit, motion_controller=pipe.motion_controller, vace=pipe.vace,
                        x=current_latents, timestep=timestep,
                        **prompt_emb_nega,
                    )
                    noise_pred = noise_pred_nega + cfg_scale * (noise_pred_posi - noise_pred_nega)
                else:
                    noise_pred = noise_pred_posi

                # Scheduler
                current_latents = pipe.scheduler.step(noise_pred, pipe.scheduler.timesteps[progress_id],current_latents)

                last_window_latents = latents[:,:,i:i+window_size].clone()
                latents[:,:,i+overlap_size:i+window_size] = current_latents[:,:,overlap_size:]
                if i > 0:
                    latents[:,:,i:i+overlap_size] = latents[:,:,i:i+overlap_size] * 0.5 + current_latents[:,:,:overlap_size] * 0.5
                else:
                    latents[:,:,i:i+overlap_size] = current_latents[:,:,:overlap_size]
                


    # Decode
    pipe.load_models_to_device(['vae'])
    frames = pipe.decode_video(latents,tiled=False)
    pipe.load_models_to_device([])
    frames = pipe.tensor2video(frames[0])

    save_video(frames, f"/root/leinyu/code/results/wan/lora/csgo/swin_diffutoon_stride10all61_get1_0_put0.5_0.5.mp4", fps=15, quality=5)


def test_fuse_latent_each_step_base_1():
    device = 'cuda:1'
    device = torch.device(device)
    dtype = torch.bfloat16
    num_inference_steps = 4
    denoising_strength = 1 
    sigma_shift = 5
    cfg_scale = 1
    num_frames = 401
    height = 480
    width = 832
    seed = 0
    prompt = '第一人称视角，fps游戏风格，csgo游戏风格，在建筑物环境中'
    negative_prompt = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
    sliding_window_stride = 10
    sliding_window_size = 21


    # Load models
    model_manager = ModelManager(device=device)
    model_manager.load_models(
        [
            "/dockerdata/VACE-Wan2.1-1.3B-Preview/diffusion_pytorch_model.safetensors",
            "/dockerdata/VACE-Wan2.1-1.3B-Preview/models_t5_umt5-xxl-enc-bf16.pth",
            "/dockerdata/VACE-Wan2.1-1.3B-Preview/Wan2.1_VAE.pth",
        ],
        torch_dtype=torch.bfloat16, # You can set `torch_dtype=torch.float8_e4m3fn` to enable FP8 quantization.
    )
    lora_path = "exp_output/train_exp/0721_wan_lora_csgo/lightning_lora_ckpts/lora-epoch=28-step=001500.ckpt"
    lora_path_1 = '/root/leinyu/model/Wan_lora/Wan21_CausVid_bidirect2_T2V_1_3B_lora_rank32.safetensors'
    model_manager.load_lora(lora_path, lora_alpha=1)
    pipe = WanVideoPipelineNonunipc.from_model_manager(model_manager, torch_dtype=torch.bfloat16, device=device)
    pipe.scheduler.set_timesteps(num_inference_steps, denoising_strength=denoising_strength, shift=sigma_shift)

    lora_ckpt = load_file(lora_path_1)
    from test import apply_lora_and_diff
    apply_lora_and_diff(pipe.dit,lora_ckpt,device=device, dtype=dtype)


    with torch.no_grad():
        noise = pipe.generate_noise((1, 16, (num_frames - 1) // 4 + 1, height//8, width//8), seed=seed, device=device, dtype=dtype)
        latents = noise
        
        # Encode prompts
        pipe.load_models_to_device(["text_encoder"])
        prompt_emb_posi = pipe.encode_prompt(prompt, positive=True)
        if cfg_scale != 1.0:
            prompt_emb_nega = pipe.encode_prompt(negative_prompt, positive=False)

        # Denoise
        pipe.load_models_to_device(["dit", "vace"])
        for progress_id, timestep in enumerate(tqdm(pipe.scheduler.timesteps)):
            timestep = timestep.unsqueeze(0).to(dtype=pipe.torch_dtype, device=pipe.device)
            noise_pred_posi = model_fn_wan_video(
                pipe.dit, motion_controller=pipe.motion_controller, vace=pipe.vace,
                x=latents, timestep=timestep,
                **prompt_emb_posi, sliding_window_stride=sliding_window_stride, sliding_window_size=sliding_window_size
            )
            if cfg_scale != 1.0:
                noise_pred_nega = model_fn_wan_video(
                    pipe.dit, motion_controller=pipe.motion_controller, vace=pipe.vace,
                    x=latents, timestep=timestep,
                    **prompt_emb_nega,sliding_window_stride=sliding_window_stride, sliding_window_size=sliding_window_size
                )
                noise_pred = noise_pred_nega + cfg_scale * (noise_pred_posi - noise_pred_nega)
            else:
                noise_pred = noise_pred_posi

            # Scheduler
            latents = pipe.scheduler.step(noise_pred, pipe.scheduler.timesteps[progress_id],latents)

    pipe.load_models_to_device(['vae'])
    frames = pipe.decode_video(latents,tiled=False)
    pipe.load_models_to_device([])
    frames = pipe.tensor2video(frames[0])

    save_video(frames, f"/root/leinyu/code/results/wan/diff_swin/csgoLora+causLora+stride{sliding_window_stride}+frames{num_frames}_0805.mp4", fps=15, quality=5)


def test_video_dataset():
    video_path = 'data/examples/wan/vace_0722/doom_depth_dav2_0+1.mp4'
    control_video = VideoData(video_path,height=480,width=832,length=81) # 输入深度视频帧数应大于 目标重绘帧数
    control_video = control_video.raw_data()
    control_video = control_video[:20]
    print(len(control_video))
    id = 0
    for image in control_video:
        print(id)
        id += 1 
    frames = control_video.raw_data()
    print(len(frames))


def test_long_video_gen_history_guidance():
    device = 'cuda:1'
    device = torch.device(device)
    dtype = torch.bfloat16
    num_inference_steps = 30
    shift = 8
    cfg = 5
    history_t_length = 1
    history_noise_level_id = 0
    target_clip_num = 10
    latent_window_size = 8
    #input_video_path = 'data/examples/csgo/csgo_video_5.mp4'
    input_video_path = 'data/examples/wan/vace_input/delta.mp4'
    name = input_video_path.split('/')[-1].split('.')[0]
    prompt = '第一人称视角，csgo游戏风格'
    save_path = f'/root/leinyu/code/results/wan/history_guide/ll8_hg1_{history_t_length}_{target_clip_num}_{name}_t10000.mp4'

    # Load models
    model_manager = ModelManager(device=device)
    model_manager.load_models(
        [
            "/dockerdata/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors",
            "/dockerdata/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth",
            "/dockerdata/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth",
        ],
        torch_dtype=torch.bfloat16, # You can set `torch_dtype=torch.float8_e4m3fn` to enable FP8 quantization.
    )
    lora_path = "/mnt/aigc_cq/private/leinyu/code/DiffSynth-Studio/exp_out/train_exp/0810_wan_lora_doom_ll8_hg1/lightning_lora_ckpts/lora-epoch=36-step=010000.ckpt"
    model_manager.load_lora(lora_path, lora_alpha=1)
    pipe = WanVideoPipelineNonunipc.from_model_manager(model_manager, torch_dtype=torch.bfloat16, device=device)
    pipe.scheduler.set_timesteps(num_inference_steps, shift=shift,denoising_strength=1)
    #history_noise_level = pipe.scheduler.timesteps[history_noise_level_id] # 0
    history_noise_level = torch.tensor(0,device=device,dtype=dtype)

    pipe.load_models_to_device(["text_encoder"])
    neg_prompt = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
    prompt_ebd = pipe.encode_prompt(prompt)
    neg_prompt_ebd = pipe.encode_prompt(neg_prompt,positive=False)

    pipe.load_models_to_device(['vae'])
    input_video_tensor = load_video_as_tensor(input_video_path,resize=(832,480)).to(device,dtype) # 3 f h w 
    input_latent_tensor = pipe.encode_video(input_video_tensor[:,-(1 + 4*(history_t_length-1)):].unsqueeze(0)) # b c ht h w
    history_latents = input_latent_tensor
    print(history_latents.shape)
    prev_latents = pipe.generate_noise((1,16,latent_window_size,60,104),device=device,dtype=dtype)  
    

    pipe.load_models_to_device(["dit"])
    denoise_timelist = pipe.scheduler.timesteps
    pipe.dit.requires_grad_(False)


    all_latents = []
    all_frames = [] 
    with torch.no_grad():
        for i in tqdm(range(target_clip_num),desc='for clip'):
            latents = torch.randn_like(prev_latents)
            #latents[:,:,:history_t_length] = pipe.scheduler.add_noise(prev_latents[:,:,-history_t_length:],latents[:,:,:history_t_length],timestep=history_noise_level)
            latents[:,:,:history_t_length] = history_latents
            for timestep in tqdm(denoise_timelist,desc='denoising'):
                timestep = torch.stack([timestep] * latents.shape[0], dim=0).to(dtype=dtype, device=device)
                # timestep = timestep.unsqueeze(1).repeat(1,21-history_t_length)
                # input_timestep = torch.cat([torch.ones(timestep.shape[0],history_t_length,dtype=dtype,device=device) * history_noise_level,timestep],dim=1)
                input_timestep = timestep 
                cond_pred = model_fn_wan_video(pipe.dit, vace=pipe.vace,x=latents, timestep=input_timestep,**prompt_ebd)
                uncond_pred = model_fn_wan_video(pipe.dit, vace=pipe.vace,x=latents, timestep=input_timestep,**neg_prompt_ebd)
                pred = uncond_pred + cfg * (cond_pred - uncond_pred)

                latents[:,:,history_t_length:] = pipe.scheduler.step(pred[:,:,history_t_length:], timestep[0] if timestep.dim()==1 else timestep[0,0], latents[:,:,history_t_length:])
            if i > 0:
                all_latents.append(latents[:,:,history_t_length:])
            else:
                latents[:,:,:history_t_length] = history_latents
                all_latents.append(latents)
            prev_latents = latents
            pipe.load_models_to_device(['vae'])
            frames = pipe.decode_video(latents) # b 3 t h w 
            frames = frames[:,:,-(1 + 4*(history_t_length-1)):].to(device=device,dtype=dtype) # b 3 17 h w 
            history_latents = pipe.encode_video(frames,tiled=False) # b c history_latent_lenght h w  。这句报了一个非常奇怪的错，类似于cpu torch
            #history_latents = latents[:,:,-history_t_length:].clone() # b 16 history_t_length h w          

        
    
    all_latents = torch.cat(all_latents,dim=2)
    pipe.load_models_to_device(['vae'])
    frames = pipe.decode_video(all_latents,tiled=False)
    pipe.load_models_to_device([])
    frames = pipe.tensor2video(frames[0])

    save_video(frames, save_path, fps=15, quality=5)
   

def test_long_video_gen_history_guidance_vace():
    device = 'cuda:0'
    device = torch.device(device)
    dtype = torch.bfloat16
    num_inference_steps = 30
    shift = 8
    cfg = 5
    history_latent_length = 4
    latent_window_size = 21
    target_clip_num = 5
    history_noise_level = 0
    ref_depth_video_path = 'data/examples/wan/vace_input/delta_depth_dav2.mp4'
    name = ref_depth_video_path.split('/')[-1].split('.')[0]
    prompt = '战士在战斗，csgo游戏风格，在建筑物环境中'
    save_path = f'/root/leinyu/code/results/wan/vace/zero5t_{history_noise_level}hg_{history_latent_length}_{target_clip_num}_{name}_latentLen{latent_window_size}_vaceScale1_重新encodeHG.mp4'

    all_latent_len = target_clip_num * (latent_window_size - history_latent_length) + history_latent_length
    all_frames_len = (all_latent_len-1) * 4 - 1

    
    control_video = VideoData(ref_depth_video_path,height=480,width=832,length=all_frames_len).raw_data()

    # Load models
    model_manager = ModelManager(device=device)
    model_manager.load_models(
        [
            "models/iic/VACE-Wan2.1-1.3B-Preview/diffusion_pytorch_model.safetensors",
            "models/iic/VACE-Wan2.1-1.3B-Preview/models_t5_umt5-xxl-enc-bf16.pth",
            "models/iic/VACE-Wan2.1-1.3B-Preview/Wan2.1_VAE.pth",
        ],
        torch_dtype=torch.bfloat16, # You can set `torch_dtype=torch.float8_e4m3fn` to enable FP8 quantization.
    )
    lora_path = "/mnt/aigc_cq/private/leinyu/code/DiffSynth-Studio/exp_out/train_exp/0731_wan_lora_csgo_zero5t/lightning_lora_ckpts/lora-epoch=37-step=003000.ckpt"
    #lora_path_1 = "/mnt/aigc_cq/private/leinyu/code/DiffSynth-Studio/exp_out/train_exp/0728_wan_lora_csgo_df/lightning_lora_ckpts/lora-epoch=28-step=001500.ckpt"
    model_manager.load_lora(lora_path, lora_alpha=1)
    #model_manager.load_lora(lora_path_1, lora_alpha=1)
    pipe = WanVideoPipelineNonunipc.from_model_manager(model_manager, torch_dtype=torch.bfloat16, device=device)
    #pipe.enable_vram_management(num_persistent_param_in_dit=None)
    pipe.scheduler.set_timesteps(num_inference_steps, shift=shift,denoising_strength=1)
    history_noise_level = pipe.scheduler.timesteps[history_noise_level] # 0


    pipe.load_models_to_device(["text_encoder"])
    neg_prompt = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
    prompt_ebd = pipe.encode_prompt(prompt)
    neg_prompt_ebd = pipe.encode_prompt(neg_prompt,positive=False)


    pipe.load_models_to_device(["dit","vace"])
    denoise_timelist = pipe.scheduler.timesteps
    pipe.dit.requires_grad_(False)


    all_latents = pipe.generate_noise((1,16,all_latent_len,60,104),dtype=dtype,device=device)
    all_latents, all_vace_kwargs = pipe.prepare_vace_kwargs(
        all_latents, control_video,
        height=480, width=832, num_frames=81, tiled=False
            ) 
    window_latents  = pipe.generate_noise((1,16,latent_window_size - history_latent_length,60,104),dtype=dtype,device=device) # b 16 latent_window_size h w
    history_t_length = history_latent_length
    with torch.no_grad():
        for i in tqdm(range(target_clip_num),desc='for clip'):
            window_left = i * (latent_window_size - history_latent_length)
            window_right = window_left + latent_window_size
            latents = all_latents[:,:,window_left:window_right].clone() 
            latents[:,:,history_latent_length:] = window_latents.clone() # 噪声部分使用相同的噪声
            vace_kwargs = {"vace_scale": 1}
            vace_kwargs['vace_context'] = all_vace_kwargs['vace_context'][:,:,window_left:window_right].clone() # b 96 t h w 

            for timestep in tqdm(denoise_timelist,desc='denoising'):
                timestep = torch.stack([timestep] * latents.shape[0], dim=0).to(dtype=dtype, device=device)
                # timestep = timestep.unsqueeze(1).repeat(1,21-history_t_length)
                # input_timestep = torch.cat([torch.ones(timestep.shape[0],history_t_length,dtype=dtype,device=device) * history_noise_level,timestep],dim=1)
                input_timestep = timestep

                cond_pred = model_fn_wan_video(pipe.dit, vace=pipe.vace,x=latents, timestep=input_timestep,**prompt_ebd, **vace_kwargs)
                uncond_pred = model_fn_wan_video(pipe.dit, vace=pipe.vace,x=latents, timestep=input_timestep,**neg_prompt_ebd, **vace_kwargs)
                pred = uncond_pred + cfg * (cond_pred - uncond_pred)
                
                if i == 0:
                    history_t_length = 0    
                else:
                    history_t_length = history_latent_length
                latents[:,:,history_t_length:] = pipe.scheduler.step(pred[:,:,history_t_length:], timestep[0] if timestep.dim()==1 else timestep[0,0], latents[:,:,history_t_length:])
                #latents = pipe.scheduler.step(pred, timestep[0] if timestep.dim()==1 else timestep[0,0], latents)

            all_latents[:,:,window_left:window_right]=latents
            # prev_latents = latents
            # pipe.load_models_to_device(['vae'])
            # frames = pipe.decode_video(latents,tiled=False) # b 3 t h w 
            # frames = frames[:,:,-(1 + 4*(history_latent_length-1)):].to(device=device,dtype=dtype) # b 3 17 h w 
            # history_latents = pipe.encode_video(frames,tiled=False) # b c history_latent_lenght h w  

    
    #all_latents = torch.cat(all_latents,dim=2)
    pipe.load_models_to_device(['vae'])
    frames = pipe.decode_video(all_latents,tiled=False)
    pipe.load_models_to_device([])
    frames = pipe.tensor2video(frames[0])

    save_video(frames, save_path, fps=15, quality=8)
   

def test_long_video_gen_history_guidance_vace_frameSpace():
    device = 'cuda:0'
    device = torch.device(device)
    dtype = torch.bfloat16
    num_inference_steps = 30
    shift = 8
    cfg = 3

    history_latent_length = 1
    latent_window_size = 5
    target_clip_num = 10

    history_noise_level = -2
    input_video_path = 'data/examples/doom/doom_video_0.mp4'
    ref_depth_video_path = 'data/examples/wan/vace_input/delta_depth_dav2.mp4'
    name = ref_depth_video_path.split('/')[-1].split('.')[0]
    prompt = '战士在战斗，csgo游戏风格，在建筑物环境中'
    save_path = f'/root/leinyu/code/results/wan/vace/csgo_vace_ll5_hg1_{history_noise_level}hg_{history_latent_length}_{target_clip_num}_{name}_latentLen{latent_window_size}.mp4'

    all_latent_len = target_clip_num * (latent_window_size - history_latent_length) + history_latent_length
    all_frames_len = (all_latent_len-1) * 4 + 1
    frame_window_size = (latent_window_size - 1) * 4 + 1

    
    control_video = VideoData(ref_depth_video_path,height=480,width=832,length=810).raw_data()

    # Load models
    model_manager = ModelManager(device=device)
    model_manager.load_models(
        [
            "models/iic/VACE-Wan2.1-1.3B-Preview/diffusion_pytorch_model.safetensors",
            "models/iic/VACE-Wan2.1-1.3B-Preview/models_t5_umt5-xxl-enc-bf16.pth",
            "models/iic/VACE-Wan2.1-1.3B-Preview/Wan2.1_VAE.pth",
        ],
        torch_dtype=torch.bfloat16, # You can set `torch_dtype=torch.float8_e4m3fn` to enable FP8 quantization.
    )
    lora_path = "/mnt/aigc_cq/private/leinyu/code/DiffSynth-Studio/exp_out/train_exp/0815_wan_vace_lora_csgo_ll5_hg1/lightning_lora_ckpts/lora-epoch=10-step=004000.ckpt"
    #lora_path_1 = "/mnt/aigc_cq/private/leinyu/code/DiffSynth-Studio/exp_out/train_exp/0728_wan_lora_csgo_df/lightning_lora_ckpts/lora-epoch=28-step=001500.ckpt"
    model_manager.load_lora(lora_path, lora_alpha=1)
    #model_manager.load_lora(lora_path_1, lora_alpha=1)
    pipe = WanVideoPipelineNonunipc.from_model_manager(model_manager, torch_dtype=torch.bfloat16, device=device)
    #pipe.enable_vram_management(num_persistent_param_in_dit=None)
    pipe.scheduler.set_timesteps(num_inference_steps, shift=shift,denoising_strength=1)
    #history_noise_level = pipe.scheduler.timesteps[history_noise_level] # 0

    def split_lora(lora_ckpt):
        dit_lora = {}
        vace_lora = {} 
        for k,v in lora_ckpt.items():
            if k.startswith('blocks'):
                dit_lora[k] = v 
            elif k.startswith('vace_blocks'):
                vace_lora[k] = v 
        return dit_lora,vace_lora
    lora_state_dict = torch.load(lora_path,map_location='cpu')
    dit_lora,vace_lora = split_lora(lora_state_dict)
    load_lora(pipe.dit,dit_lora)
    load_lora(pipe.vace,vace_lora)

    pipe.load_models_to_device(["text_encoder"])
    neg_prompt = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
    prompt_ebd = pipe.encode_prompt(prompt)
    neg_prompt_ebd = pipe.encode_prompt(neg_prompt,positive=False)

    pipe.load_models_to_device(['vae'])
    input_video_tensor = load_video_as_tensor(input_video_path,resize=(832,480)).to(device,dtype) # 3 f h w 
    input_latent_tensor = pipe.encode_video(input_video_tensor[:,-(1 + 4*(history_latent_length-1)):].unsqueeze(0)) # b c ht h w
    history_latents = input_latent_tensor


    pipe.load_models_to_device(["dit","vace"])
    denoise_timelist = pipe.scheduler.timesteps
    pipe.dit.requires_grad_(False)

    all_latents = []
    all_frames = []
    window_latents  = pipe.generate_noise((1,16,latent_window_size,60,104),dtype=dtype,device=device) # b 16 latent_window_size h w
    history_t_length = history_latent_length
    frame_stride = frame_window_size - (history_latent_length - 1) * 4 - 1
    with torch.no_grad():
        for i in tqdm(range(target_clip_num),desc='for clip'):
            latents = window_latents.clone() # 噪声部分使用相同的噪声
            latents, vace_kwargs = pipe.prepare_vace_kwargs(
                latents, control_video[i*frame_stride:i*frame_stride+frame_window_size],
                height=480, width=832, num_frames=frame_window_size, tiled=False
                    ) 
            if i == 0:
                history_t_length = 0    
            else:
                history_t_length = history_latent_length
                if history_noise_level == 0:
                    latents[:,:,:history_t_length] = history_latents
                else:
                    latents[:,:,:history_t_length] = pipe.scheduler.add_noise(latents[:,:,:history_t_length], history_latents, timestep=pipe.scheduler.timesteps[history_noise_level])

            for timestep in tqdm(denoise_timelist,desc='denoising'):
                timestep = torch.stack([timestep] * latents.shape[0], dim=0).to(dtype=dtype, device=device)
                # timestep = timestep.unsqueeze(1).repeat(1,21-history_t_length)
                # input_timestep = torch.cat([torch.ones(timestep.shape[0],history_t_length,dtype=dtype,device=device) * history_noise_level,timestep],dim=1)
                input_timestep = timestep

                cond_pred = model_fn_wan_video(pipe.dit, vace=pipe.vace,x=latents, timestep=input_timestep,**prompt_ebd, **vace_kwargs)
                uncond_pred = model_fn_wan_video(pipe.dit, vace=pipe.vace,x=latents, timestep=input_timestep,**neg_prompt_ebd, **vace_kwargs)
                pred = uncond_pred + cfg * (cond_pred - uncond_pred)
                
                latents[:,:,history_t_length:] = pipe.scheduler.step(pred[:,:,history_t_length:], timestep[0] if timestep.dim()==1 else timestep[0,0], latents[:,:,history_t_length:])
                #latents = pipe.scheduler.step(pred, timestep[0] if timestep.dim()==1 else timestep[0,0], latents)

            all_latents.append(latents[:,:,history_t_length:])
            # prev_latents = latents
            pipe.load_models_to_device(['vae'])
            frames = pipe.decode_video(latents,tiled=False) # b 3 t h w 
            all_frames.append(frames[:,:,history_t_length:])
            frames = frames[:,:,-(1 + 4*(history_latent_length-1)):].to(device=device,dtype=dtype) # b 3 17 h w 
            history_latents = pipe.encode_video(frames,tiled=False) # b c history_latent_lenght h w  

    
    # all_latents = torch.cat(all_latents,dim=2)
    # pipe.load_models_to_device(['vae'])
    # frames = pipe.decode_video(all_latents,tiled=False)
    # all_frames.append(frames[:,:,history_t_length:])
    # pipe.load_models_to_device([])
    # frames = pipe.tensor2video(frames[0])
    frames = pipe.tensor2video(torch.cat(all_frames,dim=2)[0])

    save_video(frames, save_path, fps=15, quality=8)
  

def test_hg_vace():
    device = 'cuda:1'
    device = torch.device(device)
    dtype = torch.bfloat16
    num_inference_steps = 30
    shift = 8
    cfg = 5
    latent_window_size = 4
    history_latent_length = 2
    history_noise_level = 0
    ref_depth_video_path = 'data/examples/wan/vace_0722/doom_depth_dav2_0-149.mp4'
    input_video_path = 'data/examples/wan/vace_0723/doomdepth+cagoLorall4hg2+t2000+81.mp4'
    name = ref_depth_video_path.split('/')[-1].split('.')[0]
    prompt = '战士在战斗，csgo游戏风格，建筑物场景'
    save_path = f'/root/leinyu/code/results/wan/vace/ll4_hg2_{history_noise_level}hg_{history_latent_length}_1_{name}_latentLen{latent_window_size}.mp4'

    
    control_video = VideoData(ref_depth_video_path,height=480,width=832,length=81).raw_data()

    # Load models
    model_manager = ModelManager(device=device)
    model_manager.load_models(
        [
            "models/iic/VACE-Wan2.1-1.3B-Preview/diffusion_pytorch_model.safetensors",
            "models/iic/VACE-Wan2.1-1.3B-Preview/models_t5_umt5-xxl-enc-bf16.pth",
            "models/iic/VACE-Wan2.1-1.3B-Preview/Wan2.1_VAE.pth",
        ],
        torch_dtype=torch.bfloat16, # You can set `torch_dtype=torch.float8_e4m3fn` to enable FP8 quantization.
    )
    lora_path = "/mnt/aigc_cq/private/leinyu/code/DiffSynth-Studio/exp_out/train_exp/0805_wan_lora_csgo_ll4_hg2/lightning_lora_ckpts/lora-epoch=428-step=006000.ckpt"
    model_manager.load_lora(lora_path, lora_alpha=1)
    pipe = WanVideoPipelineNonunipc.from_model_manager(model_manager, torch_dtype=torch.bfloat16, device=device)
    pipe.scheduler.set_timesteps(num_inference_steps, shift=shift,denoising_strength=1)
    history_noise_level = pipe.scheduler.timesteps[history_noise_level] # 0

    pipe.load_models_to_device(["text_encoder"])
    neg_prompt = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
    prompt_ebd = pipe.encode_prompt(prompt)
    neg_prompt_ebd = pipe.encode_prompt(neg_prompt,positive=False)

    pipe.load_models_to_device(['vae'])
    input_video_tensor = load_video_as_tensor(input_video_path,resize=(832,480)).to(device,dtype) # 3 f h w 
    input_latent_tensor = pipe.encode_video(input_video_tensor[:,:(history_latent_length-1)*4+1].unsqueeze(0)) # b c ht h w
    history_latents = input_latent_tensor
    print(history_latents.shape)


    pipe.load_models_to_device(["dit","vace"])
    denoise_timelist = pipe.scheduler.timesteps
    pipe.dit.requires_grad_(False)


    prev_latents = pipe.generate_noise((1,16,latent_window_size,60,104),dtype=dtype,device=device)
    with torch.no_grad():
        latents = torch.randn_like(prev_latents)
        latents, vace_kwargs = pipe.prepare_vace_kwargs(
            latents, control_video[:(latent_window_size-1)*4 + 1],
            height=480, width=832, num_frames=81, tiled=False
        ) 
        latents[:,:,:history_latent_length] = history_latents
        for timestep in tqdm(denoise_timelist,desc='denoising'):
            timestep = torch.stack([timestep] * latents.shape[0], dim=0).to(dtype=dtype, device=device)
            input_timestep = timestep

            cond_pred = model_fn_wan_video(pipe.dit, vace=pipe.vace,x=latents, timestep=input_timestep,**prompt_ebd, **vace_kwargs)
            uncond_pred = model_fn_wan_video(pipe.dit, vace=pipe.vace,x=latents, timestep=input_timestep,**neg_prompt_ebd, **vace_kwargs)
            pred = uncond_pred + cfg * (cond_pred - uncond_pred)
            
            latents[:,:,history_latent_length:] = pipe.scheduler.step(pred[:,:,history_latent_length:], timestep[0] if timestep.dim()==1 else timestep[0,0], latents[:,:,history_latent_length:])


    pipe.load_models_to_device(['vae'])
    frames = pipe.decode_video(latents,tiled=True)
    pipe.load_models_to_device([])
    frames = pipe.tensor2video(frames[0])

    save_video(frames, save_path, fps=5, quality=8)


def test_wan_fun_hg_frameSpace():
    device = 'cuda:0'
    device = torch.device(device)
    dtype = torch.bfloat16
    num_inference_steps = 30
    shift = 8
    cfg = 3

    history_latent_length = 1
    latent_window_size = 5
    target_clip_num = 10

    history_noise_level = 0
    ref_depth_video_path = 'data/examples/demo/ngr_demo_depth.mp4'
    name = ref_depth_video_path.split('/')[-1].split('.')[0]
    prompt = '战士在战斗，csgo style'
    save_path = f'/root/leinyu/code/results/wan/fun/csgo_{history_noise_level}hg_{history_latent_length}_{target_clip_num}_{name}_latentLen{latent_window_size}_cfg{cfg}_catFrames.mp4'

    all_latent_len = target_clip_num * (latent_window_size - history_latent_length) + history_latent_length
    all_frames_len = (all_latent_len-1) * 4 + 1
    frame_window_size = (latent_window_size - 1) * 4 + 1

    
    control_video = VideoData(ref_depth_video_path,height=480,width=832,length=810).raw_data()

    # Load models
    model_manager = ModelManager(device=device)
    model_manager.load_models(
        [
            "/root/leinyu/model/Wan2.1-Fun-1.3B-Control/diffusion_pytorch_model.safetensors",
            "/root/leinyu/model/Wan2.1-Fun-1.3B-Control/models_t5_umt5-xxl-enc-bf16.pth",
            "/root/leinyu/model/Wan2.1-Fun-1.3B-Control/Wan2.1_VAE.pth",
        ],
        torch_dtype=torch.bfloat16, # You can set `torch_dtype=torch.float8_e4m3fn` to enable FP8 quantization.
    )
    lora_path = "exp_out/train_exp/0815_wan_fun_lora_csgo_ll5_hg1/lightning_lora_ckpts/lora-epoch=13-step=005000.ckpt"
    #lora_path_1 = "/mnt/aigc_cq/private/leinyu/code/DiffSynth-Studio/exp_out/train_exp/0728_wan_lora_csgo_df/lightning_lora_ckpts/lora-epoch=28-step=001500.ckpt"
    model_manager.load_lora(lora_path, lora_alpha=1)
    #model_manager.load_lora(lora_path_1, lora_alpha=1)
    pipe = WanVideoPipelineNonunipc.from_model_manager(model_manager, torch_dtype=torch.bfloat16, device=device)
    #pipe.enable_vram_management(num_persistent_param_in_dit=None)
    pipe.scheduler.set_timesteps(num_inference_steps, shift=shift,denoising_strength=1)


    pipe.load_models_to_device(["text_encoder"])
    neg_prompt = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
    prompt_ebd = pipe.encode_prompt(prompt)
    neg_prompt_ebd = pipe.encode_prompt(neg_prompt,positive=False)



    pipe.load_models_to_device(["dit"])
    denoise_timelist = pipe.scheduler.timesteps
    pipe.dit.requires_grad_(False)

    all_latents = []
    all_frames = []
    window_latents  = pipe.generate_noise((1,16,latent_window_size,60,104),dtype=dtype,device=device) # b 16 latent_window_size h w
    history_t_length = history_latent_length
    frame_stride = frame_window_size - (history_latent_length - 1) * 4 - 1
    with torch.no_grad():
        for i in tqdm(range(target_clip_num),desc='for clip'):
            latents = window_latents.clone() # 噪声部分使用相同的噪声
            iamge_emb = pipe.prepare_controlnet_kwargs(control_video[i*frame_stride:i*frame_stride+frame_window_size],height=480,width=832,num_frames=frame_window_size,tiled=False)
            if i == 0:
                history_t_length = 0    
            else:
                history_t_length = history_latent_length
                if history_noise_level == 0:
                    latents[:,:,:history_t_length] = history_latents
                else:
                    latents[:,:,:history_t_length] = pipe.scheduler.add_noise(latents[:,:,:history_t_length], history_latents, timestep=pipe.scheduler.timesteps[history_noise_level])

            for timestep in tqdm(denoise_timelist,desc='denoising'):
                timestep = torch.stack([timestep] * latents.shape[0], dim=0).to(dtype=dtype, device=device)
                # timestep = timestep.unsqueeze(1).repeat(1,21-history_t_length)
                # input_timestep = torch.cat([torch.ones(timestep.shape[0],history_t_length,dtype=dtype,device=device) * history_noise_level,timestep],dim=1)
                input_timestep = timestep

                cond_pred = model_fn_wan_video(pipe.dit,x=latents, timestep=input_timestep,**prompt_ebd,**iamge_emb)
                if cfg > 1:
                    uncond_pred = model_fn_wan_video(pipe.dit,x=latents, timestep=input_timestep,**neg_prompt_ebd,**iamge_emb)
                    pred = uncond_pred + cfg * (cond_pred - uncond_pred)
                else:
                    pred = cond_pred 
                
                latents[:,:,history_t_length:] = pipe.scheduler.step(pred[:,:,history_t_length:], timestep[0] if timestep.dim()==1 else timestep[0,0], latents[:,:,history_t_length:])
                #latents = pipe.scheduler.step(pred, timestep[0] if timestep.dim()==1 else timestep[0,0], latents)

            all_latents.append(latents[:,:,history_t_length:])
            # prev_latents = latents
            pipe.load_models_to_device(['vae'])
            frames = pipe.decode_video(latents,tiled=False) # b 3 t h w 
            all_frames.append(frames[:,:,history_t_length:])
            frames = frames[:,:,-(1 + 4*(history_latent_length-1)):].to(device=device,dtype=dtype) # b 3 17 h w 
            history_latents = pipe.encode_video(frames,tiled=False) # b c history_latent_lenght h w  

    
    # all_latents = torch.cat(all_latents,dim=2)
    # pipe.load_models_to_device(['vae'])
    # frames = pipe.decode_video(all_latents,tiled=False)
    # pipe.load_models_to_device([])
    # frames = pipe.tensor2video(frames[0])
    frames = pipe.tensor2video(torch.cat(all_frames,dim=2)[0])

    save_video(frames, save_path, fps=15, quality=8)
  

if __name__ == "__main__":
    #test_wan1_3_inversion()
    #test_wan1_3_inversion_1()
    #test_wan1_3_varyT()
    #test_fuse_latent()
    #test_fuse_latent_each_step()
    #test_fuse_latent_each_step_1()
    #test_fuse_latent_each_step_base()
    #test_fuse_latent_each_step_base_1()
    #test_video_dataset()
    #test_long_video_gen_history_guidance()
    #test_long_video_gen_history_guidance_vace()
    #test_long_video_gen_history_guidance_vace_frameSpace()
    #test_hg_vace()
    test_wan_fun_hg_frameSpace()
    #demo()
