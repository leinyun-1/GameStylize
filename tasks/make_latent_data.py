import torch, os, imageio, argparse
from torchvision.transforms import v2
from einops import rearrange
import lightning as pl
import pandas as pd
from diffsynth import WanVideoPipeline, ModelManager, load_state_dict
from peft import LoraConfig, inject_adapter_in_model
import torchvision
from PIL import Image
import numpy as np
from csgo_dataset import CsgoVideoDataset,CsgoD2VDataset
from doom_dataset import DoomH5Dataset
from fps_dataset import *
from thuman_dataset import VideoDataset,TextDataset,VideoTextNormalDataset
from ngr_dataset import NgrVideoClipDataset,NgrD2VDataset
ROOT_MAP = {
    'csgo': '/root/leinyu/data/csgo/hdf5_dm_july2021',
    'doom': '/root/leinyu/data/doom/freedoom_random_128_u2_rs_fix_v6_1280_720_hdf5',
    'thuman': '/root/leinyu/data/round/latent/',
    'ngr': '/root/leinyu/data/ngr/ngr_clip_latent/',
}

class LightningModelForDataProcess(pl.LightningModule):
    def __init__(self,vae_path,text_encoder_path=None, image_encoder_path=None, tiled=False, tile_size=(34, 34), tile_stride=(18, 16)):
        super().__init__()
        model_path = [vae_path]
        if image_encoder_path is not None:
            model_path.append(image_encoder_path)
        if text_encoder_path is not None:
            model_path.append(text_encoder_path)
        model_manager = ModelManager(torch_dtype=torch.bfloat16, device="cpu")
        model_manager.load_models(model_path)
        self.pipe = WanVideoPipeline.from_model_manager(model_manager)

        self.tiler_kwargs = {"tiled": tiled, "tile_size": tile_size, "tile_stride": tile_stride}

        #self.save_root = '/root/leinyu/data/csgo/hdf5_dm_july2021'
        #self.save_root = '/root/leinyu/data/round/latent/'
        #self.save_root = '/root/leinyu/data/ngr/ngr_clip_latent/'
        #self.save_root = '/root/leinyu/data/csgo/hdf5_dm_july2021_d2v_latent/'
        #self.save_root = '/root/leinyu/data/ngr/ngr_d2v_latent/'
        #self.save_root = '/root/leinyu/data/thuman2.1/Thuman2.1_norm_render_1/latent_11'
        self.save_root = '/root/leinyu/data/fps/latent_d2v/delta'
        os.makedirs(self.save_root, exist_ok=True)
        
    def test_step(self, batch, batch_idx): # 仅适配 bs=1
        video, name = batch["pixel_values"], batch["name"][0]
        save_path = os.path.join(self.save_root,name+'.tensors.pth')
        
        self.pipe.device = self.device
        if video is not None:
            # video
            video = video.to(dtype=self.pipe.torch_dtype, device=self.pipe.device)
            latents = self.pipe.encode_video(video, **self.tiler_kwargs)[0]
            data = {"latents": latents}

            # prompt
            if 'text' in batch:
                text = batch['text'][0]
                prompt_emb = self.pipe.encode_prompt(text)['context'][0]
                data.update({"prompt_emb": prompt_emb})

            # image
            if "first_frame" in batch:
                first_frame = Image.fromarray(batch["first_frame"][0].cpu().numpy().astype(np.uint8))
                _, _, num_frames, height, width = video.shape
                image_emb = self.pipe.encode_image(first_frame, None, num_frames, height, width)
                data.update(image_emb)

            if "depth_values" in batch:
                depth = batch["depth_values"].to(dtype=self.pipe.torch_dtype, device=self.pipe.device)
                depth_latents = self.pipe.encode_video(depth)[0]  # 深度图通道为1，不知vae能否处理，若深度图通道为3则没那么多事（而这件事很容易办到）
                data.update({"control_latents": depth_latents})

                vace_mask = torch.ones_like(depth)
                inactive = depth * (1 - vace_mask) + 0 * vace_mask
                reactive = depth * vace_mask + 0 * (1 - vace_mask)
                inactive = self.pipe.encode_video(inactive, tiled=False).to(dtype=self.pipe.torch_dtype, device=self.device)
                reactive = self.pipe.encode_video(reactive, tiled=False).to(dtype=self.pipe.torch_dtype, device=self.device)
                vace_video_latents = torch.concat((inactive, reactive), dim=1) # c=32
                vace_mask_latents = rearrange(vace_mask[0,0], "T (H P) (W Q) -> 1 (P Q) T H W", P=8, Q=8) # 1 64 t h w 
                vace_mask_latents = torch.nn.functional.interpolate(vace_mask_latents, size=((vace_mask_latents.shape[2] + 3) // 4, vace_mask_latents.shape[3], vace_mask_latents.shape[4]), mode='nearest-exact') # mask不需要encode，只需要下采样 1 64 21 h w 
                vace_context = torch.concat((vace_video_latents, vace_mask_latents), dim=1)
                data.update({"vace_context": vace_context[0]})
            
            if "normal_pixel_values" in batch:
                video = batch['normal_pixel_values']
                video = video.to(dtype=self.pipe.torch_dtype, device=self.pipe.device)
                latents = self.pipe.encode_video(video, **self.tiler_kwargs)[0]
                data.update({"normal_latents": latents})


            #save_path = os.path.join(self.save_root,name+'.tensors.pth')
            torch.save(data, save_path)

class LightningModelForTextProcess(pl.LightningModule):
    def __init__(self,vae_path,text_encoder_path=None, image_encoder_path=None, tiled=False, tile_size=(34, 34), tile_stride=(18, 16)):
        super().__init__()
        model_path = [text_encoder_path]
        model_manager = ModelManager(torch_dtype=torch.bfloat16, device="cpu")
        model_manager.load_models(model_path)
        self.pipe = WanVideoPipeline.from_model_manager(model_manager)

        self.save_root = '/root/leinyu/data/round/latent/'
        
    def test_step(self, batch, batch_idx): # 仅适配 bs=1
        self.pipe.device = self.device
        # prompt
        data = {}
        text,name = batch['text'][0],batch['path'][0]
        prompt_emb = self.pipe.encode_prompt(text)['context'][0]
        data.update({"prompt_emb": prompt_emb})

        save_path = name.replace('.txt','tensors.pth')
        torch.save(data, save_path)


def main(cfg):
    text_encoder_path = '/dockerdata/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth'
    vae_path = '/dockerdata/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth'
    image_encoder_path = '/dockerdata/SkyReels-V2-I2V-1.3B-540P/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth'

    model = LightningModelForDataProcess(vae_path,text_encoder_path,image_encoder_path)
    #dataset = CsgoVideoDataset(image_size=(480,832),num_frames=81,num_prefix_frames=0,data_root='/dockerdata/hdf5_dm_july2021/hdf5_dm_july2021_tars')
    #dataset = CsgoD2VDataset()
    #dataset = DoomH5Dataset(size=[81,480,832],prefix_action=0)
    #dataset = VideoClipDataset(root='/root/leinyu/data/fps/delta/')
    #dataset = TextDataset()
    #dataset = NgrVideoClipDataset()
    #dataset = NgrD2VDataset()  
    #dataset = VideoTextNormalDataset(interval=2)
    dataset = FpsD2VDataset()
    dataloader = torch.utils.data.DataLoader(dataset,batch_size=cfg.batch_size,num_workers=8,pin_memory=True)

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=cfg.device,  # 自动多卡
    )

    # 开始训练
    trainer.test(model,dataloader)


def make_prompt_ebd():
    csgo_video_prompts = [
    "a dynamic csgo-style video with game-like motion and atmosphere",
    "a 3d animated video in csgo aesthetic with smooth camera movement",
    "a stylized video sequence in the visual style of csgo",
    "cinematic fps-style lighting and textures in a csgo-inspired animation",
    "a short animated scene of a csgo environment concept",
    "a video clip that mimics csgo gameplay visuals and tone",
    "an fps game-style animated sequence inspired by csgo",
    "csgo-style video with dramatic color grading and lighting transitions",
    "a gritty csgo-inspired video with fps-style motion and atmosphere",
    "an animated csgo-style video with realistic game-like rendering"
    ]

    
    model_path = ["/dockerdata/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth"]
    model_manager = ModelManager(torch_dtype=torch.bfloat16, device="cuda:0")
    model_manager.load_models(model_path)
    pipe = WanVideoPipeline.from_model_manager(model_manager,device='cuda:0')

    prompt_embs = []
    for prompt in csgo_video_prompts:
        prompt_emb = pipe.encode_prompt(prompt)['context'][0] # 512 4096
        prompt_embs.append(prompt_emb)
    
    prompt_embs = torch.stack(prompt_embs,dim=0) # 10 512 4096
    print(prompt_embs.shape)

    save_path = '/root/leinyu/data/csgo/prompt_embs_10_512_4096.pth'
    torch.save(prompt_embs,save_path)



if __name__ == "__main__":
    from types import SimpleNamespace
    config = {
    'device': [0,1],
    'batch_size': 1,
    }
    cfg = SimpleNamespace(**config)
    main(cfg)

    #make_prompt_ebd()
    


