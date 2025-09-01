import torch, os, imageio, argparse
from torchvision.transforms import v2
from einops import rearrange
import lightning as pl
from lightning.pytorch.callbacks import Callback
# from lightning.pytorch.loggers import TensorBoardLogger
import pandas as pd
from diffsynth import WanVideoPipeline, WanVideoPipelineNonunipc, ModelManager, load_state_dict
from diffsynth.schedulers.flow_match import FlowMatchScheduler
from peft import LoraConfig, inject_adapter_in_model
import torchvision
from PIL import Image
import numpy as np
import json 
import time 
from model import WanAction,load_lora
from safetensors.torch import load_file




class TextVideoYmzxDataset(torch.utils.data.Dataset):
    def __init__(self, base_path= '/mnt/kaiwu-group-y1-sh-hdd/chesterlv/datasets/394879_v2_qwen25vl_32b_caption/', max_num_frames=81, frame_interval=1, num_frames=81, height=480, width=832, is_i2v=True):
        video_path = os.path.join(base_path, "videos")
        if 'v2' in base_path:
            caption_path = os.path.join(base_path, "captions_v3")
        else:
            caption_path = os.path.join(base_path, "captions_v2")
        actions_path = os.path.join(base_path, "actions")

        map_ids = os.listdir(video_path)
        self.video_subs = []
        self.caption_subs = []
        self.action_subs = []
        for map_id in map_ids:
            video_map_id_path = os.path.join(video_path, map_id)
            # self.video_subs += [os.path.join(map_id_path, i) for i in os.listdir(map_id_path)]
            caption_map_id_path = os.path.join(caption_path, map_id)
            # self.caption_subs += [os.path.join(map_id_path, i) for i in os.listdir(map_id_path)]
            action_map_id_path = os.path.join(actions_path, map_id)
            action_subs = [i.split('.')[0] for i in os.listdir(action_map_id_path)]
            self.video_subs += [os.path.join(video_map_id_path, i + '.mp4') for i in action_subs]
            self.caption_subs += [os.path.join(caption_map_id_path, i + '.json') for i in action_subs]
            self.action_subs += [os.path.join(action_map_id_path, i + '.npy') for i in action_subs]

  
        self.max_num_frames = max_num_frames
        self.frame_interval = frame_interval
        self.num_frames = num_frames
        self.height = height
        self.width = width
        self.is_i2v = is_i2v
        self.action_num = 13
            
        self.frame_process = v2.Compose([
            v2.CenterCrop(size=(height, width)),
            v2.Resize(size=(height, width), antialias=True),
            v2.ToTensor(),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        self.prepare_action_mapping()


    def prepare_action_mapping(self):
        action_caption = ["The character is moving forward.", "The character is moving to the left.", "The character is moving backward.", "The character is moving to the right.", "The perspective in the video is looking upwards.", "The perspective in the video is looking to the left.", "The perspective in the video is looking downwards.", "The perspective in the video is looking to the right.", "The character is jumping.", "z", "x", "c"]
        action_name =    ["w", "a", "s", "d", "Key.up", "Key.left", "Key.down", "Key.right", "Key.space", "z", "x", "c", "Stop"]
        action_list =    [2,    3,   4,   1,      9,        10,         11,         12,           5,        7,  13,    8, 6]
        turn_action_list=[10,  11,  12,  13,     14,         3,         15,          4,          16,       17,  18,   19, 6]
        self.action_map = dict()
        self.turn_action_map = dict()
        for i in range(self.action_num):
            self.action_map[action_list[i]] = i
            self.turn_action_map[turn_action_list[i]] = i
        
        
    def crop_and_resize(self, image):
        width, height = image.size
        scale = max(self.width / width, self.height / height)
        image = torchvision.transforms.functional.resize(
            image,
            (round(height*scale), round(width*scale)),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR
        )
        return image


    def load_frames_using_imageio(self, file_path, max_num_frames, start_frame_id, interval, num_frames, frame_process):
        reader = imageio.get_reader(file_path)
        if reader.count_frames() < max_num_frames or reader.count_frames() - 1 < start_frame_id + (num_frames - 1) * interval:
            reader.close()
            return None
        
        frames = []
        first_frame = None
        for frame_id in range(num_frames):
            frame = reader.get_data(start_frame_id + frame_id * interval)
            frame = Image.fromarray(frame)
            frame = self.crop_and_resize(frame)
            if first_frame is None:
                first_frame = frame
            frame = frame_process(frame)
            frames.append(frame)
        reader.close()

        frames = torch.stack(frames, dim=0)
        frames = rearrange(frames, "T C H W -> C T H W")
        
        first_frame = v2.functional.center_crop(first_frame, output_size=(self.height, self.width))
        first_frame = np.array(first_frame)

        if self.is_i2v:
            return frames, first_frame
        else:
            return frames


    def load_video(self, file_path):
        start_frame_id = torch.randint(0, self.max_num_frames - (self.num_frames - 1) * self.frame_interval, (1,))[0]
        frames = self.load_frames_using_imageio(file_path, self.max_num_frames, start_frame_id, self.frame_interval, self.num_frames, self.frame_process)
        return frames
    
    
    def is_image(self, file_path):
        file_ext_name = file_path.split(".")[-1]
        if file_ext_name.lower() in ["jpg", "jpeg", "png", "webp"]:
            return True
        return False
    
    
    def load_image(self, file_path):
        frame = Image.open(file_path).convert("RGB")
        frame = self.crop_and_resize(frame)
        first_frame = frame
        frame = self.frame_process(frame)
        frame = rearrange(frame, "C H W -> C 1 H W")
        return frame


    def load_action(self, action_path):
        action = np.load(action_path) # （2f，） raw data即为一帧对应两个动作

        game_actions = []
        for frame_id in range(self.num_frames): 
            init_action = np.zeros(self.action_num*2)
            action_id = int(action[frame_id*self.frame_interval*2])
            action_id_1 = int(action[frame_id*self.frame_interval*2 + 1])

            if action_id in self.action_map:
                a_id = self.action_map[action_id]
                init_action[a_id] = 1
            else:
                a_id = 0
            if action_id in self.turn_action_map:
                t_id = self.turn_action_map[action_id]
                init_action[t_id] = 1

            if action_id_1 in self.action_map:
                a_id_1 = self.action_map[action_id_1]
                init_action[a_id_1+self.action_num] = 1
            else:
                a_id_1 = 0
            if action_id_1 in self.turn_action_map:
                t_id_1 = self.turn_action_map[action_id_1]
                init_action[t_id_1+self.action_num] = 1

            game_actions.append(init_action)
        game_actions = np.stack(game_actions) # f*(2*action_num)
        return game_actions


    def __getitem__(self, data_id):

        video_path = self.video_subs[data_id]
        caption_path = self.caption_subs[data_id]
        action_path = self.action_subs[data_id]

        
        video = self.load_video(video_path)
        if isinstance(video,tuple):
            video, first_frame = video

        with open(caption_path, 'r') as f:
            json_data = json.load(f)
        text = ''
        for item in json_data:
            text += str(item) + ':'
            text += json_data[item]

        #game_actions = self.load_action(action_path)


        id_path = video_path.split("/")[-2:]
        id_path[-1] = id_path[-1].split(".")[0] 
        id_path = '/'.join(id_path) # map_id/video_id
        data = {"text": text, "video": video, "path": id_path}  #,"action": game_actions}
        if first_frame is not None:
            data["first_frame"] = first_frame
        return data
    

    def __len__(self):
        return len(self.action_subs)



class LightningModelForYmzxDataProcess(pl.LightningModule):
    def __init__(self, text_encoder_path, vae_path, image_encoder_path=None, save_dir='/mnt/aigc_cq/private/leinyu/data/ymzx/latent_v2/', tiled=False, tile_size=(34, 34), tile_stride=(18, 16)):
        super().__init__()
        model_path = [text_encoder_path, vae_path]
        if image_encoder_path is not None:
            model_path.append(image_encoder_path)
        model_manager = ModelManager(torch_dtype=torch.bfloat16, device="cpu")
        model_manager.load_models(model_path)
        self.pipe = WanVideoPipeline.from_model_manager(model_manager)

        self.tiler_kwargs = {"tiled": tiled, "tile_size": tile_size, "tile_stride": tile_stride}
        self.save_dir = save_dir

    def test_step(self, batch, batch_idx): # for all: latent + prompt ebd + y + clip_fea
        text, video, path = batch["text"], batch["video"], batch["path"]
        bs = len(text)
        save_path = self.save_dir +  "/" + path[0] + ".tensors.pth"
        if os.path.exists(save_path):
            return 
        
        self.pipe.device = self.device
        if video is not None:
            # video
            video = video.to(dtype=self.pipe.torch_dtype, device=self.pipe.device)
            latents = self.pipe.encode_video(video, **self.tiler_kwargs)

            for bid in range(bs):
                # image
                if "first_frame" in batch:
                    first_frame = Image.fromarray(batch["first_frame"][bid].cpu().numpy())
                    _, _, num_frames, height, width = video.shape
                    image_emb = self.pipe.encode_image(first_frame, None, num_frames, height, width)
                else:
                    image_emb = {}

                # prompt
                prompt_emb = self.pipe.encode_prompt(text[bid])
                data = {"latents": latents[bid], "prompt_emb": prompt_emb, "y": image_emb['y'][0], "clip_fea": image_emb['clip_feature'][0]}

                save_dir = self.save_dir + path[bid].split("/")[0]
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                save_path = save_dir +  "/" + path[bid].split("/")[-1] + ".tensors.pth"

                torch.save(data, save_path)



class LightningModelForYmzxDataProcessYClipfea(pl.LightningModule):
    def __init__(self,text_encoder_path, vae_path, image_encoder_path, save_dir='/mnt/aigc_cq/private/leinyu/data/ymzx/latent/', tiled=False, tile_size=(34, 34), tile_stride=(18, 16)):
        super().__init__()
        model_path = [image_encoder_path, vae_path]
        model_manager = ModelManager(torch_dtype=torch.bfloat16, device="cpu")
        model_manager.load_models(model_path)
        self.pipe = WanVideoPipeline.from_model_manager(model_manager)

        self.tiler_kwargs = {"tiled": tiled, "tile_size": tile_size, "tile_stride": tile_stride}
        self.save_dir = save_dir

    def test_step(self, batch, batch_idx): # for y + clip_fea
        text, video, path = batch["text"], batch["video"], batch["path"]
        bs = len(text)
        self.pipe.device = self.device

        for bid in range(bs):
            #print(f"Processing batch {path[bid]}")
            save_path = self.save_dir +  "/" + path[bid] + ".tensors.pth"

            if os.path.exists(save_path):
                data = torch.load(save_path,map_location=self.device)
                if 'y' in data.keys():
                    continue
            else:
                continue
        
                                # image
            first_frame = Image.fromarray(batch["first_frame"][bid].cpu().numpy())
            _, _, num_frames, height, width = video.shape
            image_emb = self.pipe.encode_image(first_frame, None, num_frames, height, width)


            # prompt
            data.update({"y": image_emb['y'][0], "clip_fea": image_emb['clip_feature'][0]})
            torch.save(data, save_path)



class YmzxTensorDataset(torch.utils.data.Dataset):
    def __init__(self, base_path='/mnt/aigc_cq/private/leinyu/data/ymzx/latent/', steps_per_epoch=500, num_maps=1):
        self.problem_clip_list = self.get_problem_clip_list()
        maps = os.listdir(base_path)[:num_maps]
        self.latent_subs = []
        for map_id in maps:
            latent_map_id_path = os.path.join(base_path, map_id)
            clips = os.listdir(latent_map_id_path)
            self.latent_subs += [os.path.join(latent_map_id_path, i) for i in clips if i not in self.problem_clip_list]
        
        self.steps_per_epoch = len(self.latent_subs)

        raw_data_root = '/mnt/kaiwu-group-y1-sh-hdd/chesterlv/datasets/394879_qwen25vl_32b_caption/'
        self.actions_path = os.path.join(raw_data_root, "actions")
        self.video_path = os.path.join(raw_data_root,'videos')
        self.action_num = 20 # onehot 长度

    def get_problem_clip_list(self):
        json_path = '/root/leinyu/code/experiments/output/occlusion_stats_all.json'
        with open(json_path, 'r', encoding='utf-8') as f:
            data_dict = json.load(f)
        occ_clip_list = []
        for map, value in data_dict.items():
            for clip in value['occ_clip_name']:
                occ_clip_list.append(f"{clip}")

        json_path = '/root/leinyu/code/experiments/output/still_stats_all.json'
        with open(json_path, 'r', encoding='utf-8') as f:
            data_dict = json.load(f)
        still_clip_list = []
        for map, value in data_dict.items():
            for clip in value['still_clip_name']:
                still_clip_list.append(f"{clip}")

        problem_clip_list = occ_clip_list + still_clip_list
        problem_clip_list = set(problem_clip_list)
        
        return problem_clip_list


    def get_action(self, action_path):
        actions = np.load(action_path) # 162,
        actions = actions[::2]
        actions = np.concatenate([[actions[0]] * 3, actions])
            # Action to one-hot:
        action_batchsize = actions.shape[0]
        action_spacesize = self.action_num
        action_onehots = np.zeros((action_batchsize, action_spacesize))
        action_onehots[[i for i in range(action_batchsize)], np.array(actions).astype(np.int32)] = 1.

        action_onehots = action_onehots.reshape(action_batchsize // 4, 4, action_spacesize).reshape(action_batchsize // 4, 4 * action_spacesize)
        
        return torch.from_numpy(action_onehots) # 21 80

    def __getitem__(self, index):
        # data_id = torch.randint(0, len(self.latent_subs), (1,))[0]
        # data_id = (data_id + index) % len(self.latent_subs) # For fixed seed.
        data_id = index
        path = self.latent_subs[data_id]
        data = torch.load(path, weights_only=True, map_location="cpu")

        id_path = path.split("/")[-1].split(".")[0] 

        name = path.split('/')[-2:]
        name[-1] = name[-1].split('.')[0]
        name = '/'.join(name)

        action_path = os.path.join(self.actions_path, name+'.npy')
        action  = self.get_action(action_path)

        #data["prompt_emb"]['context'] = data["prompt_emb"]['context'][0]
        return {"prompt_emb": data["prompt_emb"], "latents": data["latents"],"image_emb":{},"name":name,"actions":action}
    

    def __len__(self):
        return self.steps_per_epoch



class LightningModelForTrain(pl.LightningModule):
    def __init__(
        self,
        dit_path,
        learning_rate=1e-5,
        lora_rank=4, lora_alpha=4, train_architecture="lora", lora_target_modules="q,k,v,o,ffn.0,ffn.2", init_lora_weights="kaiming",
        use_gradient_checkpointing=True, use_gradient_checkpointing_offload=False,
        pretrained_lora_path=None
    ):
        super().__init__()
        model_manager = ModelManager(torch_dtype=torch.bfloat16, device="cpu")
        if os.path.isfile(dit_path):
            model_manager.load_models([dit_path])
        else:
            dit_path = dit_path.split(",")
            model_manager.load_models([dit_path])
        
        self.pipe = WanVideoPipeline.from_model_manager(model_manager)
        self.pipe.scheduler.set_timesteps(1000, training=True)
        self.freeze_parameters()
        if train_architecture == "lora":
            self.add_lora_to_model(
                self.pipe.denoising_model(),
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
                lora_target_modules=lora_target_modules,
                init_lora_weights=init_lora_weights,
                pretrained_lora_path=pretrained_lora_path,
            )
        else:
            self.pipe.denoising_model().requires_grad_(True) 

        
        self.learning_rate = learning_rate
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        
        
    def freeze_parameters(self):
        # Freeze parameters
        self.pipe.requires_grad_(False)
        self.pipe.eval()
        self.pipe.denoising_model().train()
        
        
    def add_lora_to_model(self, model, lora_rank=4, lora_alpha=4, lora_target_modules="q,k,v,o,ffn.0,ffn.2", init_lora_weights="kaiming", pretrained_lora_path=None, state_dict_converter=None):
        # Add LoRA to UNet
        self.lora_alpha = lora_alpha
        if init_lora_weights == "kaiming":
            init_lora_weights = True
            
        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            init_lora_weights=init_lora_weights,
            target_modules=lora_target_modules.split(","),
        )
        model = inject_adapter_in_model(lora_config, model)
        for param in model.parameters():
            # Upcast LoRA parameters into fp32
            if param.requires_grad:
                param.data = param.to(torch.float32)
                
        # Lora pretrained lora weights
        if pretrained_lora_path is not None:
            state_dict = load_state_dict(pretrained_lora_path)
            if state_dict_converter is not None:
                state_dict = state_dict_converter(state_dict)
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            all_keys = [i for i, _ in model.named_parameters()]
            num_updated_keys = len(all_keys) - len(missing_keys)
            num_unexpected_keys = len(unexpected_keys)
            print(f"{num_updated_keys} parameters are loaded from {pretrained_lora_path}. {num_unexpected_keys} parameters are unexpected.")
    

    def training_step(self, batch, batch_idx):
        # Data
        latents = batch["latents"].to(self.device)
        prompt_emb = batch["prompt_emb"]
        prompt_emb["context"] = prompt_emb["context"][:,0].to(self.device)
        # Add CFG random dropout (30%概率置零条件)
        if torch.rand(1) < 0.3:
            prompt_emb["context"] = torch.zeros_like(prompt_emb["context"])
        image_emb = batch["image_emb"]
        if "clip_feature" in image_emb:
            image_emb["clip_feature"] = image_emb["clip_feature"][0].to(self.device)
        if "y" in image_emb:
            image_emb["y"] = image_emb["y"][0].to(self.device)

        # Loss
        self.pipe.device = self.device
        noise = torch.randn_like(latents)
        timestep_id = torch.randint(0, self.pipe.scheduler.num_train_timesteps, (1,))
        timestep = self.pipe.scheduler.timesteps[timestep_id].to(dtype=self.pipe.torch_dtype, device=self.pipe.device)
        extra_input = self.pipe.prepare_extra_input(latents)
        noisy_latents = self.pipe.scheduler.add_noise(latents, noise, timestep)
        training_target = self.pipe.scheduler.training_target(latents, noise, timestep)

        # Compute loss
        noise_pred = self.pipe.denoising_model()(
            noisy_latents, timestep=timestep, **prompt_emb, **extra_input, **image_emb,
            use_gradient_checkpointing=self.use_gradient_checkpointing,
            use_gradient_checkpointing_offload=self.use_gradient_checkpointing_offload
        )
        loss = torch.nn.functional.mse_loss(noise_pred.float(), training_target.float())
        loss = loss * self.pipe.scheduler.training_weight(timestep) # 忽略了这句话，这个weight很重

        # Record log
        self.log("train_loss", loss, prog_bar=True)  # 原来自带，则不用callback去print了。
        return loss


    def configure_optimizers(self):
        trainable_modules = filter(lambda p: p.requires_grad, self.pipe.denoising_model().parameters())
        optimizer = torch.optim.AdamW(trainable_modules, lr=self.learning_rate)
        return optimizer
    

    def on_save_checkpoint(self, checkpoint):
        checkpoint.clear()
        trainable_param_names = list(filter(lambda named_param: named_param[1].requires_grad, self.pipe.denoising_model().named_parameters()))
        trainable_param_names = set([named_param[0] for named_param in trainable_param_names])
        state_dict = self.pipe.denoising_model().state_dict()
        lora_state_dict = {}
        for name, param in state_dict.items():
            if name in trainable_param_names:
                lora_state_dict[name] = param
        checkpoint.update(lora_state_dict)



class WanActionLg(pl.LightningModule):
    def __init__(
        self,
        learning_rate=1e-5,
        lora_rank=4, lora_alpha=4, train_architecture="lora", lora_target_modules="q,k,v,o,ffn.0,ffn.2", init_lora_weights="kaiming",
        use_gradient_checkpointing=True, use_gradient_checkpointing_offload=False,
        pretrained_lora_path=None
    ):
        super().__init__()

        self.pipe = WanAction()
        wan_state_dict = load_file('models/Wan-AI/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors',device='cpu')
        self.pipe.model.load_state_dict(wan_state_dict,strict=True)
        load_lora(self.pipe.model,torch.load('exp_out/train_exp/0531/checkpoints/epoch=0-step=400.ckpt',map_location='cpu'))
        action_ckpt_path = 'exp_out/train_exp/0626_wanaction_newblock/checkpoints/epoch=0-step=6000.ckpt'
        action_ckpt = torch.load(action_ckpt_path,map_location='cpu')
        _,unexpected_keys = self.pipe.load_state_dict(action_ckpt,strict=False)
        self.pipe.to(dtype=torch.bfloat16)

        self.freeze_parameters()

        self.scheduler = FlowMatchScheduler(shift=5, sigma_min=0.0, extra_one_step=True)
        self.scheduler.set_timesteps(1000,training=True)
        
        
        self.learning_rate = learning_rate
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        
        
    def freeze_parameters(self):
        # Freeze parameters
        self.pipe.model.requires_grad_(False)
        self.pipe.eval()
        

    def training_step(self, batch, batch_idx):
        # Data
        latents = batch["latents"].to(self.device).to(torch.bfloat16)
        prompt_emb = batch["prompt_emb"]
        prompt_emb["context"] = prompt_emb["context"][:,0].to(self.device).to(torch.bfloat16)
        # Add CFG random dropout (30%概率置零条件)
        # if torch.rand(1) < 0.3:
        #     prompt_emb["context"] = torch.zeros_like(prompt_emb["context"])

        actions = batch['actions'].to(self.device).to(torch.bfloat16)
        # if torch.rand(1) < 0.2:
        #     actions = torch.zeros_like(actions)


        # Loss
        bs = latents.shape[0]
        noise = torch.randn_like(latents)
        timestep_id = torch.randint(0, self.scheduler.num_train_timesteps, (bs,))
        timestep = self.scheduler.timesteps[timestep_id].to(dtype=self.dtype, device=self.device) # 实际上通过这种方式依然一定程度上改变了训练t的分布
        noisy_latents = self.scheduler.add_noise(latents, noise, timestep) # 整个函数写的简陋，timestep的bs必须为1
        training_target = self.scheduler.training_target(latents, noise, timestep)

        # Compute loss
        noise_pred = self.pipe(
            noisy_latents, timestep=timestep, action=actions,  **prompt_emb,
            use_gradient_checkpointing=self.use_gradient_checkpointing,
        )
        loss = torch.nn.functional.mse_loss(noise_pred.float(), training_target.float(),reduction='none').mean(dim=tuple(range(1, noise_pred.dim()))) # 保留bs维度
        loss = torch.mean(loss * self.scheduler.training_weight(timestep).to(loss))# 忽略了这句话，这个weight很重

        # Record log
        self.log("train_loss", loss, prog_bar=True)  # 原来自带，则不用callback去print了。
        return loss


    def configure_optimizers(self):
        trainable_modules = filter(lambda p: p.requires_grad, self.pipe.parameters())
        optimizer = torch.optim.AdamW(trainable_modules, lr=self.learning_rate)
        return optimizer
    

    def on_save_checkpoint(self, checkpoint):
        checkpoint.clear()
        trainable_param_names = list(filter(lambda named_param: named_param[1].requires_grad, self.pipe.named_parameters()))
        trainable_param_names = set([named_param[0] for named_param in trainable_param_names])
        state_dict = self.pipe.state_dict()
        lora_state_dict = {}
        for name, param in state_dict.items():
            if name in trainable_param_names:
                lora_state_dict[name] = param
        checkpoint.update(lora_state_dict)



class WanAction_1(pl.LightningModule):
    def __init__(
        self,
        dit_path,
        learning_rate=1e-5,
        lora_rank=4, lora_alpha=4, train_architecture="lora", lora_target_modules="q,k,v,o,ffn.0,ffn.2", init_lora_weights="kaiming",
        use_gradient_checkpointing=True, use_gradient_checkpointing_offload=False,
        pretrained_lora_path=None
    ):
        super().__init__()
        model_manager = ModelManager(torch_dtype=torch.bfloat16, device="cpu")
        if os.path.isfile(dit_path):
            model_manager.load_models([dit_path])
        else:
            dit_path = dit_path.split(",")
            model_manager.load_models([dit_path])
        
        self.pipe = WanVideoPipelineNonunipc.from_model_manager(model_manager)
        self.pipe.scheduler.set_timesteps(1000, training=True)
        self.freeze_parameters()

        self.pipe.denoising_model().has_action_input = True
        self.pipe.denoising_model().set_action_projection(4*20) # 注入lora和注入lora参数、action参数同时发生， 则该模块必须在注入之前初始化，得以载入参数
        self.pipe.denoising_model().set_ar_attention()

        if train_architecture == "lora":
            self.add_lora_to_model(
                self.pipe.denoising_model(),
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
                lora_target_modules=lora_target_modules,
                init_lora_weights=init_lora_weights,
                pretrained_lora_path=pretrained_lora_path,
            )
            self.pipe.denoising_model().action_projection.requires_grad_(True) #注入lora模块时会把其他模块参数关掉，这里需手动开启
        else:
            self.pipe.denoising_model().requires_grad_(True) 
        
        #self.pipe.denoising_model().to(torch.bfloat16)

        
        self.learning_rate = learning_rate
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        
        
    def freeze_parameters(self):
        # Freeze parameters
        self.pipe.requires_grad_(False)
        self.pipe.eval()
        self.pipe.denoising_model().train()
        
        
    def add_lora_to_model(self, model, lora_rank=4, lora_alpha=4, lora_target_modules="q,k,v,o,ffn.0,ffn.2", init_lora_weights="kaiming", pretrained_lora_path=None, state_dict_converter=None):
        # Add LoRA to UNet
        self.lora_alpha = lora_alpha
        if init_lora_weights == "kaiming":
            init_lora_weights = True
            
        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            init_lora_weights=init_lora_weights,
            target_modules=lora_target_modules.split(","),
        )
        model = inject_adapter_in_model(lora_config, model)
        for param in model.parameters():
            # Upcast LoRA parameters into fp32
            if param.requires_grad:
                param.data = param.to(torch.float32)
        
        for name, param in model.named_parameters():
            if "action_projection" in name:
                param.requires_grad_(True)
                
        # Lora pretrained lora weights
        if pretrained_lora_path is not None:
            state_dict = load_state_dict(pretrained_lora_path)
            if state_dict_converter is not None:
                state_dict = state_dict_converter(state_dict)
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            all_keys = [i for i, _ in model.named_parameters()]
            num_updated_keys = len(all_keys) - len(missing_keys)
            num_unexpected_keys = len(unexpected_keys)
            print(f"{num_updated_keys} parameters are loaded from {pretrained_lora_path}. {num_unexpected_keys} parameters are unexpected.")

    

    def training_step(self, batch, batch_idx):
        # Data
        latents = batch["latents"].to(self.device).to(torch.bfloat16)
        prompt_emb = batch["prompt_emb"]
        prompt_emb["context"] = prompt_emb["context"][:,0].to(self.device).to(torch.bfloat16)
        actions = batch['actions'].to(self.device).to(torch.bfloat16)

        # Loss
        self.pipe.device = self.device
        bs = latents.shape[0]
        noise = torch.randn_like(latents)
        timestep_id = torch.randint(0, self.pipe.scheduler.num_train_timesteps, (bs,))
        timestep = self.pipe.scheduler.timesteps[timestep_id].to(dtype=self.pipe.torch_dtype, device=self.pipe.device)
        extra_input = self.pipe.prepare_extra_input(latents)
        noisy_latents = self.pipe.scheduler.add_noise(latents, noise, timestep)
        training_target = self.pipe.scheduler.training_target(latents, noise, timestep)

        # Compute loss
        noise_pred = self.pipe.denoising_model()(
            noisy_latents, timestep=timestep, **prompt_emb, **extra_input, action=actions,
            use_gradient_checkpointing=self.use_gradient_checkpointing,
            use_gradient_checkpointing_offload=self.use_gradient_checkpointing_offload
        )
        loss = torch.nn.functional.mse_loss(noise_pred.float(), training_target.float(),reduction='none').mean(dim=tuple(range(1, noise_pred.dim()))) # 保留bs维度
        loss = torch.mean( loss * self.pipe.scheduler.training_weight(timestep).to(loss) )# 忽略了这句话，这个weight很重

        # Record log
        self.log("train_loss", loss, prog_bar=True)  # 原来自带，则不用callback去print了。
        return loss


    def configure_optimizers(self):
        trainable_modules = filter(lambda p: p.requires_grad, self.pipe.denoising_model().parameters())
        optimizer = torch.optim.AdamW(trainable_modules, lr=self.learning_rate)
        return optimizer
    

    def on_save_checkpoint(self, checkpoint):
        checkpoint.clear()
        trainable_param_names = list(filter(lambda named_param: named_param[1].requires_grad, self.pipe.denoising_model().named_parameters()))
        trainable_param_names = set([named_param[0] for named_param in trainable_param_names])
        state_dict = self.pipe.denoising_model().state_dict()
        lora_state_dict = {}
        for name, param in state_dict.items():
            if name in trainable_param_names:
                lora_state_dict[name] = param
        checkpoint.update(lora_state_dict)





class BatchTimeLogger(Callback):
    def __init__(self):
        super().__init__()
        self.batch_start_time = time.time()
        self.batch_end_time = time.time()

    def on_test_batch_start(self, trainer, pl_module, batch, batch_idx):
        # 记录 batch 开始时间
        if trainer.is_global_zero:
            self.batch_start_time = time.time()
            data_time = self.batch_start_time - self.batch_end_time
            #print(f"data_time: {(self.batch_start_time - self.batch_end_time):.2f}seconds, step={trainer.global_step}")
            pl_module.log("data_time", data_time, prog_bar=True,logger=False)

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # 计算 batch 处理时间
        if trainer.is_global_zero:
            self.batch_end_time = time.time()
            net_time = self.batch_end_time - self.batch_start_time
            #print(f"net_time:  {net_time:.2f} seconds, step={trainer.global_step}")
            pl_module.log("net_time", net_time, prog_bar=True,logger=False)



class TrainBatchTimeLogger(Callback):
    def __init__(self):
        super().__init__()
        self.batch_start_time = time.time()
        self.batch_end_time = time.time()

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        # 记录 batch 开始时间
        if trainer.is_global_zero:
            self.batch_start_time = time.time()
            data_time = self.batch_start_time - self.batch_end_time
            #print(f"data_time: {(self.batch_start_time - self.batch_end_time):.2f}seconds, step={trainer.global_step}")
            pl_module.log("data_time", data_time, prog_bar=True,logger=False)


    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # 计算 batch 处理时间
        if trainer.is_global_zero:
            self.batch_end_time = time.time()
            net_time = self.batch_end_time - self.batch_start_time
            #print(f"net_time:  {net_time:.2f} seconds, step={trainer.global_step}")
            pl_module.log("net_time", net_time, prog_bar=True,logger=False)



def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--task",
        type=str,
        default="ymzx_data_process",
        required=True,
        choices=["ymzx_data_process", "train"],
        help="Task. `data_process` or `train`.",
    )
    parser.add_argument(
        '--model_type',
        type=str,
        default='wan'
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        required=False,
        help="The path of the Dataset.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./",
        help="Path to save the model.",
    )
    parser.add_argument(
        "--text_encoder_path",
        type=str,
        default=None,
        help="Path of text encoder.",
    )
    parser.add_argument(
        "--image_encoder_path",
        type=str,
        default=None,
        help="Path of image encoder.",
    )
    parser.add_argument(
        "--vae_path",
        type=str,
        default=None,
        help="Path of VAE.",
    )
    parser.add_argument(
        "--dit_path",
        type=str,
        default=None,
        help="Path of DiT.",
    )
    parser.add_argument(
        "--bs",
        type=int,
        default=1,
        help="batch size",
    )
    parser.add_argument(
        "--tiled",
        default=False,
        action="store_true",
        help="Whether enable tile encode in VAE. This option can reduce VRAM required.",
    )
    parser.add_argument(
        "--tile_size_height",
        type=int,
        default=34,
        help="Tile size (height) in VAE.",
    )
    parser.add_argument(
        "--tile_size_width",
        type=int,
        default=34,
        help="Tile size (width) in VAE.",
    )
    parser.add_argument(
        "--tile_stride_height",
        type=int,
        default=18,
        help="Tile stride (height) in VAE.",
    )
    parser.add_argument(
        "--tile_stride_width",
        type=int,
        default=16,
        help="Tile stride (width) in VAE.",
    )
    parser.add_argument(
        "--steps_per_epoch",
        type=int,
        default=500,
        help="Number of steps per epoch.",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=81,
        help="Number of frames.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Image height.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=832,
        help="Image width.",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=1,
        help="Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="Learning rate.",
    )
    parser.add_argument(
        "--accumulate_grad_batches",
        type=int,
        default=1,
        help="The number of batches in gradient accumulation.",
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=1,
        help="Number of epochs.",
    )
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        default="q,k,v,o,ffn.0,ffn.2",
        help="Layers with LoRA modules.",
    )
    parser.add_argument(
        "--init_lora_weights",
        type=str,
        default="kaiming",
        choices=["gaussian", "kaiming"],
        help="The initializing method of LoRA weight.",
    )
    parser.add_argument(
        "--training_strategy",
        type=str,
        default="auto",
        choices=["auto", "deepspeed_stage_1", "deepspeed_stage_2", "deepspeed_stage_3"],
        help="Training strategy",
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=4,
        help="The dimension of the LoRA update matrices.",
    )
    parser.add_argument(
        "--lora_alpha",
        type=float,
        default=4.0,
        help="The weight of the LoRA update matrices.",
    )
    parser.add_argument(
        "--use_gradient_checkpointing",
        default=False,
        action="store_true",
        help="Whether to use gradient checkpointing.",
    )
    parser.add_argument(
        "--use_gradient_checkpointing_offload",
        default=False,
        action="store_true",
        help="Whether to use gradient checkpointing offload.",
    )
    parser.add_argument(
        "--train_architecture",
        type=str,
        default="lora",
        choices=["lora", "full"],
        help="Model structure to train. LoRA training or full training.",
    )
    parser.add_argument(
        "--pretrained_lora_path",
        type=str,
        default=None,
        help="Pretrained LoRA path. Required if the training is resumed.",
    )
    parser.add_argument(
        "--use_swanlab",
        default=False,
        action="store_true",
        help="Whether to use SwanLab logger.",
    )
    parser.add_argument(
        "--swanlab_mode",
        default='local',
        help="SwanLab mode (cloud or local).",
    )
    parser.add_argument('--batch_size', 
            type=int,
            default=1,
            help='input batch size for training')
    parser.add_argument('--log_iters',
            type=int,
            default=200)
    args = parser.parse_args()
    return args

 
    
def ymzx_data_process(args):
    dataset = TextVideoYmzxDataset(
        max_num_frames=args.num_frames,
        frame_interval=1,
        num_frames=args.num_frames,
        height=args.height,
        width=args.width,
        is_i2v=args.image_encoder_path is not None
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=False,
        batch_size=args.bs,
        num_workers=args.dataloader_num_workers,
        pin_memory=True
    )
    model = LightningModelForYmzxDataProcess(
        text_encoder_path=args.text_encoder_path,
        image_encoder_path=args.image_encoder_path,
        vae_path=args.vae_path,
        tiled=args.tiled,
        tile_size=(args.tile_size_height, args.tile_size_width),
        tile_stride=(args.tile_stride_height, args.tile_stride_width),
    )

    trainer = pl.Trainer(
        callbacks=[BatchTimeLogger()],
        accelerator="gpu",
        devices="auto",
        default_root_dir=args.output_path,
        
    )
    trainer.test(model, dataloader)
        


def train(args):
    dataset = YmzxTensorDataset(
        args.dataset_path,
        steps_per_epoch=args.steps_per_epoch,
        num_maps=42
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=True,
        batch_size=args.batch_size,
        num_workers=args.dataloader_num_workers,
        pin_memory=True
    )
    if args.model_type == 'wan':
        model = LightningModelForTrain(
            dit_path=args.dit_path,
            learning_rate=args.learning_rate,
            train_architecture=args.train_architecture,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_target_modules=args.lora_target_modules,
            init_lora_weights=args.init_lora_weights,
            use_gradient_checkpointing=args.use_gradient_checkpointing,
            use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
            pretrained_lora_path=args.pretrained_lora_path,
        )
    elif args.model_type == 'wan_action':
        model = WanAction_1(
            dit_path=args.dit_path,
            learning_rate=args.learning_rate,
            train_architecture=args.train_architecture,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_target_modules=args.lora_target_modules,
            init_lora_weights=args.init_lora_weights,
            use_gradient_checkpointing=args.use_gradient_checkpointing,
            use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
            pretrained_lora_path=args.pretrained_lora_path,
        )
    elif args.model_type == 'wan_action_block':
        model = WanActionLg()
    else:
        raise NotImplementedError
    if args.use_swanlab:
        from swanlab.integration.pytorch_lightning import SwanLabLogger
        swanlab_config = {"UPPERFRAMEWORK": "DiffSynth-Studio"}
        swanlab_config.update(vars(args))
        swanlab_logger = SwanLabLogger(
            project="wan", 
            name="wan_action_1",
            config=swanlab_config,
            mode=args.swanlab_mode,
            logdir=os.path.join(args.output_path, "swanlog"),
        )
        logger = [swanlab_logger]
    else:
        logger = None 
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="gpu",
        devices="auto",
        precision="bf16",
        strategy=args.training_strategy,
        default_root_dir=args.output_path,
        accumulate_grad_batches=args.accumulate_grad_batches,
        callbacks=[
            pl.pytorch.callbacks.ModelCheckpoint(
                save_top_k=-1,
                every_n_train_steps=args.log_iters,  # 新增步数间隔配置
                dirpath=os.path.join(args.output_path, "checkpoints")
            ),
            TrainBatchTimeLogger()
        ],
        logger=logger,
        log_every_n_steps=1,
    )
    trainer.fit(model, dataloader)


if __name__ == '__main__':
    args = parse_args()
    if args.task == "ymzx_data_process":
        ymzx_data_process(args)
    elif args.task == "train":
        train(args)
