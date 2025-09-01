import torch, os, imageio, argparse
from torchvision.transforms import v2
from einops import rearrange
import lightning as pl
import pandas as pd
from diffsynth import WanVideoPipelineNonunipc, ModelManager, load_state_dict
from peft import LoraConfig, inject_adapter_in_model
import torchvision
from PIL import Image
import numpy as np
from csgo_dataset import CsgoLatentDataset
from doom_dataset import DoomLatentDataset
from fps_dataset import LatentDataset
DATASET_MAP = {
    'csgo': CsgoLatentDataset,
    'doom': DoomLatentDataset,
    'pubg': LatentDataset,  
    'varolent': LatentDataset,
    'delta': LatentDataset,
    'ngr': LatentDataset,
    'thuman2.1': LatentDataset,
}


class LightningModelForTrain(pl.LightningModule):
    def __init__(self,args):
        super().__init__()
        dit_path='/root/leinyu/model/Wan2.1-Fun-1.3B-Control/diffusion_pytorch_model.safetensors'
        learning_rate=args.learning_rate
        train_architecture=args.train_architecture
        lora_rank=args.lora_rank
        lora_alpha=args.lora_rank
        lora_target_modules=args.lora_target_modules
        init_lora_weights='kaiming'
        use_gradient_checkpointing=True
        use_gradient_checkpointing_offload=False
        pretrained_lora_path=args.pretrained_lora_path
        self.args = args 
        
        model_manager = ModelManager(torch_dtype=torch.bfloat16, device="cpu")
        if os.path.isfile(dit_path):
            model_manager.load_models([dit_path])
        else:
            dit_path = dit_path.split(",")
            model_manager.load_models([dit_path])
        
        self.pipe = WanVideoPipelineNonunipc.from_model_manager(model_manager)
        self.pipe.scheduler.set_timesteps(1000, training=True)
        self.freeze_parameters()
        if args.lora_path is not None:
            from diffsynth.models.lora import GeneralLoRAFromPeft
            lora_state_dict = torch.load(args.lora_path,map_location="cpu")
            GeneralLoRAFromPeft().load(self.pipe.dit,lora_state_dict) 


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

        self.prompt_ebd = torch.load('/root/leinyu/data/csgo/prompt_embs_10_512_4096.pth',map_location='cpu',weights_only=True) # 10 512 4096
        
        
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
    
    def prepare_controlnet_kwargs(self,control_latents,f,h,w):
        bsz = control_latents.shape[0]
        clip_feature = torch.zeros((bsz, 257, 1280), dtype=torch.bfloat16, device=self.device)
        y = torch.zeros((bsz, 16, f, h, w), dtype=torch.bfloat16, device=self.device)
        y = torch.concat([control_latents, y], dim=1)
        return {"clip_feature": clip_feature, "y": y}


    def training_step(self, batch, batch_idx):
        # Data
        if self.args.dataset == 'thuman2.1':
            latents = batch["normal_latents"].to(self.device)    
        else:
            latents = batch["latents"].to(self.device)
        if self.args.dataset == 'thuman2.1':
            control_latents = batch['latents'].to(self.device)
        else:
            control_latents = batch["control_latents"].to(self.device)
        bsz,f = latents.shape[0],latents.shape[2]
        #t = np.random.randint(0,f-self.args.latent_len+1)
        t = 0
        latents = latents[:,:,t:t+self.args.latent_len]  # 让历史帧不一定是图片latent，大概率是视频latent，推理时递归更加方便，但是 ll4 + hg2 效果有限。
        control_latents = control_latents[:,:,t:t+self.args.latent_len]
        f,h,w = latents.shape[2:]


        # text_id = torch.randint(0, self.prompt_ebd.shape[0], (bsz,))
        # text_embeds = self.prompt_ebd.to(device=self.device, dtype=torch.bfloat16)[text_id]  # bs 512 4096
        prompt_emb = {}
        #prompt_emb["context"] = text_embeds
        prompt_emb["context"] = batch['prompt_emb'].to(self.device,dtype=self.pipe.torch_dtype)
        image_emb = self.prepare_controlnet_kwargs(control_latents,f,h,w)


        # Loss
        self.pipe.device = self.device
        if not self.args.diff_forcing:
            if self.args.history_guide:
                zero_t_len = np.random.randint(0,self.args.history_guide_len+1)
                #zero_t_len = self.args.history_guide_len
            else:
                zero_t_len = 0
            noise = torch.randn_like(latents)
            timestep_id = torch.randint(0, self.pipe.scheduler.num_train_timesteps, (bsz,))
            timestep = self.pipe.scheduler.timesteps[timestep_id].to(dtype=self.pipe.torch_dtype, device=self.pipe.device)
            extra_input = self.pipe.prepare_extra_input(latents)
            noisy_latents = latents.clone()
            noisy_latents[:,:,zero_t_len:] = self.pipe.scheduler.add_noise(latents[:,:,zero_t_len:], noise[:,:,zero_t_len:], timestep)
            training_target = self.pipe.scheduler.training_target(latents, noise, timestep)
        else:
            latents = latents.permute(0,2,1,3,4) # b f c h w 
            noise = torch.randn_like(latents)
            timestep_id = torch.randint(0, self.pipe.scheduler.num_train_timesteps, (bsz*f,))
            timestep = self.pipe.scheduler.timesteps[timestep_id].to(dtype=self.pipe.torch_dtype, device=self.pipe.device)
            extra_input = self.pipe.prepare_extra_input(latents)
            noisy_latents = self.pipe.scheduler.add_noise(latents.flatten(0,1), noise.flatten(0,1), timestep)
            training_target = self.pipe.scheduler.training_target(latents, noise, timestep)

            training_target = training_target.permute(0,2,1,3,4)
            noisy_latents = noisy_latents.unflatten(0,(bsz,f)).permute(0,2,1,3,4)
            timestep = timestep.unflatten(0,(bsz,f))

        # Compute loss
        noise_pred = self.pipe.denoising_model()(
            noisy_latents, timestep=timestep, **prompt_emb, **extra_input, **image_emb,
            use_gradient_checkpointing=self.use_gradient_checkpointing,
            use_gradient_checkpointing_offload=self.use_gradient_checkpointing_offload
        )
        if self.args.diff_forcing:
            noise_pred = noise_pred.permute(0,2,1,3,4).flatten(0,1) # bf c h w 
            training_target = training_target.permute(0,2,1,3,4).flatten(0,1) 
            timestep = timestep.flatten(0,1)
            loss = torch.nn.functional.mse_loss(noise_pred.float(), training_target.float(),reduction='none').mean(dim=tuple(range(1, noise_pred.dim()))) # 保留bs维度
            loss = torch.mean( loss * self.pipe.scheduler.training_weight(timestep).to(loss) )# 忽略了这句话，这个weight很重
        else:  
            loss = torch.nn.functional.mse_loss(noise_pred[:,:,zero_t_len:].float(), training_target[:,:,zero_t_len:].float(),reduction='none').mean(dim=tuple(range(1, noise_pred.dim()))) # 保留bs维度
            loss = torch.mean( loss * self.pipe.scheduler.training_weight(timestep).to(loss) )# 忽略了这句话，这个weight很重

        # Record log
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
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


def train(args):
    #dataset = CsgoLatentDataset()
    #dataset = DoomLatentDataset()
    dataset_class = DATASET_MAP.get(args.dataset)
    dataset = dataset_class(root=args.dataset_root)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=False,
        batch_size=args.batch_size,
        num_workers=8,
        pin_memory=True 
    )
    model = LightningModelForTrain(args)
    
    # from swanlab.integration.pytorch_lightning import SwanLabLogger
    # swanlab_config = {}
    # swanlab_config.update(vars(args))
    # swanlab_logger = SwanLabLogger(
    #     project="wan", 
    #     name="wan_doom_lora",
    #     config=swanlab_config,
    #     mode='local',
    #     logdir=os.path.join('exp_out/train_exp/'+args.name, "swanlog"),
    # )
    # logger = [swanlab_logger]


    # 替换为TensorBoard日志记录器
    from lightning.pytorch.loggers import TensorBoardLogger
    # 创建TensorBoardLogger实例
    tensorboard_logger = TensorBoardLogger(
        save_dir=os.path.join('exp_out/train_exp', args.name),
        name='tensorboard_logs'  # 日志子目录名称
    )
    logger = [tensorboard_logger]  # 保持logger为列表形式


    from lightning.pytorch.callbacks import ModelCheckpoint
    checkpoint_callback = ModelCheckpoint(
        dirpath='exp_out/train_exp/'+args.name+'/lightning_lora_ckpts',
        filename="lora-{epoch:02d}-{step:06d}",
        save_top_k=-1,
        every_n_train_steps=500,
        save_last=True,
    )
    callbacks = [checkpoint_callback]

    trainer = pl.Trainer(
        max_epochs=1000,
        accelerator="gpu",
        devices=args.devices,
        precision="bf16",
        accumulate_grad_batches=args.accumulate_grad_batches,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=1
    )
    trainer.fit(model, dataloader)


if __name__ == "__main__":
    from types import SimpleNamespace
    config = {
    "name": '0826_wan_fun_lora_thuman2.1_i2n',
    'train_architecture': 'lora',
    'diff_forcing': False,
    'history_guide': False,
    'history_guide_len': 1,
    'learning_rate': 1e-4,
    'dataset': 'thuman2.1',
    'dataset_root': '/root/leinyu/data/thuman2.1/Thuman2.1_norm_render_1/latent',    
    'latent_len': 21,
    'devices': [1,2,3,4,5,6,7],
    'lora_rank': 128,
    'batch_size': 2,
    'accumulate_grad_batches': 1,
    'lora_target_modules': "q,k,v,o,ffn.0,ffn.2",
    'pretrained_lora_path': None,
    'lora_path': None #'exp_output/train_exp/0721_wan_lora_csgo/lightning_lora_ckpts/lora-epoch=28-step=001500.ckpt'
    }
    cfg = SimpleNamespace(**config)

    train(cfg)


