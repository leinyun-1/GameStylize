from diffsynth.models.wan_video_dit import WanModel,sinusoidal_embedding_1d, RMSNorm, flash_attention
from diffusers.models.embeddings import PixArtAlphaTextProjection
import torch 
import torch.nn as nn 
import torch.nn.functional as F
from einops import rearrange

def load_lora(model, state_dict_lora, lora_prefix="", alpha=1.0, model_resource=""):
    state_dict_model = model.state_dict()
    device, dtype, computation_device, computation_dtype = fetch_device_and_dtype(state_dict_model)
    lora_name_dict = get_name_dict(state_dict_lora)
    for name in lora_name_dict:
        weight_up = state_dict_lora[lora_name_dict[name][0]].to(device=computation_device, dtype=computation_dtype)
        weight_down = state_dict_lora[lora_name_dict[name][1]].to(device=computation_device, dtype=computation_dtype)
        if len(weight_up.shape) == 4:
            weight_up = weight_up.squeeze(3).squeeze(2)
            weight_down = weight_down.squeeze(3).squeeze(2)
            weight_lora = alpha * torch.mm(weight_up, weight_down).unsqueeze(2).unsqueeze(3)
        else:
            weight_lora = alpha * torch.mm(weight_up, weight_down)
        weight_model = state_dict_model[name].to(device=computation_device, dtype=computation_dtype)
        weight_patched = weight_model + weight_lora
        state_dict_model[name] = weight_patched.to(device=device, dtype=dtype)
    print(f"    {len(lora_name_dict)} tensors are updated.")
    model.load_state_dict(state_dict_model)

def get_name_dict(lora_state_dict):
    lora_name_dict = {}
    for key in lora_state_dict:
        if ".lora_B." not in key:
            continue
        keys = key.split(".")
        if len(keys) > keys.index("lora_B") + 2:
            keys.pop(keys.index("lora_B") + 1)
        keys.pop(keys.index("lora_B"))
        if keys[0] == "diffusion_model":
            keys.pop(0)
        target_name = ".".join(keys)
        lora_name_dict[target_name] = (key, key.replace(".lora_B.", ".lora_A."))
    return lora_name_dict

def fetch_device_and_dtype(state_dict):
    device, dtype = None, None
    for name, param in state_dict.items():
        device, dtype = param.device, param.dtype
        break
    computation_device = device
    computation_dtype = dtype
    if computation_device == torch.device("cpu"):
        if torch.cuda.is_available():
            computation_device = torch.device("cuda")
    if computation_dtype == torch.float8_e4m3fn:
        computation_dtype = torch.float32
    return device, dtype, computation_device, computation_dtype


def get_dit():
    config = dict(
                 in_dim=16,
                 dim=1536,
                 out_dim=16,
                 ffn_dim=8960,
                 freq_dim=256,
                 text_dim=4096,
                 num_heads=12,
                 num_layers=30,
                 eps=1e-6,
                 patch_size=(1, 2, 2),
                 has_image_input=False)

    model = WanModel(**config)
    model.requires_grad_(False)
    model.eval()
    return model


def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, num_heads: int, block_mask):
    q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
    k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
    v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
    x = F.scaled_dot_product_attention(q, k, v, block_mask)
    x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
    return x 


class CrossAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)
            
    def forward(self, x: torch.Tensor, y: torch.Tensor, block_mask=None):
        ctx = y
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(ctx))
        v = self.v(ctx)
        x = attention(q, k, v, self.num_heads, block_mask)
        return self.o(x)

class ActionBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, ffn_dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.norm = nn.LayerNorm(dim, eps=eps)
        self.cross_attn = CrossAttention(dim, num_heads, eps)
        self.ffn = nn.Sequential(nn.Linear(dim, ffn_dim), nn.GELU(
            approximate='tanh'), nn.Linear(ffn_dim, dim))
        
    def forward(self, x: torch.Tensor, context: torch.Tensor, block_mask):
        x = x + self.cross_attn(x, context, block_mask)        
        x = self.norm(x)
        x = x + self.ffn(x)
        return x

    # def forward(self, x: torch.Tensor, context: torch.Tensor, block_mask): # 0620 及之前
    #     x = self.norm(x)
    #     x = self.cross_attn(x, context, block_mask)        
    #     x = x + self.ffn(x)
    #     return x



class WanAction(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = get_dit()
        self.action_projection = PixArtAlphaTextProjection(in_features=20*4,hidden_size=self.model.dim,act_fn='silu')
        self.action_block_freq = 4
        self.action_block = nn.ModuleList([
            ActionBlock(self.model.dim, self.model.num_heads, self.model.ffn_dim, self.model.eps)
            for _ in range(self.model.num_layers//self.action_block_freq)])
        
    def forward(self, x, timestep, context, action, clip_feature=None, y=None,  use_gradient_checkpointing=True) -> torch.Tensor:
        t = self.model.time_embedding(sinusoidal_embedding_1d(self.model.freq_dim, timestep))
        t_mod = self.model.time_projection(t).unflatten(1, (6, self.model.dim))

        context = self.model.text_embedding(context)
        
        if self.model.has_image_input and clip_feature is not None and y is not None:
            x = torch.cat([x, y], dim=1)  # (b, c_x + c_y, f, h, w)
            clip_embdding = self.model.img_emb(clip_feature)
            context = torch.cat([clip_embdding, context], dim=1)
        
        x, (f, h, w) = self.model.patchify(x)
        
        freqs = torch.cat([
            self.model.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            self.model.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            self.model.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ], dim=-1).reshape(f * h * w, 1, -1).to(x.device)
    
        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs)
            return custom_forward
        
        block_mask = self.get_block_mask(x.shape[0],f,h,w, x.device,x.dtype)
        action_emb = self.action_projection(action) # (b f 20*4) -> (b f 2048)

        for block_id, block in enumerate(self.model.blocks):
            if use_gradient_checkpointing:
                x = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        x, context, t_mod, freqs,
                        use_reentrant=False,)
                if (block_id+1) % self.action_block_freq == 0:                    
                    x = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(self.action_block[block_id // self.action_block_freq]), x, action_emb, block_mask)
            else:
                x = block(x, context, t_mod, freqs)
                if (block_id+1) % self.action_block_freq == 0:
                    x = self.action_block[block_id // self.action_block_freq](x, action_emb, block_mask)                


        x = self.model.head(x, t)
        x = self.model.unpatchify(x, (f, h, w))
        return x

    def generate(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)
    
    def get_block_mask(self, b, f, h, w, device, dtype): # mask 就是 1,len_q,len_k 不需要b
        mask = torch.eye(f)
        pre_mask = torch.eye(f-1)
        #mask[1:,:-1] = mask[1:,:-1] + pre_mask #错误的，多看到未来的一帧，0617改正
        mask[:-1,1:] = mask[:-1,1:] + pre_mask 
        mask = mask.unsqueeze(-1).repeat(1, 1, h*w)
        mask = mask.reshape(f,f*h*w).permute(1,0).unsqueeze(0).unsqueeze(0).to(device).to(dtype) > 0 # 1 1 32760 21  [b,h,q_len,k_len]
        return mask 

if __name__ == "__main__":
    import time 
    from safetensors.torch import load_file
    import os
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    wan_state_dict = load_file('models/Wan-AI/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors',device='cpu')

    device = torch.device("cuda:0")
    with torch.device(device):
        model = WanAction()

        #model.model.load_state_dict(wan_state_dict,strict=True)
        #load_lora(model.model,torch.load('exp_out/0531/checkpoints/epoch=0-step=400.ckpt',map_location='cpu'))
        model = model.to(torch.bfloat16)
        #compile_model = torch.compile(WanAction().to(torch.bfloat16))
        bs = 1
        x = torch.randn(bs, 16, 5, 60, 104).to(torch.bfloat16)
        timestep = torch.tensor([0]).to(torch.bfloat16)
        timestep = torch.randint(0,1000,(bs,5)).to(torch.bfloat16)
        context = torch.randn(bs, 512, 4096).to(torch.bfloat16)
        action = torch.randn(bs, 21, 20*4).to(torch.bfloat16)


        output = model(x, timestep.flatten(), context, action, y=None, clip_feature=None, use_gradient_checkpointing=True)

        # s_t = time.time()
        # for _ in range(10):
        #     output = model(x, timestep, context, action, y=None, clip_feature=None, use_gradient_checkpointing=True)
        # print(f"10 ff takes: {time.time() - s_t:.4f} seconds")
        
        # s_t = time.time()
        # for _ in range(10):
        #     output = compile_model(x, timestep, context, action, y=None, clip_feature=None, use_gradient_checkpointing=True)
        # print(f"10 compile ff takes: {time.time() - s_t:.4f} seconds")
        
    print(output.shape)  # Should match the expected output shape