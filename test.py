from train_wan_t2v_ymzx import TextVideoYmzxDataset,YmzxTensorDataset
import numpy as np 
import os 
import shutil 
import torch
import torch.nn as nn
from safetensors.torch import load_file
from modelscope import snapshot_download, dataset_snapshot_download
import cv2
action_name = ["w", "a", "s", "d", "Key.up", "Key.left", "Key.down", "Key.right", "Key.space", "z", "x", "c", "Stop"]
action_list = [2, 3, 4, 1, 9, 10, 11, 12, 5, 7, 13, 8, 6]
turn_action_list=[10,  11,  12,  13,     14,         3,         15,          4,          16,       17,  18,   19, 6]

ACTION_MAPPINGS = {}
TURN_ACTION_MAPPINGS = {}
for i in range(len(action_list)):
    ACTION_MAPPINGS[action_list[i]] = action_name[i]
    TURN_ACTION_MAPPINGS[turn_action_list[i]] = action_name[i]

def get_raw_actions(name):
    action_path = '/mnt/kaiwu-group-y1-sh-hdd/chesterlv/datasets/394879_qwen25vl_32b_caption/actions/'
    action_path = action_path + name + '.npy'
    actions = np.load(action_path)[::2]
    actions = actions.reshape(-1)
    
    return actions 

def draw_images(frame, word):
    # print("frame:", frame.shape, frame.dtype, np.unique(frame))
    return cv2.putText(frame.copy(), word, (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

def write_actions(frames, actions):
    return [draw_images(np.array(frames[i]).astype(np.uint8), ACTION_MAPPINGS[int(actions[i])] if int(actions[i]) in ACTION_MAPPINGS else "TURN." + TURN_ACTION_MAPPINGS[int(actions[i])]) for i in range(actions.shape[0])]

def get_action_onehot(actions, action_num=20):

    # Action to one-hot:
    action_batchsize = actions.shape[0]
    action_spacesize = action_num
    action_onehots = np.zeros((action_batchsize, action_spacesize))
    action_onehots[[i for i in range(action_batchsize)], np.array(actions).astype(np.int32)] = 1.

    action_onehots = action_onehots.reshape(action_batchsize // 4, 4, action_spacesize).reshape(1, action_batchsize // 4, 4 * action_spacesize)

    return action_onehots

def get_actions(action_file, padding_lengh=81):
    if type(action_file) == str:
        actions = np.load(action_file).astype(np.float32)
    elif type(action_file) == list:
        actions = np.array(action_file)
    actions = actions[::2]
    action_len = actions.shape[0]
    if padding_lengh > action_len:
        actions = np.concatenate([actions, [actions[-1]] * (padding_lengh - action_len)]).reshape(-1)
    elif padding_lengh < action_len:  
        actions = actions[:padding_lengh]
    actions = np.concatenate([[actions[0]] * 3, actions]).reshape(-1)

    return actions

def get_raw_videos(name):
    import imageio
    from PIL import Image
    video_path = '/mnt/kaiwu-group-y1-sh-hdd/chesterlv/datasets/394879_qwen25vl_32b_caption/videos/'
    video_path = video_path + name + '.mp4'

    reader = imageio.get_reader(video_path)
    
    frames = []
    first_frame = None
    for frame_id in range(81):
        frame = reader.get_data(frame_id)
        frames.append(frame)
    reader.close()

    frames = np.stack(frames,axis=0)

    return frames



def test_text_video_dataset():
    dataset = TextVideoYmzxDataset(
        base_path='/mnt/kaiwu-group-y1-sh-hdd/chesterlv/datasets/394879_qwen25vl_32b_caption/',
        max_num_frames=81, 
        frame_interval=1,
        num_frames=81, 
        height=768, 
        width=1360
    )

    print(f"Dataset length: {len(dataset)}")
    dataset_size = len(dataset)

    #data = dataset[0]
    # for item in data:
    #     if item == "video":
    #         print(f"Video shape: {data[item].shape}")
    #     elif item == "text":
    #         print(f"Text: {data[item]}")
    #     elif item == "action":
    #         print(f"{item}: {data[item][:2]}")
    #     else:
    #         print(f"{item}: {data[item]}")

    root_dir = '/mnt/kaiwu-group-y1-sh-hdd/chesterlv/datasets/394879_qwen25vl_32b_caption/videos'
    dest_dir = '/mnt/aigc_cq/private/leinyu/data/ymzx/mini/'
    random_ind = np.random.randint(0,dataset_size,(100))
    prompt_txt = '/mnt/aigc_cq/private/leinyu/data/ymzx/mini/prompts.txt'
    for data_id in random_ind:
        data = dataset[data_id]
        path = data['path']
        text = data['text']
        video_path = os.path.join(root_dir,path + '.mp4')
        destination_path = os.path.join(dest_dir,'_'.join(path.split('/')) + '.mp4')
        shutil.copy(video_path,destination_path)
        text = text + '{"reference_path":"' + destination_path + '"}'

        with open(prompt_txt,'a') as f:
            f.write(text + '\n')  

def test_make_latent():
    pass 

def test_latent_dataset():
    import torch
    from tqdm import tqdm
    dataset = YmzxTensorDataset()
    print(f"Dataset length: {len(dataset)}")
    for i in tqdm(range(500)):
        data = dataset[i]
        for item in data:
            if item == "latents":
                if data[item].shape != torch.Size([16, 21, 60, 104]) :
                    print(f"latents shape: {data[item].shape}")
            elif item == "prompt_emb":
                if data[item]['context'].shape != torch.Size([1,512,4096]):
                    print(f"Text_ebd shape: {data[item]['context'].shape}")
            else:
                continue
                print(f"{item}: {data[item]}")

def test_latent():
    import torch 
    tensor_path = '/mnt/aigc_cq/private/leinyu/data/ymzx/latent/60206/616765710200000123_6.tensors.pth'
    data = torch.load(tensor_path,weights_only=True, map_location="cpu")
    print(data['latents'].shape)
    print(data['prompt_emb']['context'].shape)

def test_lora_wan1_3b():
    import torch
    import imageio
    from diffsynth import ModelManager, WanVideoPipeline, WanVideoPipelineNonunipc, save_video, VideoData

    device = torch.device("cuda:0")
    model_manager = ModelManager(torch_dtype=torch.bfloat16, device=device)
    model_manager.load_models([
        "models/Wan-AI/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors",
        "models/Wan-AI/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth",
        "models/Wan-AI/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth",
    ])
    lora_path = "exp_out/train_exp/0626_wanaction_1_scratch/checkpoints/epoch=2-step=36000.ckpt"
    #model_manager.load_lora(lora_path, lora_alpha=1) 
    pipe = WanVideoPipeline.from_model_manager(model_manager, device=device)
    pipe.dit.has_action_input = True
    pipe.dit.set_action_projection(4*20)
    pipe.dit.action_projection.to(device=device,dtype=torch.bfloat16)
    pipe.dit.set_ar_attention()


    indata = 0
    cfg = 5
    shift = 8
    infer_steps = 30
    save_dir = 'exp_out/infer_exp/wanaction/0626_wanaction_1/36000_outdata_30_c5_s8/'
    os.makedirs(save_dir, exist_ok=True)   
    #pipe.enable_vram_management(num_persistent_param_in_dit=None)
    from diffsynth.models.lora import GeneralLoRAFromPeft
    lora_state_dict = torch.load(lora_path,map_location="cpu")
    GeneralLoRAFromPeft().load(pipe.dit,lora_state_dict)  # 测试这种注入方式到底对不对
    # lora_state_dict = load_file('exp_out/lora_weights/Wan21_CausVid_bidirect2_T2V_1_3B_lora_rank32.safetensors')
    # apply_lora_and_diff(pipe.dit,lora_state_dict,alpha=1)
    
    # lora_state_dict = torch.load(lora_path,map_location="cpu")
    model_state_dict = pipe.dit.state_dict()
    i = 0
    for k, v in lora_state_dict.items():
        if "action_projection" in k and k in model_state_dict:
            model_state_dict[k] = lora_state_dict[k]
            i += 1
    print(f"{i} tensors are updated.")
    pipe.dit.load_state_dict(model_state_dict)

    
    negative_prompt = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
    text_from_dataset_60206_614664550200000123_2 = ["Environmental basics:The scene is set in a vibrant, cartoonish industrial port area with bright blue water and wooden platforms. The environment features colorful crates, barrels, and shipping containers stacked around the area. A large wooden barrel is prominently placed on one of the platforms, and there are ramps and pathways connecting different sections. The background includes distant buildings and a clear sky, adding to the lively atmosphere.Main character:A small, anthropomorphic cat character is seen running across the wooden platforms and interacting with the environment. The cat appears agile and is navigating through the industrial setting, moving from one platform to another.Game mechanisms:The primary mechanism involves a large wooden barrel that rotates slowly on its axis. The barrel has a face drawn on it and is positioned on a raised platform. As the cat approaches, the barrel continues its rotation, creating a dynamic interaction point. The cat must time its movements carefully to avoid being hit by the barrel. Additionally, there are ramps and pathways that the cat uses to move between different levels of the environment, indicating a focus on navigation and timing.Weather and lighting:The weather is clear and sunny, with bright daylight illuminating the entire scene. The lighting is vivid, enhancing the colorful and cheerful aesthetic of the environment."]
    texts = ["a cat character is walking into the home",
             'a man is playing the video',
             '...',
            'A cartoon character is playing in a cartoon game scene.',
            'A stylish woman walks down a Tokyo street filled with warm glowing neon and animated city signage. She wears a black leather jacket, a long red dress, and black boots, and carries a black purse. She wears sunglasses and red lipstick. She walks confidently and casually. The street is damp and reflective, creating a mirror effect of the colorful lights. Many pedestrians walk about.',
            'Several giant wooly mammoths approach treading through a snowy meadow, their long wooly fur lightly blows in the wind as they walk, snow covered trees and dramatic snow capped mountains in the distance, mid afternoon light with wispy clouds and a sun high in the distance creates a warm glow, the low camera view is stunning capturing the large furry mammal with beautiful photography, depth of field.',
            'A movie trailer featuring the adventures of the 30 year old space man wearing a red wool knitted motorcycle helmet, blue sky, salt desert, cinematic style, shot on 35mm film, vivid colors.'
            ]
    texts_mixcaption_dataset = [
                    #'the character is playing in the rainbow',
                    #'the character is playing in the desert',
                    'A branching bamboo forest rises through vine bridges, mossy stump platforms, waterfall drops, hollow trunk networks, and leaf-littered pavilions, accessible by swinging vines, tilting bamboo rafts, breakable root walls, and water-wheel lifts. Verdant bamboo green, mist, weathered wood, watercolor wash. cool fog, weak sunlight. The character continues heading to the next stage.',
                    "The scene is set in a vibrant, colorful arena with a bright green grassy field. The central area features a large, circular blue track with glowing patterns and symbols, surrounded by a low wall. In the background, there are structures resembling bleachers with banners displaying text, and various decorative elements such as palm trees and colorful blocks. The setting appears to be an outdoor stadium designed for a playful, competitive event. A small, animated husky-like character is running along the edge of the blue track, moving forward with a determined gait. The character's fur is gray and white, and it has a playful expression as it navigates the course. The blue track has a conveyor belt mechanism that moves continuously, assisting the character's forward motion. The track surface is adorned with glowing symbols and patterns, possibly indicating special zones or power-ups. There are also yellow coins scattered along the path, which the character can collect. The surrounding walls and structures remain stationary, providing a boundary for the track. A rotating wheel-like structure is visible in the distance, suggesting additional mechanical elements that may interact with the character. The lighting is bright and sunny, casting clear shadows on the ground, indicating a daytime setting. The weather appears clear and pleasant, enhancing the vivid colors of the environment",
                    'The scene is set in a brightly colored, indoor arena with a vibrant green floor. The arena features a circular track with blue and white markings, surrounded by orange and yellow walls. There are various colorful structures, including a large blue platform with a star symbol and a green plant-like decoration. The background includes a palm tree silhouette and a section of the wall with a grid pattern. A small, animated character resembling a cat with gray fur and white markings is running across the green floor towards the center of the arena. The character moves steadily forward, leaving a visible shadow on the ground. The arena contains a conveyor belt system that moves along the circular track. The belt has a consistent speed and direction, guiding objects placed on it around the track. There are also stationary platforms and obstacles such as the blue star platform and the green plant structure, which serve as interactive points within the environment. The character interacts with these elements by running towards them, potentially triggering movements or changes in the environment. The lighting is bright and even, suggesting an indoor setting with artificial lighting. There are no weather effects present, and the environment appears clear and well-lit',
                    'A suspended botanical laboratory features floating terrarium platforms with bioluminescent flora in magenta and cyan. Transparent tubes connect dome-shaped greenhouses containing oversized carnivorous plants. The sky reveals a perpetual golden hour through stained glass atmospheric filters. An amphibious lemur wearing photosynthesis-enhanced goggles navigates using prehensile pollen trails. Its fur dynamically changes pigmentation to match crossed vegetation, retaining partial camouflage for three seconds after landing. Pollen collector stations trigger platform growth when activated, extending vine pathways. Venus flytrap sentries periodically snap shut, requiring timed evasion. The lemur deploys seed cluster bombs that temporarily neutralize aggressive flora, creating safe passage windows. Directional light beams penetrate through rotating prismatic crystals above, casting rainbow refraction patterns that migrate across surfaces. Humidity particles glow when intersecting light paths.',
                    "An inverted dessert dimension features floating shortcake platforms with strawberry syrup waterfalls. Custard clouds drift between crumbling biscuit cliffs dusted with powdered sugar. Marzipan structures extrude whipped cream support beams that gradually melt. A marshmallow-armored penguin utilizes an edible ice staff to freeze unstable surfaces. Each jump leaves temporary frosting footholds that dissolve after 2 seconds, usable by other players in multiplayer mode. Chocolate lava geysers periodically erupt beneath platforms, requiring predictive pathing. Gummy worm elastic bridges can be stretched between platforms using the ice staff's hook function. Random \"sugar rush\" events accelerate all movement speeds for 8 seconds. Warm ambient light mimics caramel glaze coating all surfaces. Sprinkles rain vertically during phase transitions, physically interacting with character movement",
                    "Environmental basics:The scene is set in a vibrant, cartoonish industrial port area with bright blue water and wooden platforms. The environment features colorful crates, barrels, and shipping containers stacked around the area. A large wooden barrel is prominently placed on one of the platforms, and there are ramps and pathways connecting different sections. The background includes distant buildings and a clear sky, adding to the lively atmosphere.Main character:A small, anthropomorphic cat character is seen running across the wooden platforms and interacting with the environment. The cat appears agile and is navigating through the industrial setting, moving from one platform to another.Game mechanisms:The primary mechanism involves a large wooden barrel that rotates slowly on its axis. The barrel has a face drawn on it and is positioned on a raised platform. As the cat approaches, the barrel continues its rotation, creating a dynamic interaction point. The cat must time its movements carefully to avoid being hit by the barrel. Additionally, there are ramps and pathways that the cat uses to move between different levels of the environment, indicating a focus on navigation and timing.Weather and lighting:The weather is clear and sunny, with bright daylight illuminating the entire scene. The lighting is vivid, enhancing the colorful and cheerful aesthetic of the environment.",
                    "Environmental basics:The scene is set in a circular, elevated track with a futuristic design. The track features a vibrant blue and green color scheme, with a central platform marked by geometric patterns and symbols. The outer edge of the track has a series of evenly spaced yellow markers and black cylindrical supports. The background is a clear, light blue sky, giving the impression of an open, high-altitude setting.Main character:A small, humanoid character with a dark silhouette is running along the inner edge of the circular track, maintaining balance while navigating the curved path.Game mechanisms:The track includes a conveyor belt-like mechanism that moves continuously, assisting the character's forward motion. There are rotating wheels embedded in the track surface, which spin independently and may affect the character's movement. Additionally, there are pistons or mechanical arms extending from the track's edge, moving up and down in a rhythmic pattern. These mechanical components interact dynamically with the character, requiring precise timing and navigation to avoid collisions.Weather and lighting:The lighting is bright and even, suggesting a sunny day with no visible clouds. The environment is well-lit, enhancing the vivid colors of the track and creating sharp shadows cast by the character and mechanical elements.",
                    "Environmental basics:The scene is set in a vibrant, circular arena with a bright green floor and blue walls adorned with colorful patterns and text. The central area features a large, detailed emblem with a star design, surrounded by concentric rings. The arena is enclosed by transparent barriers, giving it an open yet contained feel. The background shows a light, cloud-like texture, suggesting an indoor setting with a futuristic or playful theme.Main character:A small, animated husky dog is running along a curved path within the arena. The dog appears to be navigating the course with agility, maintaining balance as it moves forward.Game mechanisms:The arena includes a series of mechanical components such as conveyor belts that move along the curved path, assisting the character's progression. There are also rotating platforms embedded in the floor, which spin slowly, requiring precise timing to navigate safely. Small, glowing orbs are scattered along the path, possibly serving as collectible items or checkpoints. The barriers around the arena have adjustable sections that can open or close, potentially acting as gates or obstacles.Weather and lighting:The lighting is bright and evenly distributed, creating a clear and vivid atmosphere. There are no visible weather effects, indicating a controlled indoor environment."
                    ]
    #action_f = [2] * 168
    action_f = [2] * 42 + [1] * 42 + [4] * 42 + [3] * 42
    actions_keys = get_actions(action_f, padding_lengh=81)
    actions = get_action_onehot(actions_keys) # 1*21*80
    actions = torch.from_numpy(actions).to(device).to(torch.bfloat16)     


    from train_wan_t2v_ymzx import YmzxTensorDataset
    dataset = YmzxTensorDataset()
    if indata:
        for i in range(10):
            data = dataset[i+10]
            name = data['name']
            prompt = data['prompt_emb']
            actions = data['actions'].to(device).to(torch.bfloat16).unsqueeze(0)
            raw_actions = get_raw_actions(name)


            video = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                action=actions,
                num_inference_steps=infer_steps,
                #denoising_strength= 0.4,
                sigma_shift=shift,
                cfg_scale= cfg,
                seed=1, tiled=False 
            )
            #save_video(video, save_dir + 'video_'+ str(i) +'.mp4', fps=15, quality=9)
            #save_video(video, save_dir + name.split('/')[-1] +'.mp4', fps=15, quality=9)
            imageio.mimwrite(save_dir + str(i)+'.mp4', write_actions(video, raw_actions.reshape(-1)), fps=16, quality=7, output_params=["-loglevel", "error"])      

    if not indata: 
        for i,prompt in enumerate(texts_mixcaption_dataset):
            video = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                action=actions,
                num_inference_steps=infer_steps,
                #denoising_strength= 0.4,
                sigma_shift=shift,
                cfg_scale= cfg,
                seed=1, tiled=False 
            )
            #save_video(video, save_dir + 'video_'+ str(i) +'.mp4', fps=15, quality=9)
            #save_video(video, save_dir +'_'+ str(i) +'.mp4', fps=15, quality=9)  
            imageio.mimwrite(save_dir + str(i)+ '_'+ prompt[:4] +'.mp4', write_actions(video, actions_keys[3:].reshape(-1)), fps=16, quality=7, output_params=["-loglevel", "error"])      

def test_wan1_3b():
    import torch
    from diffsynth import ModelManager, WanVideoPipeline, save_video, VideoData
    from modelscope import snapshot_download
    import sys 
    import time 

    # Download models
    #snapshot_download("Wan-AI/Wan2.1-T2V-1.3B", local_dir="models/Wan-AI/Wan2.1-T2V-1.3B")
    device = 'cuda:1'
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
    #lora_path = 'exp_out/train_exp/0730_wan_lora_pubg/lightning_lora_ckpts/lora-epoch=39-step=002000.ckpt'
    lora_path = 'exp_out/train_exp/0811_wan_lora_ngr/lightning_lora_ckpts/lora-epoch=03-step=002500.ckpt'
    #lora_path_1 = "exp_out/train_exp/0728_wan_lora_csgo_df/lightning_lora_ckpts/lora-epoch=56-step=003000.ckpt"
    model_manager.load_lora(lora_path, lora_alpha=1)
    #model_manager.load_lora(lora_path_1, lora_alpha=1)
    pipe = WanVideoPipeline.from_model_manager(model_manager, torch_dtype=torch.bfloat16, device=device)
    pipe.enable_vram_management(num_persistent_param_in_dit=None)

    # Text-to-video
    s_t = time.time()
    prompt = "一个战士在作战，csgo风格"
    #prompt = "a soilder is fighting, csgo style"
    #prompt = 'A stylish woman walks down a Tokyo street filled with warm glowing neon and animated city signage'
    #prompt = 'batman fights with superman in the city'


    video = pipe(
        #prompt="Environmental basics:The scene is set in a vibrant, cartoonish industrial port area with bright blue water and wooden platforms. The environment features colorful crates, barrels, and shipping containers stacked around the area. A large wooden barrel is prominently placed on one of the platforms, and there are ramps and pathways connecting different sections. The background includes distant buildings and a clear sky, adding to the lively atmosphere.Main character:A small, anthropomorphic cat character is seen running across the wooden platforms and interacting with the environment. The cat appears agile and is navigating through the industrial setting, moving from one platform to another.Game mechanisms:The primary mechanism involves a large wooden barrel that rotates slowly on its axis. The barrel has a face drawn on it and is positioned on a raised platform. As the cat approaches, the barrel continues its rotation, creating a dynamic interaction point. The cat must time its movements carefully to avoid being hit by the barrel. Additionally, there are ramps and pathways that the cat uses to move between different levels of the environment, indicating a focus on navigation and timing.Weather and lighting:The weather is clear and sunny, with bright daylight illuminating the entire scene. The lighting is vivid, enhancing the colorful and cheerful aesthetic of the environment.",
        prompt = prompt,
        negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
        num_inference_steps=30,sigma_shift=8,
        num_frames = 81, 
        seed=0, tiled=False,
        #sliding_window_size=4, sliding_window_stride=2,
    )
    print("text2video time:", time.time() - s_t)
    save_video(video, f"/root/leinyu/code/results/wan/lora/ngr/0812_{prompt}.mp4", fps=15, quality=5)

def test_wan_fun_1_3b(video_path=None,prompt=None,save_path=None):
    import torch
    from diffsynth import ModelManager, WanVideoPipeline, save_video, VideoData
    from modelscope import snapshot_download, dataset_snapshot_download
    from PIL import Image

    # Load models
    model_manager = ModelManager(device="cpu")
    model_manager.load_models(
        [
            "/dockerdata/Wan2.1-Fun-1.3B-Control/diffusion_pytorch_model.safetensors",
            "/dockerdata/Wan2.1-Fun-1.3B-Control/models_t5_umt5-xxl-enc-bf16.pth",
            "/dockerdata/Wan2.1-Fun-1.3B-Control/Wan2.1_VAE.pth",
            "/dockerdata/Wan2.1-Fun-1.3B-Control/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
        ],
        torch_dtype=torch.bfloat16, # You can set `torch_dtype=torch.float8_e4m3fn` to enable FP8 quantization.
    )
    lora_path = "exp_out/train_exp/0826_wan_fun_lora_thuman2.1_i2n/lightning_lora_ckpts/lora-epoch=02-step=002000.ckpt"
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




def test_wan_action():
    from model import WanAction
    from diffsynth.models.wan_video_vae import WanVideoVAE
    from diffsynth.schedulers.flow_match_unipc import FlowUniPCMultistepScheduler
    from diffsynth import ModelManager, WanVideoPipeline, WanVideoPipelineNonunipc, save_video, VideoData
    from safetensors.torch import load_file 
    from train_wan_t2v_ymzx import load_lora,YmzxTensorDataset
    import imageio
    from tqdm import tqdm

    device = torch.device("cuda:0")
    dtype = torch.bfloat16

    model_manager = ModelManager(torch_dtype=torch.bfloat16, device=device)
    model_manager.load_models([
        "models/Wan-AI/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth",
        "models/Wan-AI/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth",
    ])
    tool = WanVideoPipeline.from_model_manager(model_manager, device=device)

    pipe = WanAction()
    #vae = WanVideoVAE()
    scheduler = FlowUniPCMultistepScheduler()
    scheduler.set_timesteps(30,device=device,shift=8)
    cfg = 6
    indata = 0
    cfg_action = 0
    save_dir = 'exp_out/infer_exp/wanaction/0626_wanaction_newblock/6000_30_6_outdata/'
    os.makedirs(save_dir, exist_ok=True)  

    if cfg != 1.0:
        tool.load_models_to_device(["text_encoder"])
        negative_prompt = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
        prompt_emb_nega = tool.encode_prompt(negative_prompt, positive=False)

    wan_ckpt_path = 'models/Wan-AI/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors'
    wan_ckpt = load_file(wan_ckpt_path)
    pipe.model.load_state_dict(wan_ckpt,strict=True)
    lora_path = 'exp_out/train_exp/0531/checkpoints/epoch=0-step=400.ckpt'
    lora_ckpt = torch.load(lora_path,map_location='cpu')
    load_lora(pipe.model,lora_ckpt)
    action_ckpt_path = 'exp_out/train_exp/0626_wanaction_newblock/checkpoints/epoch=0-step=6000.ckpt'
    action_ckpt = torch.load(action_ckpt_path,map_location='cpu')
    _,unexpected_keys = pipe.load_state_dict(action_ckpt,strict=False)
    print("unexpected keys: ",unexpected_keys)
    pipe = pipe.to(device=device,dtype=dtype)
    pipe.requires_grad_(False)

    # vae_path = 'models/Wan-AI/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth'
    # vae_ckpt = torch.load(vae_path,map_location='cpu')
    # vae.model.load_state_dict(vae_ckpt,strict=True)
    # vae.requires_grad_(False)


    dataset = YmzxTensorDataset()
    dataloader = torch.utils.data.DataLoader(dataset,shuffle=False,batch_size=1)


    if indata:
        for bid,batch in enumerate(dataloader):
            latents = batch["latents"].to(device).to(dtype)
            prompt_emb = batch["prompt_emb"]
            prompt_emb["context"] = prompt_emb["context"][:,0].to(device).to(dtype)
            actions = batch['actions'].to(device).to(dtype)
            name = batch['name'][0]
            raw_actions = get_raw_actions(name)
            name = name.split('/')
            name = '_'.join(name)

            noise = torch.randn_like(latents)
            
            with torch.cuda.amp.autocast(dtype=dtype),torch.no_grad():
                for timestep in tqdm(scheduler.timesteps):
                    timestep = torch.stack([timestep] * latents.shape[0], dim=0).to(dtype=dtype, device=device) 
                    flow_pred = pipe(x=noise,timestep=timestep,context=prompt_emb['context'],action=actions)

                    if cfg!=1:
                        if cfg_action:
                            actions_zero = torch.zeros_like(actions)
                        else:
                            actions_zero = actions.clone()
                        uncon_flow_pred = pipe(x=noise,timestep=timestep,context=prompt_emb_nega['context'],action=actions_zero)
                        flow = uncon_flow_pred + cfg * (flow_pred - uncon_flow_pred)
                    else:
                        flow = flow_pred

                    noise = scheduler.step(flow,timestep,noise,return_dict=False)[0]

                scheduler.set_timesteps(30,device=device,shift=8)
                # vae = vae.to(device=device,dtype=dtype)
                # video = vae.decode(noise,device=device)
                tool.load_models_to_device(['vae'])
                video = tool.decode_video(noise,tiled=False)

            video = (video * 0.5 + 0.5).clamp(0, 1)
            video = video[0].permute(1,2,3,0).float().cpu().numpy()*255
            imageio.mimwrite(save_dir + name +'.mp4', write_actions(video, raw_actions.reshape(-1)), fps=16, quality=7, output_params=["-loglevel", "error"])

    texts_mixcaption_dataset = [
        'A branching bamboo forest rises through vine bridges, mossy stump platforms, waterfall drops, hollow trunk networks, and leaf-littered pavilions, accessible by swinging vines, tilting bamboo rafts, breakable root walls, and water-wheel lifts. Verdant bamboo green, mist, weathered wood, watercolor wash. cool fog, weak sunlight. The character continues heading to the next stage.',
        'A small, animated character running towards the rainbow',
        "The scene is set in a vibrant, colorful arena with a bright green grassy field. The central area features a large, circular blue track with glowing patterns and symbols, surrounded by a low wall. In the background, there are structures resembling bleachers with banners displaying text, and various decorative elements such as palm trees and colorful blocks. The setting appears to be an outdoor stadium designed for a playful, competitive event. A small, animated husky-like character is running along the edge of the blue track, moving forward with a determined gait. The character's fur is gray and white, and it has a playful expression as it navigates the course. The blue track has a conveyor belt mechanism that moves continuously, assisting the character's forward motion. The track surface is adorned with glowing symbols and patterns, possibly indicating special zones or power-ups. There are also yellow coins scattered along the path, which the character can collect. The surrounding walls and structures remain stationary, providing a boundary for the track. A rotating wheel-like structure is visible in the distance, suggesting additional mechanical elements that may interact with the character. The lighting is bright and sunny, casting clear shadows on the ground, indicating a daytime setting. The weather appears clear and pleasant, enhancing the vivid colors of the environment",
        'The scene is set in a brightly colored, indoor arena with a vibrant green floor. The arena features a circular track with blue and white markings, surrounded by orange and yellow walls. There are various colorful structures, including a large blue platform with a star symbol and a green plant-like decoration. The background includes a palm tree silhouette and a section of the wall with a grid pattern. A small, animated character resembling a cat with gray fur and white markings is running across the green floor towards the center of the arena. The character moves steadily forward, leaving a visible shadow on the ground. The arena contains a conveyor belt system that moves along the circular track. The belt has a consistent speed and direction, guiding objects placed on it around the track. There are also stationary platforms and obstacles such as the blue star platform and the green plant structure, which serve as interactive points within the environment. The character interacts with these elements by running towards them, potentially triggering movements or changes in the environment. The lighting is bright and even, suggesting an indoor setting with artificial lighting. There are no weather effects present, and the environment appears clear and well-lit',
        'A suspended botanical laboratory features floating terrarium platforms with bioluminescent flora in magenta and cyan. Transparent tubes connect dome-shaped greenhouses containing oversized carnivorous plants. The sky reveals a perpetual golden hour through stained glass atmospheric filters. An amphibious lemur wearing photosynthesis-enhanced goggles navigates using prehensile pollen trails. Its fur dynamically changes pigmentation to match crossed vegetation, retaining partial camouflage for three seconds after landing. Pollen collector stations trigger platform growth when activated, extending vine pathways. Venus flytrap sentries periodically snap shut, requiring timed evasion. The lemur deploys seed cluster bombs that temporarily neutralize aggressive flora, creating safe passage windows. Directional light beams penetrate through rotating prismatic crystals above, casting rainbow refraction patterns that migrate across surfaces. Humidity particles glow when intersecting light paths.',
        "An inverted dessert dimension features floating shortcake platforms with strawberry syrup waterfalls. Custard clouds drift between crumbling biscuit cliffs dusted with powdered sugar. Marzipan structures extrude whipped cream support beams that gradually melt. A marshmallow-armored penguin utilizes an edible ice staff to freeze unstable surfaces. Each jump leaves temporary frosting footholds that dissolve after 2 seconds, usable by other players in multiplayer mode. Chocolate lava geysers periodically erupt beneath platforms, requiring predictive pathing. Gummy worm elastic bridges can be stretched between platforms using the ice staff's hook function. Random \"sugar rush\" events accelerate all movement speeds for 8 seconds. Warm ambient light mimics caramel glaze coating all surfaces. Sprinkles rain vertically during phase transitions, physically interacting with character movement",
        "Environmental basics:The scene is set in a vibrant, cartoonish industrial port area with bright blue water and wooden platforms. The environment features colorful crates, barrels, and shipping containers stacked around the area. A large wooden barrel is prominently placed on one of the platforms, and there are ramps and pathways connecting different sections. The background includes distant buildings and a clear sky, adding to the lively atmosphere.Main character:A small, anthropomorphic cat character is seen running across the wooden platforms and interacting with the environment. The cat appears agile and is navigating through the industrial setting, moving from one platform to another.Game mechanisms:The primary mechanism involves a large wooden barrel that rotates slowly on its axis. The barrel has a face drawn on it and is positioned on a raised platform. As the cat approaches, the barrel continues its rotation, creating a dynamic interaction point. The cat must time its movements carefully to avoid being hit by the barrel. Additionally, there are ramps and pathways that the cat uses to move between different levels of the environment, indicating a focus on navigation and timing.Weather and lighting:The weather is clear and sunny, with bright daylight illuminating the entire scene. The lighting is vivid, enhancing the colorful and cheerful aesthetic of the environment.",
        "Environmental basics:The scene is set in a circular, elevated track with a futuristic design. The track features a vibrant blue and green color scheme, with a central platform marked by geometric patterns and symbols. The outer edge of the track has a series of evenly spaced yellow markers and black cylindrical supports. The background is a clear, light blue sky, giving the impression of an open, high-altitude setting.Main character:A small, humanoid character with a dark silhouette is running along the inner edge of the circular track, maintaining balance while navigating the curved path.Game mechanisms:The track includes a conveyor belt-like mechanism that moves continuously, assisting the character's forward motion. There are rotating wheels embedded in the track surface, which spin independently and may affect the character's movement. Additionally, there are pistons or mechanical arms extending from the track's edge, moving up and down in a rhythmic pattern. These mechanical components interact dynamically with the character, requiring precise timing and navigation to avoid collisions.Weather and lighting:The lighting is bright and even, suggesting a sunny day with no visible clouds. The environment is well-lit, enhancing the vivid colors of the track and creating sharp shadows cast by the character and mechanical elements.",
        "Environmental basics:The scene is set in a vibrant, circular arena with a bright green floor and blue walls adorned with colorful patterns and text. The central area features a large, detailed emblem with a star design, surrounded by concentric rings. The arena is enclosed by transparent barriers, giving it an open yet contained feel. The background shows a light, cloud-like texture, suggesting an indoor setting with a futuristic or playful theme.Main character:A small, animated husky dog is running along a curved path within the arena. The dog appears to be navigating the course with agility, maintaining balance as it moves forward.Game mechanisms:The arena includes a series of mechanical components such as conveyor belts that move along the curved path, assisting the character's progression. There are also rotating platforms embedded in the floor, which spin slowly, requiring precise timing to navigate safely. Small, glowing orbs are scattered along the path, possibly serving as collectible items or checkpoints. The barriers around the arena have adjustable sections that can open or close, potentially acting as gates or obstacles.Weather and lighting:The lighting is bright and evenly distributed, creating a clear and vivid atmosphere. There are no visible weather effects, indicating a controlled indoor environment."
        ]
    #action_f = [4] * 84 + [1] * 78 + [7] * 42 + [5] * 84 + [2] * 100 + [3] * 122 # + [2] * 512 + [1] * 512
    action_f = [3] * 168
    actions_keys = get_actions(action_f, padding_lengh=81)
    actions = get_action_onehot(actions_keys) # 1*21*80
    actions = torch.from_numpy(actions).to(device).to(torch.bfloat16)    
    nega_actions = torch.zeros_like(actions)

    if not indata:
        with torch.cuda.amp.autocast(dtype=dtype),torch.no_grad():
            for i,text in enumerate(texts_mixcaption_dataset):
                prompt_emb = tool.encode_prompt(text)
                noise = torch.randn(1,16,21,60,104).to(device=device,dtype=dtype)
                
                for timestep in tqdm(scheduler.timesteps):
                    timestep = torch.stack([timestep] * noise.shape[0], dim=0).to(dtype=dtype, device=device) 
                    flow_pred = pipe(x=noise,timestep=timestep,context=prompt_emb['context'],action=actions)

                    if cfg!=1:
                        uncon_flow_pred = pipe(x=noise,timestep=timestep,context=prompt_emb_nega['context'],action=actions)
                        flow_pred = uncon_flow_pred + cfg * (flow_pred - uncon_flow_pred)

                    noise = scheduler.step(flow_pred,timestep,noise,return_dict=False)[0]

                scheduler.set_timesteps(30,device=device,shift=8)
                # vae = vae.to(device=device,dtype=dtype)
                # video = vae.decode(noise,device=device)
                tool.load_models_to_device(['vae'])
                video = tool.decode_video(noise,tiled=False)

                video = (video * 0.5 + 0.5).clamp(0, 1)
                video = video[0].permute(1,2,3,0).float().cpu().numpy()*255
                imageio.mimwrite(save_dir + str(i)+'_'+ text[:2] +'.mp4', write_actions(video, actions_keys[3:].reshape(-1)), fps=16, quality=7, output_params=["-loglevel", "error"])

def tensor2video(self, frames):
    from einops import rearrange
    from PIL import Image
    frames = rearrange(frames, "C T H W -> T H W C")
    frames = ((frames.float() + 1) * 127.5).clip(0, 255).cpu().numpy().astype(np.uint8)
    frames = [Image.fromarray(frame) for frame in frames]
    return frames
    
def lora_weights_test():
    lora_file = 'exp_out/lora_weights/Wan21_CausVid_bidirect2_T2V_1_3B_lora_rank32.safetensors'
    lora_state_dict = load_file(lora_file)
    for key in list(lora_state_dict.keys())[:10]:
        print(key)
    
    lora_file = 'exp_out/lightning_logs/version_3/checkpoints/epoch=9-step=5000.ckpt'
    lora_state_dict = torch.load(lora_file,map_location="cpu")
    for key in list(lora_state_dict.keys())[:20]:
        print(key)

def diffusers_wan1_3b():
    import torch
    from diffusers.utils import export_to_video
    from diffusers import AutoencoderKLWan, WanPipeline
    from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler

    # Available models: Wan-AI/Wan2.1-T2V-14B-Diffusers, Wan-AI/Wan2.1-T2V-1.3B-Diffusers
    model_id = "./models/Wan-AI/Wan2.1-T2V-1.3B" # 这种参数存储结构不是diffusers标准，因此无法本地载入
    vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
    flow_shift = 5.0 # 5.0 for 720P, 3.0 for 480P
    scheduler = UniPCMultistepScheduler(prediction_type='flow_prediction', use_flow_sigmas=True, num_train_timesteps=1000, flow_shift=flow_shift)
    pipe = WanPipeline.from_pretrained(model_id, vae=vae, torch_dtype=torch.bfloat16)
    pipe.scheduler = scheduler
    pipe.to("cuda")

    prompt = "A cat and a dog baking a cake together in a kitchen. The cat is carefully measuring flour, while the dog is stirring the batter with a wooden spoon. The kitchen is cozy, with sunlight streaming through the window."
    negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"

    output = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=720,
        width=1280,
        num_frames=81,
        guidance_scale=5.0,
        ).frames[0]
    export_to_video(output, "output.mp4", fps=16)

def apply_lora_and_diff(base_model: nn.Module, lora_state_dict: dict, alpha: float = 1.0, device=torch.device("cuda"), dtype=torch.bfloat16):
    computation_device = device
    computation_dtype = dtype
    count = 0

    for name, param in lora_state_dict.items():
        if name.endswith(".lora_up.weight"):
            # 提取模块前缀
            prefix = name.replace(".lora_up.weight", "")
            lora_up = param.to(device=computation_device, dtype=computation_dtype)
            lora_down = lora_state_dict[prefix + ".lora_down.weight"].to(device=computation_device, dtype=computation_dtype)

            # 找原始层
            prefix = prefix.replace("diffusion_model.", "")
            target_module = get_module_by_name(base_model, prefix)

            # 计算 delta_w = lora_up @ lora_down
            delta_w = (lora_up @ lora_down) * alpha #/ lora_down.shape[0]  # scaled
            with torch.no_grad():
                target_module.weight += delta_w

        elif name.endswith(".diff"):
            # 这是对 weight 的残差补丁
            prefix = name.replace(".diff", "")
            prefix = prefix.replace("diffusion_model.", "")
            target_module = get_module_by_name(base_model, prefix)
            with torch.no_grad():
                target_module.weight += param.to(device=computation_device, dtype=computation_dtype)

        elif name.endswith(".diff_b"):
            # 对 bias 的残差补丁
            prefix = name.replace(".diff_b", "")
            prefix = prefix.replace("diffusion_model.", "")
            target_module = get_module_by_name(base_model, prefix)
            with torch.no_grad():
                target_module.bias += param.to(device=computation_device, dtype=computation_dtype)
        
        count += 1
    print(f"Applied {count} LoRA patches.")

def get_module_by_name(model: nn.Module, name: str) -> nn.Module:
    parts = name.split(".")
    for part in parts:
        if part.isdigit():
            model = model[int(part)]
        else:
            model = getattr(model, part)
    return model

def condition_modules():
    import torch
    from diffsynth import ModelManager, WanVideoPipeline, save_video, VideoData
    from PIL import Image 

    # Load models
    model_manager = ModelManager(device="cuda:0")
    model_manager.load_models(
        [
            "/dockerdata/VACE-Wan2.1-1.3B-Preview/diffusion_pytorch_model.safetensors",
            "/dockerdata/VACE-Wan2.1-1.3B-Preview/models_t5_umt5-xxl-enc-bf16.pth",
            "/dockerdata/VACE-Wan2.1-1.3B-Preview/Wan2.1_VAE.pth",
        ],
        torch_dtype=torch.bfloat16,
    )

    lora_path = "exp_out/train_exp/0820_wan_lora_vace_csgo/lightning_lora_ckpts/lora-epoch=03-step=002500.ckpt"
    #lora_path_1 = "/root/leinyu/model/Wan_lora/Wan2.1-1.3b-lora-exvideo-v1/model.safetensors"
    #lora_path = "exp_out/train_exp/0722_wan_lora_doom/lightning_lora_ckpts/lora-epoch=00-step=000500.ckpt"
    #lora_path = 'exp_out/train_exp/0531/checkpoints/epoch=0-step=400.ckpt'
    model_manager.load_lora(lora_path, lora_alpha=1)
    #model_manager.load_lora(lora_path_1, lora_alpha=1)
    pipe = WanVideoPipeline.from_model_manager(model_manager, torch_dtype=torch.bfloat16, device="cuda:0")
    #pipe.enable_vram_management(num_persistent_param_in_dit=None)

    for name, module in pipe.named_children():
        print(name)

    ###
    #ref_depth_video_path = 'data/examples/wan/vace_input/doom_vis.mp4'  # Path to the depth video
    #ref_depth_video_path = 'data/examples/wan/vace_input/60180_614733430200000123_6_vis.mp4'
    #ref_depth_video_path = 'data/examples/wan/vace_0722_1/ym_depth_dav2.mp4'
    ref_depth_video_path = 'data/examples/wan/vace_0722/doom_depth_dav2_0-149.mp4'
    #ref_image_path = 'data/examples/wan/vace_input/ft_local/doom_w_csgo_style.png' 

    control_video = VideoData(ref_depth_video_path,height=480,width=832,length=81)
    #mask_video = VideoData('data/examples/wan/60011_615534990200000123_4_mask.mp4',height=480,width=832)
    video = pipe(
        prompt="战士在战斗，csgo风格",
        #prompt="The scene is set in a brightly colored, indoor arena with a vibrant green floor. The arena features a circular track with blue and white markings, surrounded by orange and yellow walls. There are various colorful structures, including a large blue platform with a star symbol and a green plant-like decoration. The background includes a palm tree silhouette and a section of the wall with a grid pattern",
        negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
        num_inference_steps=50,
        height=480, width=832, num_frames=81,
        vace_video=control_video,
        vace_video_mask=None,
        #vace_reference_image=Image.open(ref_image_path).convert('RGB').resize((832, 480)),
        seed=1, tiled=False
    )
    save_video(video, "/root/leinyu/code/DiffSynth-Studio/data/examples/wan/vace_0723/doomdepth_csgoVaceLora.mp4", fps=15, quality=5)

def test_wan_14b():
    import torch
    from diffsynth import ModelManager, WanVideoPipeline, save_video, VideoData
    from modelscope import snapshot_download
    import sys 
    import time 
    import imageio

    # Download models
    #snapshot_download("Wan-AI/Wan2.1-T2V-1.3B", local_dir="models/Wan-AI/Wan2.1-T2V-1.3B")
    device = 'cuda:0'
    device = torch.device(device)

    # Load models
    model_manager = ModelManager(device="cpu")
    dit_path = '/dockerdata/Wan2.1-T2V-14B/'
    dit_paths = [[
                dit_path + "diffusion_pytorch_model-00001-of-00006.safetensors",
                dit_path + "diffusion_pytorch_model-00002-of-00006.safetensors",
                dit_path + "diffusion_pytorch_model-00003-of-00006.safetensors",
                dit_path + "diffusion_pytorch_model-00004-of-00006.safetensors",
                dit_path + "diffusion_pytorch_model-00005-of-00006.safetensors",
                dit_path + "diffusion_pytorch_model-00006-of-00006.safetensors",
                    ],
                dit_path + "models_t5_umt5-xxl-enc-bf16.pth",
                dit_path + "Wan2.1_VAE.pth",]
    
    model_manager.load_models(
        dit_paths,
        torch_dtype=torch.bfloat16, # You can set `torch_dtype=torch.float8_e4m3fn` to enable FP8 quantization.
    )
    #lora_path = "/mnt/aigc_sh/private/jianqima/code/DiffSynth-Studio/experiments/Wan_t2v_14B_lora_16_ymzx_ngr_h5_per_map_resume_with_casual_action_2nodes/version_14/epoch=0-step=1000.ckpt"
    #model_manager.load_lora(lora_path, lora_alpha=1)
    pipe = WanVideoPipeline.from_model_manager(model_manager, torch_dtype=torch.bfloat16, device=device) # 使用完14b后记得调回wanmodel,用jianqi的wan action替换我的wan action，推理没有任何问题，反之待测试。
    pipe.enable_vram_management(num_persistent_param_in_dit=None)

    from diffsynth.models.lora import GeneralLoRAFromPeft
    lora_path = "/mnt/aigc_sh/private/jianqima/code/DiffSynth-Studio/experiments/Wan_t2v_14B_lora_16_ymzx_ngr_h5_per_map_resume_with_casual_action_v3/lightning_logs/version_0/checkpoints/epoch=11-step=6000.ckpt"
    lora_state_dict = torch.load(lora_path,map_location="cpu")
    GeneralLoRAFromPeft().load(pipe.dit,lora_state_dict)  # 测试这种注入方式到底对不对

    pipe.dit.set_action_projection(80)
    model_state_dict = pipe.dit.state_dict()
    i = 0
    for k, v in lora_state_dict.items():
        if "action_projection" in k and k in model_state_dict:
            model_state_dict[k] = lora_state_dict[k]
            i += 1
    print(f"{i} tensors are updated.")
    pipe.dit.load_state_dict(model_state_dict)

    pipe.dit.to(device=device,dtype=torch.bfloat16)

    pipe.dit.has_action_input = True 
    for block in pipe.dit.blocks:
        block.cross_attn.has_action_input = True 
    


    #action_f = [3] * 168
    action_f = [4] * 84 + [1] * 78 + [7] * 42 
    actions_keys = get_actions(action_f, padding_lengh=81)
    actions = get_action_onehot(actions_keys) # 1*21*80
    actions = torch.from_numpy(actions).to(device).to(torch.bfloat16)  

    # Text-to-video
    s_t = time.time()
    video = pipe(
        prompt="Environmental basics:The scene is set in a vibrant, cartoonish industrial port area with bright blue water and wooden platforms. The environment features colorful crates, barrels, and shipping containers stacked around the area. A large wooden barrel is prominently placed on one of the platforms, and there are ramps and pathways connecting different sections. The background includes distant buildings and a clear sky, adding to the lively atmosphere.Main character:A small, anthropomorphic cat character is seen running across the wooden platforms and interacting with the environment. The cat appears agile and is navigating through the industrial setting, moving from one platform to another.Game mechanisms:The primary mechanism involves a large wooden barrel that rotates slowly on its axis. The barrel has a face drawn on it and is positioned on a raised platform. As the cat approaches, the barrel continues its rotation, creating a dynamic interaction point. The cat must time its movements carefully to avoid being hit by the barrel. Additionally, there are ramps and pathways that the cat uses to move between different levels of the environment, indicating a focus on navigation and timing.Weather and lighting:The weather is clear and sunny, with bright daylight illuminating the entire scene. The lighting is vivid, enhancing the colorful and cheerful aesthetic of the environment.",
        negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
        num_inference_steps=30,sigma_shift=8,
        action=actions,
        seed=0, tiled=False
    )
    print("text2video time:", time.time() - s_t)
    save_dir = "exp_out/infer_exp/others/wan_ta2v_14b_6000_"
    #save_video(video, "exp_out/infer_exp/others/wan_ta2v_14b_0.mp4", fps=15, quality=5)
    imageio.mimwrite(save_dir + str(i)+ '_4' +'.mp4', write_actions(video, actions_keys[3:].reshape(-1)), fps=16, quality=7, output_params=["-loglevel", "error"])      

def test_action():
    from train_wan_t2v_ymzx import YmzxTensorDataset
    import imageio
    dataset = YmzxTensorDataset()
    index = np.random.randint(0,len(dataset)) + 1
    data = dataset[index]
    name = data['name']

    actions = get_raw_actions(name)
    video = get_raw_videos(name)
    save_dir = 'exp_out/infer_exp/others/check_action_video.mp4'
    imageio.mimwrite(save_dir, write_actions(video, actions.reshape(-1)), fps=16, quality=7, output_params=["-loglevel", "error"])      

def download_parms():
    snapshot_download(
    model_id="DiffSynth-Studio/Wan2.1-1.3b-lora-exvideo-v1",
    local_dir="/root/leinyu/model/Wan_lora/Wan2.1-1.3b-lora-exvideo-v1",
    allow_file_pattern="*.safetensors"
    )



if __name__ == "__main__":
    #test_text_video_dataset()
    #test_latent_dataset()
    #test_latent()
    #test_lora_wan1_3b() 
    #test_wan1_3b()
    #test_wan_fun_1_3b()
    demo_wan_fun_1_3b()
    #test_wan_action()
    #lora_weights_test()
    #condition_modules()
    #test_action()
    #test_wan_14b()
    #download_parms()