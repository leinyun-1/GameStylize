import torch 
from diffsynth import SDVideoPipelineRunner

def test_diffutoon():
    config_stage_1_template = {
    "models": {
        "model_list": [
            "models/stable_diffusion/aingdiffusion_v12.safetensors",
            "models/ControlNet/control_v11p_sd15_softedge.pth",
            "models/ControlNet/control_v11f1p_sd15_depth.pth"
        ],
        "textual_inversion_folder": "models/textual_inversion",
        "device": "cuda",
        "lora_alphas": [],
        "controlnet_units": [
            {
                "processor_id": "softedge",
                "model_path": "models/ControlNet/control_v11p_sd15_softedge.pth",
                "scale": 0.5
            },
            {
                "processor_id": "depth",
                "model_path": "models/ControlNet/control_v11f1p_sd15_depth.pth",
                "scale": 0.5
            }
        ]
    },
    "data": {
        "input_frames": {
            "video_file": "/content/input_video.mp4",
            "image_folder": None,
            "height": 512,
            "width": 512,
            "start_frame_id": 0,
            "end_frame_id": 30
        },
        "controlnet_frames": [
            {
                "video_file": "/content/input_video.mp4",
                "image_folder": None,
                "height": 512,
                "width": 512,
                "start_frame_id": 0,
                "end_frame_id": 30
            },
            {
                "video_file": "/content/input_video.mp4",
                "image_folder": None,
                "height": 512,
                "width": 512,
                "start_frame_id": 0,
                "end_frame_id": 30
            }
        ],
        "output_folder": "data/examples/diffutoon_edit/color_video",
        "fps": 25
    },
    "smoother_configs": [
        {
            "processor_type": "FastBlend",
            "config": {}
        }
    ],
    "pipeline": {
        "seed": 0,
        "pipeline_inputs": {
            "prompt": "best quality, perfect anime illustration, orange clothes, night, a girl is dancing, smile, solo, black silk stockings",
            "negative_prompt": "verybadimagenegative_v1.3",
            "cfg_scale": 7.0,
            "clip_skip": 1,
            "denoising_strength": 0.9,
            "num_inference_steps": 20,
            "animatediff_batch_size": 8,
            "animatediff_stride": 4,
            "unet_batch_size": 8,
            "controlnet_batch_size": 8,
            "cross_frame_attention": True,
            "smoother_progress_ids": [-1],
            # The following parameters will be overwritten. You don't need to modify them.
            "input_frames": [],
            "num_frames": 30,
            "width": 512,
            "height": 512,
            "controlnet_frames": []
        }
    }
}

    config_stage_2_template = {
        "models": {
            "model_list": [
                "models/stable_diffusion/aingdiffusion_v12.safetensors",
                "models/AnimateDiff/mm_sd_v15_v2.ckpt",
                "models/ControlNet/control_v11f1e_sd15_tile.pth",
                #"models/ControlNet/control_v11p_sd15_lineart.pth"
                "models/ControlNet/control_v11f1p_sd15_depth.pth"
            ],
            "textual_inversion_folder": "models/textual_inversion",
            "device": "cuda",
            "lora_alphas": [],
            "controlnet_units": [
                {
                    "processor_id": "tile",
                    "model_path": "models/ControlNet/control_v11f1e_sd15_tile.pth",
                    "scale": 0.5
                },
                # {
                #     "processor_id": "lineart",
                #     "model_path": "models/ControlNet/control_v11p_sd15_lineart.pth",
                #     "scale": 0.5
                # }
                {
                    "processor_id": "depth",
                    "model_path": "models/ControlNet/control_v11f1p_sd15_depth.pth",
                    "scale": 0.5
                }
            ]
        },
        "data": {
            "input_frames": {
                "video_file": "/content/input_video.mp4",
                "image_folder": None,
                "height": 1024,
                "width": 1024,
                "start_frame_id": 0,
                "end_frame_id": 30
            },
            "controlnet_frames": [
                {
                    "video_file": "/content/input_video.mp4",
                    "image_folder": None,
                    "height": 1024,
                    "width": 1024,
                    "start_frame_id": 0,
                    "end_frame_id": 30
                },
                {
                    "video_file": "/content/input_video.mp4",
                    "image_folder": None,
                    "height": 1024,
                    "width": 1024,
                    "start_frame_id": 0,
                    "end_frame_id": 30
                }
            ],
            "output_folder": "/content/output",
            "fps": 25
        },
        "pipeline": {
            "seed": 0,
            "pipeline_inputs": {
                "prompt": "a soldier with a gun is fighting in the room, walls in background, csgo style, realistic, high quality",
                "negative_prompt": "verybadimagenegative_v1.3",
                "cfg_scale": 7.0,
                "clip_skip": 2,
                "denoising_strength": 1.0,
                "num_inference_steps": 10,
                "animatediff_batch_size": 16,
                "animatediff_stride": 8,
                "unet_batch_size": 1,
                "controlnet_batch_size": 1,
                "cross_frame_attention": False,
                # The following parameters will be overwritten. You don't need to modify them.
                "input_frames": [],
                "num_frames": 90, #30,
                "width": 1536,
                "height": 1536,
                "controlnet_frames": []
            }
        }
    }

    config = config_stage_2_template.copy()
    config["data"]["input_frames"] = {
        "video_file": "data/examples/sd/doom.mp4",
        "image_folder": None,
        "height": 1024,
        "width": 1024,
        "start_frame_id": 0,
        "end_frame_id": 90 #30 
    }
    config["data"]["controlnet_frames"] = [config["data"]["input_frames"], config["data"]["input_frames"]]
    config["data"]["output_folder"] = "/root/leinyu/code/results/sd/diffutoon/doom_baseSD15_csgoInPmt_depthCtrnet&tileCtrnet"
    config["data"]["fps"] = 25

    runner = SDVideoPipelineRunner()
    runner.run(config)



if __name__ == "__main__":
    test_diffutoon()