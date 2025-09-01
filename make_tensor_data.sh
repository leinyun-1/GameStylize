#sh /mnt/aigc_cq/private/leinyu/code/scripts/segnet_cudnn/occ_clean.sh
CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" python train_wan_t2v_ymzx.py \
  --task ymzx_data_process \
  --output_path ./exp_out \
  --text_encoder_path "models/Wan-AI/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth" \
  --vae_path "models/Wan-AI/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth" \
  --image_encoder_path "/dockerdata/SkyReels-V2-I2V-1.3B-540P/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth" \
  --num_frames 81 \
  --height 480 \
  --width 832 \
  --bs 2 \
  --dataloader_num_workers 4
#sh /mnt/aigc_cq/private/leinyu/code/scripts/segnet_cudnn/occ_run.sh