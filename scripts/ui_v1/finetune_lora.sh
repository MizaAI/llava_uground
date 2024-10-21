#!/bin/bash

deepspeed llava/train/train_mem.py \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path /your_path/LLaVA/vicuna-7b-v1.5-16k \
    --version v1 \
    --data_path /your_path/LLaVA/playground/data/llava_v1_5_mix665k_box_converted.json \
    --image_folder /your_path/LLaVA/playground/data \
    --vision_tower openai/clip-vit-large-patch14 \
    --pretrain_mm_mlp_adapter /your_path/ui_llava_fine_tune/checkpoints/projector/ui-llava-7b-16k-224-pretrain/mm_projector.bin \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio anyres_ui \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir /your_path/ui_llava_fine_tune/checkpoints/new_base_256_pretrain/lora_llava-v1.5-vicuna-7b-16k-pad-no-fusion_converted \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 240000 \
    --save_total_limit 1 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 16384 \
    --gradient_checkpointing True \
    --dataloader_num_workers 16 \
    --lazy_preprocess True \
    --report_to wandb
