#!/bin/bash

# Mixtral-ORPO series are trained on 8 * H100s

accelerate launch --config_file ./src/accelerate/fsdp.yaml main.py \
    --lr 5e-6 \
    --torch_compile False \
    --alpha 0.05 \
    --lr_scheduler_type inverse_sqrt \
    --warmup_steps 100 \
    --model_name mistralai/Mixtral-8x7B-v0.1 \
    --data_name argilla/distilabel-capybara-dpo-7k-binarized \
    --num_train_epochs 3 \
    --optim adamw_torch \
    --gradient_accumulation_steps 1 \
    --prompt_max_length 1792 \
    --response_max_length 2048 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 2 \
    --num_proc 32 \
    --flash_attention_2 \
    --push_to_hub true \
    --save_dir data/Mixtral-8x7B-capybara-dpo-7k-v0.1 \
    --hub_model_id orpo-explorers/Mixtral-8x7B-capybara-dpo-7k-v0.1