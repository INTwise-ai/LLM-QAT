#!/bin/bash

source ~/sdxl/tweaks_for_40XX.sh

export RAYON_RS_NUM_CPUS=32

deepspeed --include localhost:2,3 \
	train.py \
	--local_dir "./output" \
	--input_model_filename "NousResearch/Meta-Llama-3-8B" \
	--output_model_filename "8B-finetuned" \
	--train_data_local_path "./gen_data/1k.parquet" \
	--eval_data_local_path "./gen_data/wiki2.parquet" \
	--do_train True \
	--do_eval False \
	--model_max_length 1024 \
	--fp16 False \
	--bf16 True \
	--log_on_each_node False \
	--logging_dir /tmp/output/runs/current \
	--deepspeed "./deepspeed.json" \
	--num_train_epochs 5 \
	--per_device_train_batch_size 1 \
	--per_device_eval_batch_size 1 \
	--gradient_accumulation_steps 1 \
	--evaluation_strategy "no" \
	--save_strategy "steps" \
	--save_steps 2000 \
	--report_to "wandb" \
	--save_total_limit 1 \
	--learning_rate 2e-5 \
	--weight_decay 0. \
	--warmup_ratio 0. \
	--lr_scheduler_type "cosine" \
	--logging_steps 1 \
	--tf32 False \
	--gradient_checkpointing True \
	--qat True \
	--w_bits 4 \
	--a_bits 8 \
	--kv_bits 4 \
	--use_kd False
