#!/bin/bash

python3 run_qa.py \
	--model_name_or_path roberta-base \
	--dataset_name squad_v2 \
	--do_train \
	--do_eval \
	--per_device_train_batch_size 16 \
	--learning_rate 1e-4 \
	--num_train_epochs 3 \
	--max_seq_length 384 \
	--doc_stride 128 \
	--output_dir ./output/seq_bn_default/ \
	--train_adapter \
	--adapter_config ./configs/adapter_config_seq_bn_default.json \
	--version_2_with_negative \
	--overwrite_output_dir

python3 run_qa.py \
	--model_name_or_path roberta-base \
	--dataset_name squad_v2 \
	--do_train \
	--do_eval \
	--per_device_train_batch_size 16 \
	--learning_rate 1e-4 \
	--num_train_epochs 3 \
	--max_seq_length 384 \
	--doc_stride 128 \
	--output_dir ./output/pf_default/ \
	--train_adapter \
	--adapter_config ./configs/adapter_config_pf_default.json \
	--version_2_with_negative \
	--overwrite_output_dir
