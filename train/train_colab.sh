#!/bin/bash

configs=("seq_bn_default" "double_seq_bn_default" "prefix_tuning_default" "lora_default" "seq_bn_inv_default")

for config in "${configs[@]}"; do
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
		--output_dir "/content/drive/MyDrive/adapter/outs/${config}/" \
		--train_adapter \
		--adapter_config "./configs/${config}.json" \
		--version_2_with_negative \
		--overwrite_output_dir

	cp -r "/content/drive/MyDrive/adapter/outs/${config}/squad_v2" "/content/drive/MyDrive/adapter/outs/adapters/${config}"

	cp -r "/content/drive/MyDrive/adapter/outs/${config}/eval_results.json" "/content/drive/MyDrive/adapter/outs/results/${config}.json"
done
