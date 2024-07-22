#!/bin/bash

configs=("mh_seq_bn_default" "mh_seq_bn_rf_4" "mh_seq_bn_rf_8" "mh_seq_bn_rf_32")

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
		--output_dir "./output/${config}/" \
		--train_adapter \
		--adapter_config "./configs/${config}.json" \
		--version_2_with_negative \
		--overwrite_output_dir

	cp -r "./output/${config}/squad_v2" "./adapters/${config}"

	cp -r "./output/${config}/eval_results.json" "./results/${config}.json"
done
