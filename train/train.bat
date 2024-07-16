@echo off

setlocal

set configs=seq_bn_default double_seq_bn_default prefix_tuning_default lora_default seq_bn_inv_default

for %%c in (%configs%) do (
    python run_qa.py ^
        --model_name_or_path roberta-base ^
        --dataset_name squad_v2 ^
        --do_train ^
        --do_eval ^
        --per_device_train_batch_size 16 ^
        --learning_rate 1e-4 ^
        --num_train_epochs 3 ^
        --max_seq_length 384 ^
        --doc_stride 128 ^
        --output_dir .\output\%%c\ ^
        --train_adapter ^
        --adapter_config .\configs\%%c.json ^
        --version_2_with_negative ^
        --overwrite_output_dir

    xcopy .\output\%%c\squad_v2 .\adapters\%%c /E /I

    xcopy .\output\%%c\eval_results.json .\results\%%c.json
)

endlocal
