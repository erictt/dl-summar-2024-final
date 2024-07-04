# dl-final-project

1. Create conda env

`$ conda env create --file environment.yml`

2. Train adapter

e.g.

```
python3 run_qa.py \
  --model_name_or_path roberta-base  \
  --dataset_name rajpurkar/squad_v2 \
  --do_train \
  --do_eval \
  --per_device_train_batch_size 12 \
  --learning_rate 3e-5 \
  --num_train_epochs 2 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --output_dir ./seq_bn/ \
  --train_adapter \
  --adapter_config seq_bn \
  --overwrite_output_dir
```
