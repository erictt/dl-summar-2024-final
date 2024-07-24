# Fine-tuning RoBERT on SQuAD2.0

The [`run_qa.py`](https://github.com/huggingface/transformers/blob/main/examples/pytorch/question-answering/run_qa.py) script is refactored to
allows to train adapters with RoBERT model .

Below is the example for training roberta-base with seq_bn adapter:

```bash
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
  --output_dir /tmp/debug_squad/ \
  --train_adapter \
  --adapter_config seq_bn \
  --version_2_with_negative \
  --overwrite_output_dir
```
