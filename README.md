# Deep Learning - Final Project

Adaptive Layers for Efficient Learning: Scaling Architecture in RoBERTa for Optimized Performance on SQuAD 2.0

## Prepare Env

1. Create conda environment

`conda env create --file environment.yml`

2. Activate conda environment

`conda activate cs7643-final`

## Train adapters

To train the adapters, you first need to add configurations under `./train/configs/`, then change the list in the train script.

Once the training is done, the adapters and the evaluation results will be stored in folder `./train/adapters` and `./train/results/` respectively.

These are the script used for training in different platform/OS:

- `./train.sh` // Linux/macOS
- `./train_colab.sh` // will save the output to your google drive.(you may need add apapter/outs folder in your google drive)
- `train.bat` // Windows
