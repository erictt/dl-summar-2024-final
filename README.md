# dl-final-project

1. Create conda env

`$ conda env create --file environment.yml` 

2. Train adapter

`$ cd train/`

`$ ./train.sh`

- The script will run train all of the adapters configured in `./train/configs/` and save the adapters and the evaluation results to the folder `./train/adapters` and `./train/results/` respectively.

`$ ./train_colab.sh` will save the output to your google drive.(you may need add apapter/outs folder in your google drive)

`train.bat` is for Windows OS.