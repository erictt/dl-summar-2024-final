# Deep Learning - Final Project

## Adaptive Layers for Efficient Learning: Scaling Architecture in RoBERTa for Optimized Performance on SQuAD 2.0

### Prepare Environment

1. **Create Conda Environment**

   ```bash
   conda env create --file environment.yml
   ```

2. **Activate Conda Environment**

   ```bash
   conda activate cs7643-final
   ```

### Train Adapters

To train the adapters, follow these steps:

1. Add configurations under the `./train/configs/` directory.
2. Update the list in the train script accordingly.

Upon completion, the adapters and evaluation results will be stored in `./train/adapters` and `./train/results/` directories, respectively.

**Training Scripts:**

- For Linux/macOS:
  
  ```bash
  ./train.sh
  ```

- For Google Colab (outputs will be saved to your Google Drive; ensure you have `adapter/outs` folders in your Drive):
  
  ```bash
  ./train_colab.sh
  ```

- For Windows:

  ```bash
  train.bat
  ```
