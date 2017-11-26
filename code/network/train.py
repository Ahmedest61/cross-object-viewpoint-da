import torch
import config
from dataset_rend import RenderedDataset

def main():

  # Create training DataLoader
  train_dataset = \
    RenderedDataset(config.DATA_DIR, config.DATA_LABELS_FP)

if __name__ == "__main__":
    main()
