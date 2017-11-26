import os
import torch
import config
from dataset_rend import RenderedDataset

def main():

  # Create training DataLoader
  train_ims_dir = os.path.join(config.DATA_DIR, 'train')
  train_dataset = \
    RenderedDataset(train_ims_dir, config.DATA_LABELS_FP)

if __name__ == "__main__":
    main()
