import os
import config
import datetime
import torch
from torchvision import transforms, utils

# Imports from src files
from dataset_rend import RenderedDataset
from transform_rend import ToTensor

def main():

  # Print beginning debug info
  print "Beginning training..."
  print datetime.datetime.now()
  config.PRINT_CONFIG()

  # Create training DataLoader
  train_ims_dir = os.path.join(config.DATA_DIR, 'train')
  train_dataset = RenderedDataset(data_dir=train_ims_dir, 
                                  data_labels_fp=config.DATA_LABELS_FP,
                                  transform=transforms.Compose([ToTensor()]))

if __name__ == "__main__":
    main()
