import os
import config
import datetime
import torch
from torch.utils.data import DataLoader
from torchvision import transforms, utils

# Imports from src files
from dataset_rend import RenderedDataset
from transform_rend import ToTensor

#####################
#   BEGIN HELPERS   #
#####################

def create_rend_dataloader(data_dir, data_labels_fp):
  dataset = RenderedDataset(data_dir=data_dir, 
                            data_labels_fp=data_labels_fp,
                            transform=transforms.Compose([ToTensor()]))
  dataloader = DataLoader(dataset, 
                          batch_size=config.BATCH_SIZE, 
                          shuffle=True,
                          num_workers=4)
  return dataloader

#####################
#    END HELPERS    #
#####################

def main():

  # Print beginning debug info
  print "Beginning training..."
  print datetime.datetime.now()
  config.PRINT_CONFIG()

  # Create training DataLoader
  train_ims_dir = os.path.join(config.DATA_DIR, 'train')
  train_dataloader =  \
    create_rend_dataloader(train_ims_dir, config.DATA_LABELS_FP)

  # Create validation DataLoader
  validate_ims_dir = os.path.join(config.DATA_DIR, 'validation')
  validate_dataloader = \
    create_rend_dataloader(validate_ims_dir, config.DATA_LABELS_FP)

if __name__ == "__main__":
    main()
