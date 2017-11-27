import os
import config
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import transforms, models

# Imports from src files
from dataset_rend import RenderedDataset
#from transform_rend import ToTensor

#####################
#   BEGIN HELPERS   #
#####################

def create_rend_dataloader(data_dir, data_labels_fp):
  dataset = RenderedDataset(data_dir=data_dir, 
                            data_labels_fp=data_labels_fp,
                            transform=transforms.Compose([transforms.ToTensor()]))
  dataloader = DataLoader(dataset, 
                          batch_size=config.BATCH_SIZE, 
                          shuffle=True,
                          num_workers=4)
  return dataloader

def create_model():
  model = models.resnet18(pretrained=config.PRETRAINED)
  out_classes = config.AZIMUTH_BINS * config.ELEVATION_BINS
  model.fc = nn.Linear(model.fc.in_features, out_classes)
  return model

def train_model(model, loss, optimizer, explorer, num_epochs):
  #TODO
  pass

#####################
#    END HELPERS    #
#####################

def main():

  # Print beginning debug info
  print "Beginning training process at..."
  print datetime.datetime.now()
  config.PRINT_CONFIG()

  # Create training DataLoader
  print "Loading training data..."
  train_ims_dir = os.path.join(config.DATA_DIR, 'train')
  train_dataloader =  \
    create_rend_dataloader(train_ims_dir, config.DATA_LABELS_FP)

  """
  # Create validation DataLoader
  print "Loading validation data..."
  validate_ims_dir = os.path.join(config.DATA_DIR, 'val')
  validate_dataloader = \
    create_rend_dataloader(validate_ims_dir, config.DATA_LABELS_FP)
  """

  # Set up model for training
  print "Creating model..."
  model = create_model()
  if config.GPU and torch.cuda.is_available():
    print "Enabling GPU"
    if config.MULTI_GPU and torch.cuda.device_count() > 1:
      print "Using multiple GPUs:", torch.cuda.device_count()
      model = nn.DataParallel(model)
    model = model.cuda()

  # Set up loss and optimizer
  loss= nn.CrossEntropyLoss()
  optimizer = optim.SGD(model.parameters(), 
                        lr=config.LEARNING_RATE,
                        momentum=config.MOMENTUM)
  explorer = lr_scheduler.StepLR(optimizer, 
                                       step_size=config.STEP_SIZE,
                                       gamma=config.GAMMA)

  # Perform training
  print "!!!Starting training!!!"
  model = train_model(model, loss, optimizer, explorer, config.EPOCHS)

  # Create testing DataLoader
  print "Loading testing data..."
  train_ims_dir = os.path.join(config.DATA_DIR, 'test')
  train_dataloader =  \
    create_rend_dataloader(train_ims_dir, config.DATA_LABELS_FP)

  #TODO: test accuracy

if __name__ == "__main__":
    main()
