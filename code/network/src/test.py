import os
import sys
import math
import test_config
import datetime
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import transforms

# Imports from src files
from data_viewpoint import ViewpointDataset
from viewpoint_loss import ViewpointLoss
import models

#####################
#   BEGIN HELPERS   #
#####################

def log_print(string):
  print "[%s]\t %s" % (datetime.datetime.now(), string)

def create_rend_dataloader(data_base_dir, data_list, data_set):
  dataset = ViewpointDataset(data_base_dir=data_base_dir,
                            data_list=data_list,
                            data_set=data_set,
                            transform=transforms.Compose([transforms.ToTensor()]))
  dataloader = DataLoader(dataset, 
                          batch_size=test_config.BATCH_SIZE, 
                          shuffle=True,
                          num_workers=4)
  return dataloader

def create_model(network_type):
  res_v = test_config.RESNET_LAYERS

  if network_type == "VIEWPOINT":
    model = models.viewpoint_net(layers=res_v, pretrained=test_config.PRETRAINED)
  elif network_type == "VIEWPOINT_CLASS_DOMAIN":
    model = models.vcd_net(layers=res_v, pretrained=test_config.PRETRAINED)
    model.fc2_class = nn.Linear(model.fc2_class.in_features, test_config.NUM_OBJ_CLASSES)

  # Adjust network size
  model.fc_azi = nn.Linear(model.fc_azi.in_features, test_config.AZIMUTH_BINS)
  model.fc_ele = nn.Linear(model.fc_ele.in_features, test_config.ELEVATION_BINS)
  return model

def test_model(model, test_dataloader):
  test_az_err = test_ele_err = 0
  print_interval = 100
  predictions = []
  for data in test_dataloader:
    # Gather batch data (images + corresponding annots)
    im_fps, inputs, annot_azimuths, annot_elevations, annot_classes, annot_domains= \
      data['image_fp'], data['image'], data['azimuth'], data['elevation'], data['class_id'], data['domain_id']

    # Wrap as pytorch autograd Variable
    if test_config.GPU and torch.cuda.is_available():
      inputs = Variable(inputs.cuda())
      annot_azimuths = Variable(annot_azimuths.cuda())
      annot_elevations = Variable(annot_elevations.cuda())
    else:
      inputs = Variable(inputs)
      annot_azimuths = Variable(annot_azimuths)
      annot_elevations = Variable(annot_elevations)

    # Forward pass and calculate loss
    if test_config.NETWORK_TYPE == "VIEWPOINT":
      out_azimuths, out_elevations = model(inputs)
    elif test_config.NETWORK_TYPE == "VIEWPOINT_CLASS_DOMAIN":
      out_azimuths, out_elevations, out_classes, out_domains = model(inputs)

    # Update accuracy
    _, pred_azimuths = torch.max(out_azimuths.data, 1)
    azimuth_diffs = torch.abs(pred_azimuths - annot_azimuths.data)
    azimuth_errs = torch.min(azimuth_diffs, 360-azimuth_diffs)
    test_az_err += azimuth_errs.sum()
    _, pred_elevations = torch.max(out_elevations.data, 1)
    elevation_diffs = torch.abs(pred_elevations - annot_elevations.data)
    elevation_errs = torch.min(elevation_diffs, 360-elevation_diffs)
    test_ele_err += elevation_errs.sum()

    for i in xrange(len(im_fps)):
      if pred_elevations[i] >= 180:
        predictions.append([im_fps[i], pred_azimuths[i], pred_elevations[i]-360])
      else:
        predictions.append([im_fps[i], pred_azimuths[i], pred_elevations[i]])
    
  # Report epoch results
  num_images = len(test_dataloader.dataset)
  test_az_err = float(test_az_err) / float(num_images)
  test_ele_err = float(test_ele_err) / float(num_images)
  log_print("[TEST SET] %i ims - Azimuth Err: %f   Elevation Err: %f" % (num_images, test_az_err, test_ele_err))

  return predictions

#####################
#    END HELPERS    #
#####################

def main():

  # Redirect output to log file
  sys.stdout = open(test_config.OUT_LOG_FP, 'w')
  sys.stderr = sys.stdout
  log_print("Beginning script...")

  # Print beginning debug info
  log_print("Printing test_config file...")
  test_config.PRINT_CONFIG()

  # Set up model for training
  log_print("Creating model...")
  model = create_model(test_config.NETWORK_TYPE)
  if test_config.GPU and torch.cuda.is_available():
    log_print("Enabling GPU")
    if test_config.MULTI_GPU and torch.cuda.device_count() > 1:
      log_print("Using multiple GPUs: %i" % torch.cuda.device_count())
      model = nn.DataParallel(model)
    model = model.cuda()
  else:
    log_print("Ignoring GPU (CPU only)")

  # Load model weights
  log_print("Loading model weights from %s..." % test_config.IN_WEIGHTS_FP)
  if test_config.IN_WEIGHTS_FP != None:
    model.load_state_dict(torch.load(test_config.IN_WEIGHTS_FP))

  # Create testing DataLoader
  log_print("Loading testing data...")
  test_dataloader =  \
    create_rend_dataloader(test_config.DATA_BASE_DIR, test_config.DATA_TEST_LIST, 'test')

  # Test and output accuracy/predictions
  log_print("Testing model on test set...")
  predictions = test_model(model, test_dataloader)
  log_print("Writing predictions to %s..." % test_config.OUT_PRED_FP)
  out_f = open(test_config.OUT_PRED_FP, 'w')
  for p in predictions:
    out_f.write("%s,%i,%i\n" % (p[0], p[1], p[2]))
  out_f.close()

  log_print("Script DONE!")

if __name__ == "__main__":
    main()
