import os
import sys
import math
import config
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
                          batch_size=config.BATCH_SIZE, 
                          shuffle=True,
                          num_workers=4)
  return dataloader

def create_model(network_type):
  res_v = config.RESNET_LAYERS

  if network_type == "VIEWPOINT":
    model = models.viewpoint_net(layers=res_v, pretrained=config.PRETRAINED)
  elif network_type == "VIEWPOINT_CLASS_DOMAIN":
    model = models.vcd_net(layers=res_v, pretrained=config.PRETRAINED)
    model.fc2_class = nn.Linear(model.fc2_class.in_features, config.NUM_OBJ_CLASSES)

  # Adjust network size
  model.fc_azi = nn.Linear(model.fc_azi.in_features, config.AZIMUTH_BINS)
  model.fc_ele = nn.Linear(model.fc_ele.in_features, config.ELEVATION_BINS)
  return model


class FC(nn.Module):
  def __init__(self, nFeat = 1000, nOut = 360):
    super(FC, self).__init__()
    self.fc = nn.Linear(nFeat, nOut)
  def forward(self, input):
    out = self.fc(input)
    return out

class Discriminator(nn.Module):
  def __init__(self, nFeat = 1000, nClass = 2):
    super(Discriminator, self).__init__()
    self.fc1 = nn.Linear(nFeat, nFeat)
    self.fc2 = nn.Linear(nFeat, nClass)
  def forward(self, input):
    x = self.fc1(x)
    x = self.fc2(x)
    return out

def split_model(model):
  encoder = model
  fc_azi = FC(nOut = 360)
  fc_ele = FC(nOut = 360)
  fc_azi.fc.weight.data = encoder.module.fc_azi.weight.data.clone()
  fc_azi.fc.bias.weight.data = encoder.module.fc_azi_bias.weight.data.clone()
  fc_ele.fc.weight.data = encoder.module.fc_ele.weight.data.clone()
  fc_ele.fc.bias.weight.data = encoder.module.fc_ele.bias.weight.data.clone()
  
  encoder.module.fc_azi = nn.ReLU()
  encoder.module.fc_ele = nn.ReLU()
  
  discriminator_class = Discriminator(nClass = 12)

  
  return encoder, fc_azi, fc_ele, discriminator_class, discriminator_domain
  


def train_model(model, train_dataloader, val_dataloader, loss_f_viewpoint, loss_f_class, loss_f_domain, optimizer, explorer, epochs):

  init_time = time.time()
  best_loss = best_az_err = best_ele_err = 0.0
  best_weights = model.state_dict()
  best_epoch = -1

  for epoch in xrange(epochs):
    log_print("Epoch %i/%i: %i batches of %i images each" % (epoch+1, epochs, len(train_dataloader.dataset)/config.BATCH_SIZE, config.BATCH_SIZE))

    for phase in ["train", "val"]:
      if phase == "train":
        explorer.step()
        model.train(True)
        dataloader = train_dataloader
      else:
        model.train(False)
        dataloader = val_dataloader

      # Iterate over dataset
      epoch_loss = epoch_az_err = epoch_ele_err = 0
      curr_loss = curr_az_err = curr_ele_err = 0
      print_interval = 100
      batch_count = 0
      for data in dataloader:
        # Gather batch data (images + corersponding annots)
        im_fps, inputs, annot_azimuths, annot_elevations, annot_classes, annot_domains= \
          data['image_fp'], data['image'], data['azimuth'], data['elevation'], data['class_id'], data['domain_id']

        # Wrap as pytorch autograd Variable
        if config.GPU and torch.cuda.is_available():
          inputs = Variable(inputs.cuda())
          annot_azimuths = Variable(annot_azimuths.cuda())
          annot_elevations = Variable(annot_elevations.cuda())
          annot_classes = Variable(annot_classes.cuda())
          annot_domains = Variable(annot_domains.cuda())
        else:
          inputs = Variable(inputs)
          annot_azimuths = Variable(annot_azimuths)
          annot_elevations = Variable(annot_elevations)
          annot_classes = Variable(annot_classes)
          annot_domains = Variable(annot_domains)

        # Forward pass and calculate loss
        optimizer.zero_grad()
        if config.NETWORK_TYPE == "VIEWPOINT":
          out_azimuths, out_elevations = model(inputs)
        elif config.NETWORK_TYPE == "VIEWPOINT_CLASS_DOMAIN":
          out_azimuths, out_elevations, out_classes, out_domains = model(inputs)
        
        # Calculate losses
        if config.NETWORK_TYPE == "VIEWPOINT":
          loss_azimuth = loss_f_viewpoint(out_azimuths, annot_azimuths)
          loss_elevation = loss_f_viewpoint(out_elevations, annot_elevations)
          loss = loss_azimuth + loss_elevation
          curr_loss += loss.data[0]
        elif config.NETWORK_TYPE == "VIEWPOINT_CLASS_DOMAIN":
          loss_azimuth = loss_f_viewpoint(out_azimuths, annot_azimuths)
          loss_elevation = loss_f_viewpoint(out_elevations, annot_elevations)
          loss_class = loss_f_class(out_classes, annot_classes)
          loss_domain = loss_f_domain(out_domains, annot_domains.float())

          # TODO: fine tune these values
          lambda_1 = .35
          lambda_2 = .35
          loss = (loss_azimuth + loss_elevation) + lambda_1*loss_class + lambda_2*loss_domain
          curr_loss += loss.data[0]

        # Update accuracy
        _, pred_azimuths = torch.max(out_azimuths.data, 1)
        azimuth_diffs = torch.abs(pred_azimuths - annot_azimuths.data)
        azimuth_errs = torch.min(azimuth_diffs, 360-azimuth_diffs)
        curr_az_err += azimuth_errs.sum()
        _, pred_elevations = torch.max(out_elevations.data, 1)
        elevation_diffs = torch.abs(pred_elevations - annot_elevations.data)
        elevation_errs = torch.min(elevation_diffs, 360-elevation_diffs)
        curr_ele_err += elevation_errs.sum()
        
        # Backward pass (if train)
        if phase == "train":
          loss.backward()
          optimizer.step()

        # Output
        if batch_count % print_interval == 0 and batch_count != 0:
          epoch_loss += curr_loss
          epoch_az_err += curr_az_err
          epoch_ele_err += curr_ele_err
          if phase == "train":
            curr_loss = float(curr_loss) / float(print_interval*config.BATCH_SIZE)
            curr_az_err = float(curr_az_err) / float(print_interval*config.BATCH_SIZE)
            curr_ele_err = float(curr_ele_err) / float(print_interval*config.BATCH_SIZE)
            log_print("\tBatches %i-%i -\tLoss: %f \t Azimuth Err: %f   Elevation Err: %f" % (batch_count-print_interval+1, batch_count, curr_loss, curr_az_err, curr_ele_err))
          curr_loss = curr_az_err = curr_ele_err = 0
        batch_count += 1
      
      # Report epoch results
      num_images = len(dataloader.dataset)
      epoch_loss = float(epoch_loss+curr_loss) / float(num_images)
      epoch_az_err = float(epoch_az_err+curr_az_err) / float(num_images)
      epoch_ele_err = float(epoch_ele_err+curr_ele_err) / float(num_images)
      log_print("\tEPOCH %i [%s] - Loss: %f   Azimuth Err: %f   Elevation Err: %f" % (epoch+1, phase, epoch_loss, epoch_az_err, epoch_ele_err))

      # Save best model weights from epoch
      err_improvement = (best_az_err - epoch_az_err) + (best_ele_err - epoch_ele_err)
      if phase == "val" and (err_improvement >= 0 or epoch == 0):
        best_az_err = epoch_az_err
        best_ele_err = epoch_ele_err
        best_loss = epoch_loss
        best_weights = model.state_dict()
        best_epoch = epoch

  # Finish up
  time_elapsed = time.time() - init_time
  log_print("BEST EPOCH: %i/%i - Loss: %f   Azimuth Err: %f   Elevation Err: %f" % (best_epoch+1, epochs, best_loss, best_az_err, best_ele_err))
  log_print("Training completed in %sm %ss" % (time_elapsed // 60, time_elapsed % 60))
  model.load_state_dict(best_weights)
  return model

def save_model_weights(model, filepath):
  torch.save(model.state_dict(), filepath)

def test_model(model, test_dataloader, loss_f_viewpoint):
  test_loss = test_az_err = test_ele_err = 0
  print_interval = 100
  predictions = []
  for data in test_dataloader:
    # Gather batch data (images + corresponding annots)
    im_fps, inputs, annot_azimuths, annot_elevations, annot_classes, annot_domains= \
      data['image_fp'], data['image'], data['azimuth'], data['elevation'], data['class_id'], data['domain_id']

    # Wrap as pytorch autograd Variable
    if config.GPU and torch.cuda.is_available():
      inputs = Variable(inputs.cuda())
      annot_azimuths = Variable(annot_azimuths.cuda())
      annot_elevations = Variable(annot_elevations.cuda())
    else:
      inputs = Variable(inputs)
      annot_azimuths = Variable(annot_azimuths)
      annot_elevations = Variable(annot_elevations)

    # Forward pass and calculate loss
    if config.NETWORK_TYPE == "VIEWPOINT":
      out_azimuths, out_elevations = model(inputs)
    elif config.NETWORK_TYPE == "VIEWPOINT_CLASS_DOMAIN":
      out_azimuths, out_elevations, out_classes, out_domains = model(inputs)
    loss_azimuth = loss_f_viewpoint(out_azimuths, annot_azimuths)
    loss_elevation = loss_f_viewpoint(out_elevations, annot_elevations)
    loss = loss_azimuth + loss_elevation
    test_loss += loss.data[0]

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
  test_loss = float(test_loss) / float(num_images)
  test_az_err = float(test_az_err) / float(num_images)
  test_ele_err = float(test_ele_err) / float(num_images)
  log_print("[TEST SET] %i ims - Loss: %f   Azimuth Err: %f   Elevation Err: %f" % (num_images, test_loss, test_az_err, test_ele_err))

  return predictions

#####################
#    END HELPERS    #
#####################

def main():

  # Redirect output to log file
  sys.stdout = open(config.OUT_LOG_FP, 'w')
  sys.stderr = sys.stdout
  log_print("Beginning script...")

  # Print beginning debug info
  log_print("Printing config file...")
  config.PRINT_CONFIG()

  # Create training DataLoader
  log_print("Loading training data...")
  train_dataloader =  \
    create_rend_dataloader(config.DATA_BASE_DIR, config.DATA_TRAIN_LIST, 'train')

  # Create validation DataLoader
  log_print("Loading validation data...")
  val_dataloader = \
    create_rend_dataloader(config.DATA_BASE_DIR, config.DATA_VAL_LIST, 'val')

  # Set up model for training
  log_print("Creating model...")
  model = create_model(config.NETWORK_TYPE)
  encoder, fc_azi, fc_ele, discriminator_class, discriminator_domain = split_model(model)
  if config.GPU and torch.cuda.is_available():
    log_print("Enabling GPU")
    if config.MULTI_GPU and torch.cuda.device_count() > 1:
      log_print("Using multiple GPUs: %i" % torch.cuda.device_count())
      model = nn.DataParallel(model)
    model = model.cuda()
  else:
    log_print("Ignoring GPU (CPU only)")

  # Set up loss and optimizer
  loss_f_viewpoint = nn.CrossEntropyLoss()
  loss_f_class = nn.CrossEntropyLoss()
  loss_f_domain = nn.BCELoss()
  if config.GPU and torch.cuda.is_available():
    loss_f_viewpoint  = loss_f_viewpoint.cuda()
    loss_f_class = loss_f_class.cuda()
    loss_f_domain = loss_f_domain.cuda()
  optimizer = optim.SGD(model.parameters(), 
                        lr=config.LEARNING_RATE,
                        momentum=config.MOMENTUM)
  explorer = lr_scheduler.StepLR(optimizer, 
                                       step_size=config.STEP_SIZE,
                                       gamma=config.GAMMA)

  # Perform training
  log_print("!!!!!Starting training!!!!!")
  model = train_model(model, train_dataloader, val_dataloader, loss_f_viewpoint, loss_f_class, loss_f_domain, optimizer, explorer, config.EPOCHS)
  
  # Save model weights
  log_print("Saving model weights to %s..." % config.OUT_WEIGHTS_FP)
  save_model_weights(model, config.OUT_WEIGHTS_FP)

  # Create testing DataLoader
  log_print("Loading testing data...")
  test_dataloader =  \
    create_rend_dataloader(config.DATA_BASE_DIR, config.DATA_TEST_LIST, 'test')

  # Test and output accuracy/predictions
  if config.TEST_AFTER_TRAIN:
    log_print("Testing model on test set...")
    predictions = test_model(model, test_dataloader, loss_f_viewpoint)
    log_print("Writing predictions to %s..." % config.OUT_PRED_FP)
    out_f = open(config.OUT_PRED_FP, 'w')
    for p in predictions:
      out_f.write("%s,%i,%i\n" % (p[0], p[1], p[2]))
    out_f.close()

  log_print("Script DONE!")

if __name__ == "__main__":
    main()
