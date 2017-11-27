import os
import sys
import config
import datetime
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import transforms, models

# Imports from src files
from dataset_rend import RenderedDataset

#####################
#   BEGIN HELPERS   #
#####################

def log_print(string):
  print "[%s]\t %s" % (datetime.datetime.now(), string)

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

def train_model(model, train_dataloader, val_dataloader, loss_f, optimizer, explorer, epochs):

  init_time = time.time()
  best_acc = 0.0
  best_weights = model.state_dict()
  best_epoch = -1

  for epoch in xrange(epochs):
    log_print("Epoch %i/%i" % (epoch+1, epochs))

    for phase in ["train", "val"]:
      if phase == "train":
        explorer.step()
        model.train(True)
        dataloader = train_dataloader
      else:
        model.train(False)
        dataloader = val_dataloader

      # Iterate over dataset
      epoch_loss = epoch_correct = curr_loss = curr_correct = 0
      print_interval = 100
      batch_count = 0
      for data in dataloader:
        inputs, annots = data['image'], data['annot']

        # Wrap as pytorch autograd Variable
        if config.GPU and torch.cuda.is_available():
          inputs = Variable(inputs.cuda())
          annots = Variable(annots.cuda())
        else:
          inputs = Variable(inputs)
          annots= Variable(annots)

        # Forward pass and calculate loss
        optimizer.zero_grad()
        outputs = model(inputs)
        _, preds = torch.max(outputs.data, 1)
        loss = loss_f(outputs, annots)
        curr_loss += loss.data[0]
        curr_correct += torch.sum(preds == annots.data)

        # Backward pass (if train)
        if phase == "train":
          loss.backward()
          optimizer.step()

        # Output
        #if batch_count != 0 and batch_count % (print_interval-1) == 0:
        if batch_count % print_interval == 0 and batch_count != 0:
          epoch_loss += curr_loss
          epoch_correct += curr_correct
          if phase == "train":
            curr_loss = float(curr_loss) / float(print_interval*config.BATCH_SIZE)
            curr_correct = float(curr_correct) / float(print_interval*config.BATCH_SIZE)
            log_print("\tBatches %i-%i -\tLoss: %f \t Acc: %f" % (batch_count-print_interval+1, batch_count, curr_loss, curr_correct))
          curr_loss = curr_correct = 0
        batch_count += 1
      
      # Report epoch results
      num_images = len(dataloader.dataset)
      epoch_loss = float(epoch_loss+curr_loss) / float(num_images)
      epoch_acc = float(epoch_correct+curr_correct) / float(num_images)
      log_print("\tEPOCH %i [%s] - Loss: %f   Acc: %f" % (epoch+1, phase, epoch_loss, epoch_acc))

      # Save best model weights from epoch
      if phase == "val" and epoch_acc >= best_acc:
        best_acc = epoch_acc
        best_loss = epoch_loss
        best_weights = model.state_dict()
        best_epoch = epoch

  # Finish up
  time_elapsed = time.time() - init_time
  log_print("BEST EPOCH: %i/%i - Loss: %f   Acc: %f" % (best_epoch+1, epochs, best_loss, best_acc))
  log_print("Training completed in %sm %ss" % (time_elapsed // 60, time_elapsed % 60))
  model.load_state_dict(best_weights)
  return model

def save_model_weights(model, filepath):
  torch.save(model.state_dict(), filepath)

def test_model(model, test_dataloader, loss_f):
  test_loss = test_correct = 0
  print_interval = 100
  batch_count = 0
  for data in test_dataloader:
    inputs, annots = data['image'], data['annot']

    # Wrap as pytorch autograd Variable
    if config.GPU and torch.cuda.is_available():
      inputs = Variable(inputs.cuda())
      annots = Variable(annots.cuda())
    else:
      inputs = Variable(inputs)
      annots= Variable(annots)

    # Forward pass and calculate loss
    outputs = model(inputs)
    _, preds = torch.max(outputs.data, 1)
    loss = loss_f(outputs, annots)
    test_loss += loss.data[0]
    test_correct += torch.sum(preds == annots.data)

  # Report epoch results
  num_images = len(test_dataloader.dataset)
  test_loss = float(test_loss) / float(num_images)
  test_acc = float(test_correct) / float(num_images)
  log_print("[TEST SET] %i ims - Loss: %f   Acc: %f" % (num_images, test_loss, test_acc))

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
  train_ims_dir = os.path.join(config.DATA_DIR, 'train')
  train_dataloader =  \
    create_rend_dataloader(train_ims_dir, config.DATA_LABELS_FP)

  # Create validation DataLoader
  log_print("Loading validation data...")
  val_ims_dir = os.path.join(config.DATA_DIR, 'val')
  val_dataloader = \
    create_rend_dataloader(val_ims_dir, config.DATA_LABELS_FP)

  # Set up model for training
  log_print("Creating model...")
  model = create_model()
  if config.GPU and torch.cuda.is_available():
    log_print("Enabling GPU")
    if config.MULTI_GPU and torch.cuda.device_count() > 1:
      log_print("Using multiple GPUs: %i" % torch.cuda.device_count())
      model = nn.DataParallel(model)
    model = model.cuda()

  # Set up loss and optimizer
  loss_f = nn.CrossEntropyLoss()
  optimizer = optim.SGD(model.parameters(), 
                        lr=config.LEARNING_RATE,
                        momentum=config.MOMENTUM)
  explorer = lr_scheduler.StepLR(optimizer, 
                                       step_size=config.STEP_SIZE,
                                       gamma=config.GAMMA)

  # Perform training
  log_print("!!!!!Starting training!!!!!")
  model = train_model(model, train_dataloader, val_dataloader, loss_f, optimizer, explorer, config.EPOCHS)
  
  # Save model weights
  log_print("Saving model weights to %s..." % config.OUT_WEIGHTS_FP)
  save_model_weights(model, config.OUT_WEIGHTS_FP)

  # Create testing DataLoader
  log_print("Loading testing data...")
  test_ims_dir = os.path.join(config.DATA_DIR, 'test')
  test_dataloader =  \
    create_rend_dataloader(test_ims_dir, config.DATA_LABELS_FP)

  # Test and report accuracy
  if config.TEST_AFTER_TRAIN:
    log_print("Testing model on test set...")
    accuracy = test_model(model, test_dataloader, loss_f)

  log_print("Script DONE!")

if __name__ == "__main__":
    main()
