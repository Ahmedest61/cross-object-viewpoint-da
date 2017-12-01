# Filepaths
RUN_NAME = "TEST"
DATA_BASE_DIR = "../../../data"
DATA_TRAIN_LIST = ["Bicycle_shapenet", "Boat_shapenet", "Bottle_shapenet", "Bus_shapenet", "Car_shapenet", "Chair_shapenet", "DiningTable_shapenet", "Motorbike_shapenet", "Sofa_shapenet", "Train_shapenet", "TvMonitor_shapenet"]
DATA_VAL_LIST = ["Aeroplane_shapenet"]
DATA_TEST_LIST = ["Aeroplane_shapenet"]
OUT_WEIGHTS_FP = "../models/%s.pt" % RUN_NAME
OUT_LOG_FP = "../logs/%s.log" % RUN_NAME
OUT_PRED_FP = "../preds/%s.pred" % RUN_NAME

# Program parameters
GPU = True
MULTI_GPU = True
TEST_AFTER_TRAIN = True
AZIMUTH_BINS = 360
ELEVATION_BINS = 360

# Learning parameters
RESNET_LAYERS = 18 #18, 34, 50, 101, 152
PRETRAINED = True
BATCH_SIZE = 30
EPOCHS = 4
LEARNING_RATE = 0.01
MOMENTUM = 0.9
STEP_SIZE = 6
GAMMA = 0.1

# Debugging print method
def PRINT_CONFIG():
  print "~~~~~ BEGIN CONFIG ~~~~~"
  print " "

  print "~~FILEPATHS~~"
  print "RUN_NAME:\t", RUN_NAME
  print "DATA_BASE_DIR:\t", DATA_BASE_DIR
  print "DATA_TRAIN_LIST:\t", DATA_TRAIN_LIST
  print "DATA_VAL_LIST:\t", DATA_TRAIN_LIST
  print "DATA_TEST_LIST:\t", DATA_TEST_LIST
  print "OUT_WEIGHTS_FP:\t", OUT_WEIGHTS_FP
  print "OUT_LOG_FP:\t", OUT_LOG_FP
  print "OUT_PRED_FP:\t", OUT_PRED_FP
  print " "

  print "~~PROGRAM PARAMS~~"
  print "GPU:\t", GPU
  print "MULTI_GPU:\t", MULTI_GPU
  print "TEST_AFTER_TRAIN:\t", TEST_AFTER_TRAIN
  print "AZIMUTH_BINS:\t", AZIMUTH_BINS
  print "ELEVATION_BINS:\t", ELEVATION_BINS
  print ""

  print "~~LEARNING PARAMS~~"
  print "RESNET_LAYERS:\t", RESNET_LAYERS
  print "PRETRAINED:\t", PRETRAINED
  print "BATCH_SIZE:\t", BATCH_SIZE
  print "EPOCHS:\t", EPOCHS
  print "LEARNING_RATE:\t", LEARNING_RATE
  print "MOMENTUM:\t", MOMENTUM
  print "STEP_SIZE:\t", STEP_SIZE
  print "GAMMA:\t", GAMMA
  print ""

  print "~~~~~~ END CONFIG ~~~~~~"
