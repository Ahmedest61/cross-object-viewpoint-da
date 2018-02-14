# Filepaths
RUN_NAME = "test-car-rend2real"
RUN_DESCRIPTION = "trained on rend data, testing on real data"
DATA_BASE_DIR = "../../../data/V2"
DATA_TEST_LIST = ["car_imagenet", "car_pascal"]
IN_WEIGHTS_FP = "../models/car-rend-supervised.pt"
OUT_LOG_FP = "../logs/%s.log" % RUN_NAME
OUT_PRED_FP = "../preds/%s.pred" % RUN_NAME

# Program parameters
GPU = True
MULTI_GPU = True
TEST_AFTER_TRAIN = True
NETWORK_TYPE = "VIEWPOINT_CLASS_DOMAIN" #VIEWPOINT, VIEWPOINT_CLASS_DOMAIN
AZIMUTH_BINS = 360
ELEVATION_BINS = 360
NUM_OBJ_CLASSES = 12

# Learning parameters
RESNET_LAYERS = 18 #18, 34, 50, 101, 152
PRETRAINED = True
BATCH_SIZE = 60

# Debugging print method
def PRINT_CONFIG():
  print "~~~~~ BEGIN CONFIG ~~~~~"
  print " "

  print "~~FILEPATHS~~"
  print "RUN_NAME:\t", RUN_NAME
  print "RUN_DESCRIPTION:\t", RUN_DESCRIPTION
  print "DATA_BASE_DIR:\t", DATA_BASE_DIR
  print "DATA_TEST_LIST:\t", DATA_TEST_LIST
  print "IN_WEIGHTS_FP:\t", IN_WEIGHTS_FP
  print "OUT_LOG_FP:\t", OUT_LOG_FP
  print "OUT_PRED_FP:\t", OUT_PRED_FP
  print " "

  print "~~PROGRAM PARAMS~~"
  print "GPU:\t", GPU
  print "MULTI_GPU:\t", MULTI_GPU
  print "TEST_AFTER_TRAIN:\t", TEST_AFTER_TRAIN
  print "NETWORK_TYPE:\t", NETWORK_TYPE
  print "AZIMUTH_BINS:\t", AZIMUTH_BINS
  print "ELEVATION_BINS:\t", ELEVATION_BINS
  print "NUM_OBJ_CLASSES:\t", NUM_OBJ_CLASSES
  print ""

  print "~~LEARNING PARAMS~~"
  print "RESNET_LAYERS:\t", RESNET_LAYERS
  print "PRETRAINED:\t", PRETRAINED
  print "BATCH_SIZE:\t", BATCH_SIZE
  print ""

  print "~~~~~~ END CONFIG ~~~~~~"
