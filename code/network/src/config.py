# Filepaths
RUN_NAME = "test"
RUN_DESCRIPTION = "test test test"
DATA_BASE_DIR = "../../../data/V2"
DATA_TRAIN_LIST = ["car_shapenet"]
#DATA_REGULAR_LIST = ["car_imagenet", "car_pascal"]
DATA_REGULAR_LIST = None
DATA_VAL_LIST = ["car_imagenet", "car_pascal"]
DATA_TEST_LIST = ["car_imagenet", "car_pascal"]
OUT_WEIGHTS_FP = "../models/%s.pt" % RUN_NAME
OUT_LOG_FP = "../logs/%s.log" % RUN_NAME
OUT_PRED_FP = "../preds/%s.pred" % RUN_NAME

# Program parameters
GPU = True
MULTI_GPU = True
TEST_AFTER_TRAIN = True
NETWORK_TYPE = "SIMPLE" #VIEWPOINT, VIEWPOINT_CLASS_DOMAIN, SIMPLE
BOTTLENECK_SIZE = 0
AZIMUTH_BINS = 360
ELEVATION_BINS = 360
NUM_OBJ_CLASSES = 12

# Learning parameters
RESNET_LAYERS = 18 #18, 34, 50, 101, 152
PRETRAINED = True
BATCH_SIZE = 60
EPOCHS = 1
LEARNING_RATE = 0.01
MOMENTUM = 0.9
STEP_SIZE = 6
GAMMA = 0.1
LAMBDA_MMD = 0.25

# Debugging print method
def PRINT_CONFIG():
  print "~~~~~ BEGIN CONFIG ~~~~~"
  print " "

  print "~~FILEPATHS~~"
  print "RUN_NAME:\t", RUN_NAME
  print "RUN_DESCRIPTION:\t", RUN_DESCRIPTION
  print "DATA_BASE_DIR:\t", DATA_BASE_DIR
  print "DATA_TRAIN_LIST:\t", DATA_TRAIN_LIST
  print "DATA_REGULAR_LIST:\t", DATA_REGULAR_LIST
  print "DATA_VAL_LIST:\t", DATA_VAL_LIST
  print "DATA_TEST_LIST:\t", DATA_TEST_LIST
  print "OUT_WEIGHTS_FP:\t", OUT_WEIGHTS_FP
  print "OUT_LOG_FP:\t", OUT_LOG_FP
  print "OUT_PRED_FP:\t", OUT_PRED_FP
  print " "

  print "~~PROGRAM PARAMS~~"
  print "GPU:\t", GPU
  print "MULTI_GPU:\t", MULTI_GPU
  print "TEST_AFTER_TRAIN:\t", TEST_AFTER_TRAIN
  print "NETWORK_TYPE:\t", NETWORK_TYPE
  print "BOTTLENECK_SIZE:\t", BOTTLENECK_SIZE
  print "AZIMUTH_BINS:\t", AZIMUTH_BINS
  print "ELEVATION_BINS:\t", ELEVATION_BINS
  print "NUM_OBJ_CLASSES:\t", NUM_OBJ_CLASSES
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
  print "LAMBDA_MMD: \t", LAMBDA_MMD
  print ""

  print "~~~~~~ END CONFIG ~~~~~~"



#DATA_TRAIN_LIST = ["bicycle_imagenet", "bicycle_pascal", "boat_imagenet", "boat_pascal", "bottle_imagenet", "bottle_pascal", "bus_imagenet", "bus_pascal", "car_imagenet", "car_pascal", "chair_imagenet", "chair_pascal", "diningtable_imagenet", "diningtable_pascal", "motorbike_imagenet", "motorbike_pascal", "sofa_imagenet", "sofa_pascal", "train_imagenet", "train_pascal", "tvmonitor_imagenet", "tvmonitor_pascal"]
