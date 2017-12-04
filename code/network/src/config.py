# Filepaths
RUN_NAME = "viewpoint-loss-test"
RUN_DESCRIPTION = "simple test to make sure viewpoint loss is working"
DATA_BASE_DIR = "../../../data/V1"
#DATA_TRAIN_LIST = ["aeroplane_imagenet", "aeroplane_pascal", "aeroplane_shapenet", "bicycle_imagenet", "bicycle_pascal", "bicycle_shapenet", "boat_imagenet", "boat_pascal", "boat_shapenet", "bottle_imagenet", "bottle_pascal", "bottle_shapenet", "bus_imagenet", "bus_pascal", "bus_shapenet", "car_shapenet_tiny", "chair_imagenet", "chair_pascal", "chair_shapenet", "diningtable_imagenet", "diningtable_pascal", "diningtable_shapenet", "motorbike_imagenet", "motorbike_pascal", "motorbike_shapenet", "sofa_imagenet", "sofa_pascal", "sofa_shapenet", "train_imagenet", "train_pascal", "train_shapenet", "tvmonitor_imagenet", "tvmonitor_pascal", "tvmonitor_shapenet"]
#DATA_TRAIN_LIST = ["aeroplane_imagenet", "aeroplane_pascal", "aeroplane_shapenet", "bicycle_imagenet", "bicycle_pascal", "bicycle_shapenet", "boat_imagenet", "boat_pascal", "boat_shapenet", "bus_imagenet", "bus_pascal", "bus_shapenet", "motorbike_imagenet", "motorbike_pascal", "motorbike_shapenet", "train_imagenet", "train_pascal", "train_shapenet", "car_shapenet_tiny"]
DATA_TRAIN_LIST = ["car_shapenet"]
#DATA_VAL_LIST = ["car_imagenet", "car_pascal"]
DATA_VAL_LIST = ["car_shapenet"]
#DATA_TEST_LIST = ["car_imagenet", "car_pascal"]
DATA_TEST_LIST = ["car_shapenet"]
OUT_WEIGHTS_FP = "../models/%s.pt" % RUN_NAME
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
BATCH_SIZE = 30
EPOCHS = 30
LEARNING_RATE = 0.01
MOMENTUM = 0.9
STEP_SIZE = 6
GAMMA = 0.1
LAMBDA_CLASS = 0
LAMBDA_DOMAIN = 0

# Debugging print method
def PRINT_CONFIG():
  print "~~~~~ BEGIN CONFIG ~~~~~"
  print " "

  print "~~FILEPATHS~~"
  print "RUN_NAME:\t", RUN_NAME
  print "RUN_DESCRIPTION:\t", RUN_DESCRIPTION
  print "DATA_BASE_DIR:\t", DATA_BASE_DIR
  print "DATA_TRAIN_LIST:\t", DATA_TRAIN_LIST
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
  print "LAMBDA_CLASS:\t", LAMBDA_CLASS
  print "LAMBDA_DOMAIN:\t", LAMBDA_DOMAIN
  print ""

  print "~~~~~~ END CONFIG ~~~~~~"



#DATA_TRAIN_LIST = ["bicycle_imagenet", "bicycle_pascal", "boat_imagenet", "boat_pascal", "bottle_imagenet", "bottle_pascal", "bus_imagenet", "bus_pascal", "car_imagenet", "car_pascal", "chair_imagenet", "chair_pascal", "diningtable_imagenet", "diningtable_pascal", "motorbike_imagenet", "motorbike_pascal", "sofa_imagenet", "sofa_pascal", "train_imagenet", "train_pascal", "tvmonitor_imagenet", "tvmonitor_pascal"]
