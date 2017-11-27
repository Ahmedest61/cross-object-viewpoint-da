# Filepaths
DATA_DIR = "../../data/shapenet/chair/V1"
DATA_LABELS_FP = "../../data/shapenet/chair/V1/annots.txt"

# Program parameters
GPU = True
MULTI_GPU = True
TEST_AFTER_TRAIN = True
AZIMUTH_BINS = 360
ELEVATION_BINS = 135

# Learning parameters
PRETRAINED = True
MAX_EPOCHS = 20
BATCH_SIZE = 30

# Debugging print method
def PRINT_CONFIG():
  print "~~~~~ BEGIN CONFIG ~~~~~"
  print " "

  print "~~FILEPATHS~~"
  print "DATA_DIR:\t", DATA_DIR
  print "DATA_LABELS_FP:\t", DATA_LABELS_FP
  print " "

  print "~~PROGRAM PARAMS~~"
  print "GPU:\t", GPU
  print "MULTI_GPU:\t", MULTI_GPU
  print "TEST_AFTER_TRAIN:\t", TEST_AFTER_TRAIN
  print "AZIMUTH_BINS:\t", AZIMUTH_BINS
  print "ELEVATION_BINS:\t", ELEVATION_BINS
  print ""

  print "~~LEARNING PARAMS~~"
  print "PRETRAINED:\t", PRETRAINED
  print "MAX_EPOCHS:\t", MAX_EPOCHS
  print "BATCH_SIZE:\t", BATCH_SIZE

  print "~~~~~~ END CONFIG ~~~~~~"
