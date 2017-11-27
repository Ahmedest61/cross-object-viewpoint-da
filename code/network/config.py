# Filepaths
DATA_DIR = "../../data/shapenet/chair/V1"
DATA_LABELS_FP = "../../data/shapenet/chair/V1/annots.txt"

# Program parameters
GPU = True
MULTI_GPU = True
TEST_AFTER_TRAIN = True

# Learning parameters
MAX_EPOCHS = 20
BATCH_SIZE = 30

# Debugging print method
def PRINT_CONFIG():
  print "~~~~~ BEGIN CONFIG ~~~~~"
  print " "

  print "DATA_DIR:\t", DATA_DIR
  print "DATA_LABELS_FP:\t", DATA_LABELS_FP
  print " "

  print "GPU:\t", GPU
  print "MULTI_GPU:\t", MULTI_GPU
  print "TEST_AFTER_TRAIN:\t", TEST_AFTER_TRAIN
  print ""

  print "BATCH_SIZE:\t", BATCH_SIZE

  print "~~~~~~ END CONFIG ~~~~~~"
  pass
