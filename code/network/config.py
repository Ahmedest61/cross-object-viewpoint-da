# Filepaths
DATA_DIR = "../../data/shapenet/chair/V1"
DATA_LABELS_FP = "../../data/shapenet/chair/V1/annots.txt"
#TODO

# Running parameters
GPU = True
MULTI_GPU = True
TEST_AFTER_TRAIN = True

# Learning parameters
#TODO

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

  print "~~~~~~ END CONFIG ~~~~~~"
  pass
