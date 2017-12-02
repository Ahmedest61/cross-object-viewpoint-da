# Script to split list of renderings .pngs into a training, validation, and testing set

import os
import shutil
import random

# Input vars
MOVE = True
val_percent = .15
test_percent = .15
data_dir = "../../data/tvmonitor_imagenet"

# Get list of images
ims_list = []
for f in os.listdir(data_dir):
    if os.path.isfile(os.path.join(data_dir, f)) and f.split(".")[-1] == "png":
        ims_list.append(f)

# Determine splits
num_ims = len(ims_list)
num_test = int(num_ims * test_percent)
split_test = random.sample(ims_list, num_test)
num_val = int(num_ims * val_percent)
remaining = list(set(ims_list) - set(split_test))
split_val = random.sample(remaining, num_val)
split_train = list(set(remaining) - set(split_val))

# Write split files
train_fp = os.path.join(data_dir, "train_split.txt")
train_f = open(train_fp, 'w')
for t in split_train:
    train_f.write("%s\n" % t)
train_f.close()

val_fp = os.path.join(data_dir, "val_split.txt")
val_f = open(val_fp, 'w')
for t in split_val:
    val_f.write("%s\n" % t)
val_f.close()

test_fp = os.path.join(data_dir, "test_split.txt")
test_f = open(test_fp, 'w')
for t in split_test:
    test_f.write("%s\n" % t)
test_f.close()

# Move into dirs if necessary
if MOVE:
    # Create dirs
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")
    test_dir = os.path.join(data_dir, "test")
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    if not os.path.exists(val_dir):
        os.makedirs(val_dir)
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)

    # Move each file
    for i in split_train:
        src_fp = os.path.join(data_dir, i)
        dest_fp = os.path.join(train_dir, i)
        shutil.move(src_fp, dest_fp)
    for i in split_val:
        src_fp = os.path.join(data_dir, i)
        dest_fp = os.path.join(val_dir, i)
        shutil.move(src_fp, dest_fp)
    for i in split_test:
        src_fp = os.path.join(data_dir, i)
        dest_fp = os.path.join(test_dir, i)
        shutil.move(src_fp, dest_fp)
