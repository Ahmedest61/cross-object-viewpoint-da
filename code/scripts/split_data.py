# Script to split list of renderings .pngs into a training, validation, and testing set

import os
import shutil
import random

# Input vars
MOVE = True
val_percent = .15
test_percent = .15
data_dir = "../../data/shapenet/chair/V1"

# Get list of images
ims_list = []
for f in os.listdir(data_dir):
    if os.path.isfile(os.path.join(data_dir, f)) and f.split(".")[-1] == "png":
        ims_list.append(f)

# Organize by model
ims_dict = {}
for i in ims_list:
    model = int(i.split("_")[0])
    if model not in ims_dict:
        ims_dict[model] = []
    ims_dict[model].append(i)

# Determine test split
models = ims_dict.keys()
num_models = len(models)
num_test = int(num_models * test_percent)
split_test = random.sample(models, num_test)

# Determine val split
num_val = int(num_models * val_percent)
remaining = list(set(model) - set(split_test))
split_val = random.sample(remaining, num_val)
split_train = list(set(remaining)) - set(split_val))

# Write split files
train_fp = os.path.join(data_dir, "train_split.txt")
train_f = open(train_fp, 'w')
for t in split_train:
    for i in ims_dict[t]:
        train_f.write("%s\n" % i)
train_f.close()

val_fp = os.path.join(data_dir, "val_split.txt")
val_f = open(val_fp, 'w')
for t in split_val:
    for i in ims_dict[t]:
        val_f.write("%s\n" % i)
val_f.close()

test_fp = os.path.join(data_dir, "test_split.txt")
test_f = open(test_fp, 'w')
for t in split_test:
    for i in ims_dict[t]:
        test_f.write("%s\n" % i)
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
    for m in models:
        if m in split_train:
            for i in ims_dict[m]:
                src_fp = os.path.join(data_dir, i)
                dest_fp = os.path.join(train_dir, i)
                shutil.move(src_fp, dest_fp)
        elif m in split_val:
            for i in ims_dict[m]:
                src_fp = os.path.join(data_dir, i)
                dest_fp = os.path.join(val_dir, i)
                shutil.move(src_fp, dest_fp)
        elif m in split_test:
            for i in ims_dict[m]:
                src_fp = os.path.join(data_dir, i)
                dest_fp = os.path.join(test_dir, i)
                shutil.move(src_fp, dest_fp)
