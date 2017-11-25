# Script to split list of renderings .pngs into a training and testing set

import os
import shutil
import random

# Input vars
MOVE = True
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
split_train = list(set(models) - set(split_test))

# Write split files
train_fp = os.path.join(data_dir, "train_split.txt")
train_f = open(train_fp, 'w')
for t in split_train:
    for i in ims_dict[t]:
        train_f.write("%s\n" % i)
train_f.close()

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
    test_dir = os.path.join(data_dir, "test")
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)

    # Move each file
    for m in models:
        if m in split_train:
            for i in ims_dict[m]:
                src_fp = os.path.join(data_dir, i)
                dest_fp = os.path.join(train_dir, i)
                shutil.move(src_fp, dest_fp)
        else:
            for i in ims_dict[m]:
                src_fp = os.path.join(data_dir, i)
                dest_fp = os.path.join(test_dir, i)
                shutil.move(src_fp, dest_fp)
