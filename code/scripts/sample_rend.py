
import os
import shutil
import random

# Input vars
src_dir = "../../data/V2/tvmonitor_shapenet/train"
dest_dir = "../../data/V2/tvmonitor_shapenet_tiny/train"
models_keep = 10
ims_keep = 100

# Get list of images
ims_list = []
for f in os.listdir(src_dir):
    if os.path.isfile(os.path.join(src_dir, f)) and f.split(".")[-1] == "png":
        ims_list.append(f)

ims_dict = {}
for i in ims_list:
    model = int(i.split("_")[0])
    if model not in ims_dict:
        ims_dict[model] = []
    ims_dict[model].append(i)

models = ims_dict.keys()
keep = random.sample(models, min(models_keep, len(models)))

if not os.path.exists(dest_dir):
    os.makedirs(dest_dir)

for m in keep:
    ims = random.sample(ims_dict[m], min(ims_keep, len(ims_dict[m])))
    for i in ims:
        shutil.copy2(os.path.join(src_dir, i), dest_dir)
