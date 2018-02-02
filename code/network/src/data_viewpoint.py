import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset

class_ids = {
  "aeroplane": 0,
  "bicycle": 1,
  "boat": 2,
  "bottle": 3,
  "bus": 4,
  "car": 5,
  "chair": 6,
  "diningtable": 7,
  "motorbike": 8,
  "sofa": 9,
  "train": 10,
  "tvmonitor": 11
}

domain_ids = {
  "shapenet": 0,
  "imagenet": 1,
  "pascal": 1
}

class ViewpointDataset(Dataset):

  """ 
    Instance vars
    - self.data_dir => string (dir holding all ims)
    - self.ims_list => list
    - self.images => dictionary (key=filename, val=annot)
    - self.transform => transformation on image
  """

  def __init__(self, data_base_dir, data_list, data_set, transform=None):
    
    # Get list of all PNGs in data_dir
    self.data_base_dir = data_base_dir
    init_list  = []
    for d in data_list:
      full_d = os.path.join(data_base_dir, d, data_set)
      init_list += [os.path.join(full_d, f).strip() for f in os.listdir(full_d) if (os.path.isfile(os.path.join(full_d, f)) and f.split(".")[-1] == "png")]

    # Load labels
    ims_set = set(init_list)
    self.images = {}
    for d in data_list:
      data_labels_fp = os.path.join(data_base_dir, d, 'annots.txt')
      data_labels_f = open(data_labels_fp, 'r')
      lines = data_labels_f.readlines()
      data_labels_f.close()
      for l in lines:
        split = l.strip().split(",")
        if d.split("_")[1] == "shapenet":
          im_name = "{}.png".format(split[0])
        else:
          im_name = "{}_0.png".format(split[0])
        im_fp = os.path.join(data_base_dir, d, data_set, im_name).strip()
        #print im_fp
        if im_fp in ims_set:
          self.images[im_fp] = [int(split[1]), int(split[2])]

    # Save vars
    self.ims_list = self.images.keys()
    self.transform = transform

  def __len__(self):
    return len(self.ims_list)

  def __getitem__(self, idx):
    # Fetch image
    im_fp = self.ims_list[idx].strip()
    image = cv2.imread(im_fp)

    # Fetch angle annots
    annot = self.images[im_fp]    #(azimuth, elevation)
    azimuth = annot[0]
    elevation = annot[1]
    if elevation < 0:
      elevation = 360 + elevation

    # Fetch class and domain annots
    dataset_name = im_fp.split("/")[-3].split(".")[0]
    class_id = int(class_ids[dataset_name.split("_")[0]])
    domain_id = int(domain_ids[dataset_name.split("_")[1]])

    # Return sample
    sample = {"image_fp": im_fp, "image": image, "azimuth": azimuth, "elevation": elevation, "class_id": class_id, "domain_id": domain_id}
    if self.transform:
        sample["image"] = self.transform(sample["image"])
    return sample
