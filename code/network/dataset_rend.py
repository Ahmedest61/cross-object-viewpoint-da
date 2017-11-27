import os
import cv2
import torch
from torch.utils.data import Dataset

class RenderedDataset(Dataset):

  """ 
    Instance vars
    - self.data_dir => string (dir holding all ims)
    - self.ims_list => list
    - self.images => dictionary (key=filename, val=annot)
    - self.transform => transformation on image
  """

  def __init__(self, data_dir, data_labels_fp, transform=None):
    
    # Get list of all PNGs in data_dir
    self.data_dir = data_dir
    self.ims_list = [f for f in os.listdir(data_dir) if (os.path.isfile(os.path.join(data_dir, f)) and f.split(".")[-1] == "png")]

    # Load labels
    data_labels_f = open(data_labels_fp, 'r')
    lines = data_labels_f.readlines()
    data_labels_f.close()
    self.images = {}
    for l in lines:
      split = l.strip().split(",")
      im_name = "%s.png" % split[0]
      if im_name in self.ims_list:
        self.images[im_name] = [int(split[1]), int(split[2])]

    # Save transform
    self.transform = transform

  def __len__(self):
    return len(self.ims_list)

  def __getitem__(self, idx):
    im_name = self.ims_list[idx]
    im_fp = os.path.join(self.data_dir, im_name)
    image = cv2.imread(im_fp)
    sample = {"image": image, "annot": self.images[im_name]}
    if self.transform:
        sample = self.transform(sample)
    return sample
