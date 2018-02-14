import sys
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
import matplotlib
import pylab as plt
from sklearn.decomposition import PCA

sys.path.insert(0, "../network/src")
from data_viewpoint import ViewpointDataset
import models

#WEIGHTS_FP = "../network/models/test-supervised.pt"
WEIGHTS_FP = "../network/models/multi-test-normal.pt"
DATA_BASE_DIR = "../../data/V2"
#DATA_LIST = ["car_imagenet", "car_pascal"]
#DATA_LIST = ["aeroplane_imagenet", "aeroplane_pascal", "bicycle_imagenet", "bicycle_pascal", "boat_imagenet", "boat_pascal", "bottle_imagenet", "bottle_pascal", "bus_imagenet", "bus_pascal", "chair_imagenet", "chair_pascal", "diningtable_imagenet", "diningtable_pascal", "motorbike_imagenet", "motorbike_pascal", "sofa_imagenet", "sofa_pascal", "train_imagenet", "train_pascal", "tvmonitor_imagenet", "tvmonitor_pascal"]
DATA_LIST = ["aeroplane_imagenet", "aeroplane_pascal", "bicycle_imagenet", "bicycle_pascal", "boat_imagenet", "boat_pascal", "bottle_imagenet", "bottle_pascal", "bus_imagenet", "bus_pascal", "car_imagenet", "car_pascal", "chair_imagenet", "chair_pascal", "diningtable_imagenet", "diningtable_pascal", "motorbike_imagenet", "motorbike_pascal", "sofa_imagenet", "sofa_pascal", "train_imagenet", "train_pascal", "tvmonitor_imagenet", "tvmonitor_pascal"]
DATA_SET = "test"
BOTTLENECK_SIZE = 0
x_min, x_max = -40, 40
y_min, y_max = -40, 40

def extract_feat(model, dataloader):
  
  flag = False

  for data in dataloader:
    inputs = data['image']
    if torch.cuda.is_available():
      inputs = Variable(inputs.cuda())
    else:
      inputs = Variable(inputs)

    _ = model(inputs)
    embeddings = model.get_embedding()
    embeddings = embeddings.cpu().data.numpy()
    if not flag:
      X = embeddings
      flag = True
    else:
      X = np.append(X, embeddings, axis=0)

  return X

def main():
  
  # Create model
  print "Creating model..."
  model = models.resnet_experiment(bottleneck_size = BOTTLENECK_SIZE, layers=18)
  model.load_state_dict(torch.load(WEIGHTS_FP))
  if torch.cuda.is_available():
    model = model.cuda()

  # Create dataloader
  print "Creating dataloader..."
  dataset = ViewpointDataset(data_base_dir=DATA_BASE_DIR,
                             data_list=DATA_LIST,
                             data_set=DATA_SET,
                             transform=transforms.Compose([transforms.ToTensor()]))
  dataloader = DataLoader(dataset,
                          batch_size=30,
                          shuffle=True,
                          num_workers=4)

  # Perform PCA on features
  print "Extracting features..."
  X = extract_feat(model, dataloader) # (n,d) - n images, d features per image
  print "Performing PCA..."
  pca = PCA(n_components=2)
  pca.fit(X)
  X_pca = pca.transform(X)

  # Plot features
  print "Plotting PCA features..."
  fig = plt.figure()
  ax = fig.add_subplot(1,1,1)
  ax.scatter(X_pca[:,0], X_pca[:,1], alpha=0.2)
  plt.title("Representation - 2D PCA \n %s" % WEIGHTS_FP)
  plt.xlim([x_min, x_max])
  plt.ylim([y_min, y_max])
  plt.show()

if __name__ == "__main__":
  main()
