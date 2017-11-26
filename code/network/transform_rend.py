import torch

class ToTensor(object):
  """ Convert ndarrays to Tensors"""

  def __call__(self, sample):
    image, annot = sample['image'], sample['annot']
    image = image.transpose((2,0,1))
    return {'image': torch.from_numpy(image),
            'annot': torch.from_numpy(annot)}
