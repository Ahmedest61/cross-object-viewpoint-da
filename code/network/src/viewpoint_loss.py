
import config
import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable

class ViewpointLoss(nn.Module):
  
  def __init__(self):
    super(ViewpointLoss, self).__init__()
    #TODO
    pass

  def forward(self, outputs, annots):
    
    print "output", outputs.requires_grad
    _, preds = torch.max(outputs, 1)
    print "preds", preds.requires_grad
    print "_", _.requires_grad
    annots = annots.float()
    preds = preds.float()
    batch_size = outputs.size(0)
    loss = torch.zeros(1)

    # Enable GPU and Autograd
    if config.GPU and torch.cuda.is_available():
      #loss = Variable(loss.cuda(), requires_grad=True)
      loss = Variable(loss.cuda())
    else:
      #loss = Variable(loss, requires_grad=True)
      loss = Variable(loss)

    for i in xrange(preds.size()[0]):
      pred_i = preds[i]
      annot_i = annots[i]

      out_azimuth = torch.floor(pred_i / config.ELEVATION_BINS)
      out_elevation = torch.floor((pred_i % config.ELEVATION_BINS) - (config.ELEVATION_BINS - 90) + 1)
      annot_azimuth = torch.floor(annot_i / config.ELEVATION_BINS)
      annot_elevation = torch.floor((annot_i % config.ELEVATION_BINS) - (config.ELEVATION_BINS - 90) + 1)

      one = torch.abs(annot_azimuth - out_azimuth)
      two = 360 - one
      azimuth_err = torch.min(one, two)
      elevation_err = torch.abs(annot_elevation - out_elevation)
      geodesic_dist = azimuth_err + elevation_err
      if i == 0:
        loss = geodesic_dist
      else:
        loss += geodesic_dist

    out_loss = loss/batch_size
    print "BATCH LOSS:", out_loss 
    return out_loss
"""
    for i in xrange(batch_size):
      output_i = outputs[i]
      annot_i = annots[i]
      annot_azimuth = annot_i / config.ELEVATION_BINS
      annot_elevation = (annot_i % config.ELEVATION_BINS) - (config.ELEVATION_BINS - 90) + 1

      for j in xrange(output_i.size()[0]):
        out_azimuth = j / config.ELEVATION_BINS
        out_elevation = (j % config.ELEVATION_BINS) - (config.ELEVATION_BINS - 90) + 1

        # Calc error
        one = torch.abs(annot_azimuth - out_azimuth)
        two = 360 - one
        azimuth_err = torch.min(one, two)
        elevation_err = torch.abs(annot_elevation - out_elevation)
        geodesic_dist = azimuth_err + elevation_err
        loss += geodesic_dist
"""
    
    
