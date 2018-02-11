
import torch
import torch.nn.functional as F
from torch import nn
import config
from torch.autograd import Variable

class MMDLoss(nn.Module):
  def __init__(self):
    super(MMDLoss, self).__init__()

  def forward(self, embed, regular_embed):

    assert(embed.size(0) <= regular_embed.size(0))

    embed_avg = torch.zeros(embed.size(1))
    if config.GPU and torch.cuda.is_available():
      embed_avg = Variable(embed_avg.cuda())
    else:
      embed_avg = Variable(embed_avg)
    for inst_id in xrange(embed.size(0)):
      embed_avg += embed[inst_id]
    embed_avg = embed_avg / float(embed.size(0))

    regular_avg = torch.zeros(regular_embed.size(1))
    if config.GPU and torch.cuda.is_available():
      regular_avg = Variable(regular_avg.cuda())
    else:
      regular_avg = Variable(regular_avg)
    for inst_id in xrange(embed.size(0)):
      regular_avg += regular_embed[inst_id]
    regular_avg = regular_avg / float(embed.size(0))

    loss = torch.norm(embed_avg - regular_avg, 2)
    return loss
