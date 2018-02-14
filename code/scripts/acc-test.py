# Script to calculate Acc-theta for certain bin size, given preds file and gt annots file

import sys
from math import pi, acos, cos, sin

# Input vars
preds_fp = "../network/preds/multi-test-mmd.pred"
annots_fps = [ \
  "../../data/V2/car_imagenet/annots.txt", \
  "../../data/V2/car_pascal/annots.txt" \
]
thetas = [2, 3, 4, 6]

def degree_to_radian(degree):
  return ((float(degree)*pi)/180.0)

# Read preds
preds_f = open(preds_fp, 'r')
preds = [f.strip().split(",") for f in preds_f.readlines()]
preds_f.close()

# Read annots
annots = []
for annots_fp in annots_fps:
  annots_f = open(annots_fp, 'r')
  annots += [f.strip().split(",") for f in annots_f.readlines()]
  annots_f.close()

# Index annots
annots_map = {}
for a in annots:
  annots_map[a[0]] = [a[1], a[2]]

# Init
for theta in thetas:
  correct = 0
  total = 0

  theta_pi = pi / float(theta)

  # Calc AVP
  for p in preds:
    full_fp = p[0]
    p_index = (full_fp.split("/")[-1]).split(".")[0][:-2]
    annot_val = annots_map[p_index]
  
    pred_az = int(p[1])
    pred_ele = int(p[2])
    annot_az = int(annot_val[0])
    annot_ele = int(annot_val[1])

    # fix elevation
    if pred_ele < 0:
      pred_ele += 360
    if annot_ele < 0:
      annot_ele += 360

    pred_az = degree_to_radian(pred_az)
    pred_ele = degree_to_radian(pred_ele)
    annot_az = degree_to_radian(annot_az)
    annot_ele = degree_to_radian(annot_ele)

    # Bin and test
    inside = sin(pred_ele)*sin(annot_ele) + cos(pred_ele)*cos(annot_ele)*cos(abs(pred_az-annot_az))
    if abs(inside - 1.0) < 0.0001:
        correct += 1
        total += 1
        continue
    diff = acos(inside)
    if diff <= theta_pi:
      correct += 1
    total += 1
  
  # Report AVP
  print "~"*10
  print preds_fp
  print "THETA: PI/%i" % theta
  print "ACCURACY: %f (%i / %i)" % ((float(correct)/float(total)), correct, total)
