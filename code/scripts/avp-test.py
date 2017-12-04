# Script to calculate AVP for certain bin size, given preds file and gt annots file

# Input vars
preds_fp = "../network/preds/chair-avp-SupervisedBoth-DANN.pred"
annots_fp1 = "../../data/V2/chair_imagenet/annots.txt" 
annots_fp2 = "../../data/V2/chair_pascal/annots.txt" 
bin_views = 16 #4, 8, 16, 24

# Read preds
preds_f = open(preds_fp, 'r')
preds = [f.strip().split(",") for f in preds_f.readlines()]
preds_f.close()

# Read annots
annots_f = open(annots_fp1, 'r')
annots = [f.strip().split(",") for f in annots_f.readlines()]
annots_f.close()
annots_f = open(annots_fp2, 'r')
annots += [f.strip().split(",") for f in annots_f.readlines()]
annots_f.close()

# Index annots
annots_map = {}
for a in annots:
  annots_map[a[0]] = [a[1], a[2]]

# Init
bin_size = 360 / bin_views
correct_az = 0
correct_ele = 0
total_az = 0
total_ele = 0

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

  # Bin and test
  pred_az_bin = pred_az / bin_size
  pred_ele_bin = pred_ele / bin_size
  annot_az_bin = annot_az / bin_size
  annot_ele_bin = annot_ele / bin_size
  if pred_az_bin == annot_az_bin:
    correct_az += 1
  if pred_ele_bin == annot_ele_bin:
    correct_ele += 1
  total_az += 1
  total_ele += 1

# Report AVP
print "~"*10
print preds_fp
print "BINS: %iV" % bin_views
print "AZIMUTH AVP: %f" % (float(correct_az)/float(total_az))
print "ELEVATION AVP: %f" % (float(correct_ele)/float(total_ele))
