import os
import random
import numpy as np
import matplotlib
#matplotlib.use('agg')
import pylab as plt

# INPUTS
pred_name = "multi-test-both.pred"
gts = ["car_imagenet", "car_pascal"]
sample_size = 1000;
DISP = "PREDS" # 'PREDS', 'GT', or 'BOTH'
x_min, x_max = -20, 380
y_min, y_max = -20, 60

base_dir = os.path.join("..", "..")
preds_fp = os.path.join(base_dir, "code", "network", "preds", pred_name)
gts_fp = [os.path.join(base_dir, "data", "V2", g, "annots.txt") for g in gts]

# Preprocess preds
preds_f = open(preds_fp, 'r')
preds_lines = [l.strip() for l in preds_f.readlines()]
preds_f.close()
preds = {}
for l in preds_lines:
    split = l.split(",")
    p_id = split[0].split("/")[-1].split(".")[0]
    p_id_split = p_id.split("_")
    if len(p_id_split) > 2:
        p_id = "_".join(p_id_split[:-1])

    if p_id in preds:
        print "COLLISION ON KEY"
    preds[p_id] = [int(split[1]), int(split[2])]

# Preprocess gt files
annots = {}
for gt_fp in gts_fp:
    gt_f = open(gt_fp, 'r')
    gt_lines = [l.strip() for l in gt_f.readlines()]
    gt_f.close()
    for l in gt_lines:
        split = l.split(",")
        annots[split[0]] = [int(split[1]), int(split[2])]

# Subsample if necessary
keys_sample = random.sample(preds.keys(), min(len(preds.keys()), sample_size))
preds_sample = {}
for k in keys_sample:
    preds_sample[k] = preds[k]
preds = preds_sample

# Convert data points to numpy
  # Preds
preds_x = np.array([])
preds_y = np.array([])
for p in preds.keys():
    preds_x = np.append(preds_x, preds[p][0])
    preds_y = np.append(preds_y, preds[p][1])

  # Annots
count = 0
gt_x = np.array([])
gt_y = np.array([])
for p in preds.keys():
    if p not in annots:
        count += 1
        continue
    gt_x = np.append(gt_x, annots[p][0])
    gt_y = np.append(gt_y, annots[p][1])
if count != 0:
    print "%i/%i data points not in gt" % (count, len(preds.keys()))

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
if DISP == "PREDS" or DISP == "BOTH":
    ax.scatter(preds_x, preds_y, alpha=0.8, c='blue', edgecolors='none', s=30, label='pred')
if DISP == "GT" or DISP == "BOTH":
    ax.scatter(gt_x, gt_y, alpha=0.8, c='red', edgecolors='none', s=30, label='gt')
plt.title('Predictions vs Ground Truth:\n %s, %s ' % (pred_name, "&".join(gts)))
plt.xlabel('Azimuth (in degrees)')
plt.ylabel('Elevation (in degrees)')
plt.xlim([x_min, x_max])
plt.ylim([y_min, y_max])
plt.legend(loc=2)
plt.show()
