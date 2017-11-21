
import os

base_dir = "/data/ShapeNetCore.v1/03001627"
out_fp = "./models-chair.txt"

out_files = []
for root, dirs, files in os.walk(base_dir):
    if root.split("/")[-1] == "images":
        continue
    for name in files:
        if name.endswith(('.obj')):
            out_files.append(os.path.join(root, name))

out_f = open(out_fp, 'w')
for f in out_files:
    out_f.write("%s\n" % f)
out_f.close()
