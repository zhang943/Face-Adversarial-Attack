import os
import sys

root = "/data/FaceRecognition/WebFace/webface_align_112"

dirs = os.listdir(root)
dirs.sort()

n = 0

with open("{}/{}".format("/data/FaceRecognition/WebFace", "align_train.list"), 'w') as f:
    for i, d in enumerate(dirs):
        imgs = os.listdir("{}/{}".format(root, d))
        imgs.sort()
        for img in imgs:
            f.write("{}/{} {}\n".format(d, img, i))

