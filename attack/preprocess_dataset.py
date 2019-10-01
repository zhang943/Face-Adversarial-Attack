import os

import cv2
import numpy as np
import torch
import torch.utils.data as data


def img_loader(path):
    try:
        with open(path, 'rb') as f:
            img = cv2.imread(path)
            if len(img.shape) == 2:
                img = np.stack([img] * 3, 2)
            return img
    except IOError:
        print('Cannot load image ' + path)


class LFW(data.Dataset):
    def __init__(self, root, dev_path, transform=None, loader=img_loader, flip=False):
        self.LFW_root = "/data/FaceRecognition/LFW/lfw_align_112"
        self.root = root
        self.transform = transform
        self.loader = loader

        self.dev = np.loadtxt(dev_path, dtype=str, delimiter=',', skiprows=1)

        self.flip = flip

    def __getitem__(self, index):
        person = self.dev[index, 2]
        image_name = self.dev[index, 1]

        imglist = []
        for img in os.listdir("{}/{}".format(self.LFW_root, person)):
            imglist.append(os.path.join(self.LFW_root, person, img))
        imglist.sort()
        imglist.insert(0, os.path.join(self.root, image_name))

        imgs = []
        if self.transform is not None:
            for img_path in imglist:
                img = self.transform(self.loader(img_path))
                if self.flip:
                    img = torch.flip(img, dims=(2,))
                imgs.append(img)
        else:
            for img_path in imglist:
                imgs.append(torch.from_numpy(self.loader(img_path)))

        return imgs

    def __len__(self):
        return self.dev.shape[0]
