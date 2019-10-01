import cv2
import numpy as np
import scipy.io
import torch
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset

device = 'cuda'


class AttackDataset(Dataset):
    attack_masks = None
    x, y, w, h = None, None, None, None

    def __init__(self, root, dev_path, features_path, flip_features_path, test=False):
        super(AttackDataset, self).__init__()

        self.root = root

        self.dev = np.loadtxt(dev_path, dtype=str, delimiter=',', skiprows=1)
        self.to_be_attacked = self.dev[:, 1]

        features = scipy.io.loadmat(features_path)
        self.features_query = features['features_query']
        self.features_avg = features['features_avg']

        flip_features = scipy.io.loadmat(flip_features_path)
        self.flip_features_query = flip_features['features_query']
        self.flip_features_avg = flip_features['features_avg']

        self.num_features = self.features_query.shape[0]

        self.transform = transforms.Compose([
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])

        self.test = test

    def __getitem__(self, index):
        img = self.img_loader("{}/{}".format(self.root, self.to_be_attacked[index]))
        img = np.transpose(img, axes=[2, 0, 1])
        img = torch.from_numpy(img).float().to(device)

        attack_mask = torch.clamp(self.attack_masks[index], -25.5, 25.5)
        img_after_attack = img.clone()
        img_after_attack[:, self.y:self.y + self.h, self.x:self.x + self.w] = \
            img_after_attack[:, self.y:self.y + self.h, self.x:self.x + self.w] + attack_mask
        img_after_attack = torch.clamp(img_after_attack, 0, 255)

        if self.test:
            return img, img_after_attack

        is_flip = np.random.random() < 0.5

        img_t = self.transform(img_after_attack / 255)
        img_t = self.augment(img_t, is_flip)

        if is_flip:
            feature = self.flip_features_avg[index]
        else:
            feature = self.features_avg[index]

        return img_t, feature, img, img_after_attack

    def __len__(self):
        return len(self.to_be_attacked)

    def eval(self, features_after_attack=None):
        features_query = self.features_query
        if features_after_attack is not None:
            assert features_after_attack.shape == self.features_query.shape
            features_query = features_after_attack

        norm_query = np.linalg.norm(features_query, axis=1, keepdims=True)
        norm_avg = np.linalg.norm(self.features_avg, axis=1, keepdims=True)

        cos_dist = np.matmul(features_query, self.features_avg.T) / np.matmul(norm_query, norm_avg.T)
        preds = np.argmax(cos_dist, axis=1)

        acc = np.sum(preds == np.arange(0, self.num_features)) / self.num_features
        return acc, cos_dist

    @staticmethod
    def augment(img, flag_flip):
        if flag_flip:
            img = torch.flip(img, (2,))
        return img

    @staticmethod
    def img_loader(path):
        try:
            img = cv2.imread(path)
            if len(img.shape) == 2:
                img = np.stack([img] * 3, 2)
            return img
        except IOError:
            print('Cannot load image ' + path)

    @staticmethod
    def init_attack_masks(masks_size, x, y):
        if AttackDataset.attack_masks is not None:
            return

        AttackDataset.x = x
        AttackDataset.y = y
        AttackDataset.w = masks_size[-1]
        AttackDataset.h = masks_size[-2]

        assert AttackDataset.x + AttackDataset.w <= 112
        assert AttackDataset.y + AttackDataset.h <= 112

        AttackDataset.attack_masks = []
        for i in range(masks_size[0]):
            AttackDataset.attack_masks.append(torch.zeros(size=masks_size[1:], device=device, requires_grad=True))

    @staticmethod
    def load_attack_masks(path):
        AttackDataset.attack_masks = torch.load(path)
