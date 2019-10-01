import os

import cv2
import numpy as np
import torch
from torch.nn import DataParallel

from attack_dataset import AttackDataset
from backbone import mobilefacenet, cbam, attention
from util import get_log_filename

device = 'cuda'


def loadModel(args, idx):
    if args.backbone_net[idx] == 'MobileFace':
        net = mobilefacenet.MobileFaceNet()
    elif args.backbone_net[idx] == 'CBAM_50':
        net = cbam.CBAMResNet(50, feature_dim=args.feature_dim[idx], mode='ir')
    elif args.backbone_net[idx] == 'CBAM_50_SE':
        net = cbam.CBAMResNet(50, feature_dim=args.feature_dim[idx], mode='ir_se')
    elif args.backbone_net[idx] == 'CBAM_100':
        net = cbam.CBAMResNet(100, feature_dim=args.feature_dim[idx], mode='ir')
    elif args.backbone_net[idx] == 'CBAM_100_SE':
        net = cbam.CBAMResNet(100, feature_dim=args.feature_dim[idx], mode='ir_se')
    elif args.backbone_net[idx] == 'CBAM_152':
        net = cbam.CBAMResNet(152, feature_dim=args.feature_dim[idx], mode='ir')
    elif args.backbone_net[idx] == 'CBAM_152_SE':
        net = cbam.CBAMResNet(152, feature_dim=args.feature_dim[idx], mode='ir_se')
    elif args.backbone_net[idx] == 'Attention_56':
        net = attention.ResidualAttentionNet_56(feature_dim=args.feature_dim[idx])
    else:
        net = None
        print(args.backbone_net[idx], ' is not available!')
        assert 1 == 0

    # gpu init
    multi_gpus = False
    # if len(args.gpus.split(',')) > 1:
    #     multi_gpus = True
    # os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    net.load_state_dict(torch.load(args.resume[idx])['net_state_dict'])

    if multi_gpus:
        net = DataParallel(net).to(device)
    else:
        net = net.to(device)

    return net.eval()


def get_dist(args, load_attack_masks=False):
    attack_dataset = AttackDataset(args.root, args.dev_path, args.features_path[0], args.flip_features_path[0], True)
    if load_attack_masks:
        AttackDataset.init_attack_masks(args.masks_size, args.pt_x, args.pt_y)
        log_filename = get_log_filename(args)
        path = "{}/{}.pth".format(args.masks_dir, log_filename)
        attack_dataset.load_attack_masks(path)

    dists = []
    with torch.no_grad():
        for i, (img, img_after_attack) in enumerate(attack_dataset):
            dist = torch.mean(torch.sqrt(torch.sum(torch.pow(img_after_attack - img, 2), dim=(0,))))
            dists.append(dist.cpu().numpy())

    return np.mean(dists)


def get_dist_from_images(root_dir, output_dir):
    dists = []
    for filename in sorted(os.listdir(root_dir)):
        img = cv2.imread("{}/{}".format(root_dir, filename)).astype(np.float32)
        img_a = cv2.imread("{}/{}".format(output_dir, filename)).astype(np.float32)

        dist = np.mean(np.sqrt(np.sum(np.power((img_a - img), 2), axis=-1)))
        dists.append(dist)

    return np.mean(dists)
