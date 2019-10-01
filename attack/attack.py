import argparse
import os
import shutil
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader

from attack_dataset import AttackDataset
from functions import loadModel, get_dist, get_dist_from_images
from util import AverageMeter, init_log, get_log_filename, init_random_state

device = 'cuda'


def argparser():
    parser = argparse.ArgumentParser(description='securityAI')
    parser.add_argument('--root', type=str, default='/data/FaceRecognition/securityAI/securityAI_round1_images')
    parser.add_argument('--dev_path', type=str, default='/data/FaceRecognition/securityAI/securityAI_round1_dev.csv')

    parser.add_argument('--backbone_net', type=list,
                        default=['CBAM_50_SE', 'MobileFace', "CBAM_50", 'CBAM_100_SE', 'CBAM_100',
                                 'CBAM_152', "Attention_56", ])

    parser.add_argument('--feature_dim', type=list,
                        default=[512, 128, 512, 512, 512, 512, 512])

    parser.add_argument('--resume', type=list,
                        default=['../model/SERes50_IR_SERES50_IR_20190819_165550/Iter_120000_net.ckpt',
                                 '../model/Mobile_MOBILEFACE_20190813_112144/Iter_054000_net.ckpt',
                                 '../model/Res50_IR_RES50_IR_20190821_181502/Iter_108000_net.ckpt',
                                 '../model/SERes100_IR_SERES100_IR_20190820_161900/Iter_078000_net.ckpt',
                                 '../model/Res100_IR_RES100_IR_20190824_180052/Iter_111000_net.ckpt',
                                 '../model/Res152_IR_RES152_IR_20190904_094041/Iter_099000_net.ckpt',
                                 '../model/Attention_56_ATTENTION_56_20190822_164221/Iter_093000_net.ckpt', ])

    parser.add_argument('--features_path', type=list,
                        default=['../result/features_attacked_SERes50.mat',
                                 '../result/features_attacked_Mobile.mat',
                                 '../result/features_attacked_Res50.mat',
                                 '../result/features_attacked_SERes100.mat',
                                 '../result/features_attacked_Res100.mat',
                                 '../result/features_attacked_Res152.mat',
                                 '../result/features_attacked_Attention56.mat', ])

    parser.add_argument('--flip_features_path', type=list,
                        default=['../result/flip_features_attacked_SERes50.mat',
                                 '../result/flip_features_attacked_Mobile.mat',
                                 '../result/flip_features_attacked_Res50.mat',
                                 '../result/flip_features_attacked_SERes100.mat',
                                 '../result/flip_features_attacked_Res100.mat',
                                 '../result/flip_features_attacked_Res152.mat',
                                 '../result/flip_features_attacked_Attention56.mat', ])

    parser.add_argument('--masks_size', type=list, default=[712, 3, 64, 64])
    parser.add_argument('--pt_x', type=int, default=22)
    parser.add_argument('--pt_y', type=int, default=36)

    parser.add_argument('--batch_size', type=int, default=32, help='batch_size')
    parser.add_argument('--epochs', type=int, default=17 * 2, help='epochs')
    parser.add_argument('--lr', type=float, default=1000000.0 / 2, help='learning_rate')
    parser.add_argument('--wd', type=float, default=0.0, help='weight_decay')
    parser.add_argument('--alpha', type=float, default=0.2, help='weight_decay')

    parser.add_argument('--output_path', type=str, default='/data/FaceRecognition/securityAI/images')
    parser.add_argument('--log_dir', type=str, default='log/')
    parser.add_argument('--masks_dir', type=str, default='masks/')
    parser.add_argument('--random_state', type=str, default='state/(3.4962) random_state.obj')

    args = parser.parse_args()
    return args


def eval_after_attack(net, attack_loader):
    all_features = []
    with torch.no_grad():
        for imgs_t, _, _, _ in attack_loader:
            all_features.append(net(imgs_t).cpu().numpy())

        all_features = np.concatenate(all_features, axis=0)
        acc, cos_dist = attack_loader.dataset.eval(all_features)

        return acc, cos_dist


def attack(net, attack_loader, optimizer):
    loss_meter = AverageMeter()
    cosine_similarity = nn.CosineSimilarity(dim=1, eps=1e-6).to(device)

    for imgs_t, features, imgs, imgs_after_attack in attack_loader:
        features = features.to(device)
        outputs = net(imgs_t)

        similarity = cosine_similarity(outputs, features)
        loss = torch.mean(similarity)
        loss_meter.update(loss.item(), imgs_t.shape[0])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return loss_meter.avg


def attack_with_dist_constraint(net, attack_loader, optimizer, alpha=0.2):
    loss_meter = AverageMeter()
    cos_criterion = nn.CosineEmbeddingLoss(margin=-1.0).to(device)
    dist_criterion = nn.L1Loss().to(device)

    for imgs_t, features, imgs, imgs_after_attack in attack_loader:
        features = features.to(device)
        outputs = net(imgs_t)

        cos_loss = cos_criterion(outputs, features, torch.tensor([-1.0]).to(device))
        dist_loss = dist_criterion(imgs, imgs_after_attack)
        loss = torch.add(cos_loss, dist_loss, alpha=alpha)
        loss_meter.update(cos_loss.item() + cos_criterion.margin, imgs_t.shape[0])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return loss_meter.avg


def main():
    args = argparser()

    log_filename = get_log_filename(args)
    logger = init_log(args.log_dir, "{}.log".format(log_filename))

    init_random_state(args)

    n_models = len(args.backbone_net)

    nets = []
    attack_loaders = []
    for idx in range(n_models):
        net = loadModel(args, idx)
        nets.append(net)
        attack_dataset = AttackDataset(args.root, args.dev_path, args.features_path[idx], args.flip_features_path[idx])
        attack_loaders.append(DataLoader(attack_dataset, batch_size=args.batch_size, shuffle=False))

    AttackDataset.init_attack_masks(args.masks_size, args.pt_x, args.pt_y)

    the_last_batch_size = args.masks_size[0] % args.batch_size
    the_last_batch_lr = args.lr * (the_last_batch_size / args.batch_size)
    optimizer = optim.SGD([
        {'params': AttackDataset.attack_masks[:-the_last_batch_size]},
        {'params': AttackDataset.attack_masks[-the_last_batch_size:], 'lr': the_last_batch_lr}],
        lr=args.lr, weight_decay=args.wd
    )

    for epoch in range(args.epochs):
        for idx in range(n_models):
            cos_similarity = attack_with_dist_constraint(nets[idx], attack_loaders[idx], optimizer, args.alpha)
            acc, _ = eval_after_attack(nets[idx], attack_loaders[idx])

            l2_dist = get_dist(args)

            logger.info("Model{}, Epoch {:02d}, Acc: {:.4f}, Cos_Similarity: {:6.4f}, L2_dist: {:6.4f}".format(
                idx, epoch, acc, cos_similarity, l2_dist))

            torch.save(AttackDataset.attack_masks, "{}/{}.pth".format(args.masks_dir, log_filename))


def generate_imgs(load_attack_masks=False):
    args = argparser()

    attack_dataset = AttackDataset(args.root, args.dev_path, args.features_path[0], args.flip_features_path[0], True)

    if load_attack_masks:
        attack_dataset.init_attack_masks(args.masks_size, args.pt_x, args.pt_y)
        attack_dataset.load_attack_masks("{}/{}.pth".format(args.masks_dir, get_log_filename(args)))

    print("L2: {}".format(get_dist(args)))

    if os.path.exists(args.output_path):
        shutil.rmtree(args.output_path)
    os.mkdir(args.output_path)

    with torch.no_grad():
        for i, (img, img_after_attack) in enumerate(attack_dataset):
            img_after_attack = np.transpose(img_after_attack.cpu().numpy(), [1, 2, 0])
            cv2.imwrite("{}/{:05}.png".format(args.output_path, i + 1), img_after_attack)
            shutil.move("{}/{:05}.png".format(args.output_path, i + 1), "{}/{:05}.jpg".format(args.output_path, i + 1))

    print("L2: {}".format(get_dist_from_images(args.root, args.output_path)))


if __name__ == '__main__':
    main()
    generate_imgs(load_attack_masks=True)
