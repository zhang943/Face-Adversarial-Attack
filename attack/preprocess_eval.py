import argparse
import os

import numpy as np
import scipy.io
import torch
import torchvision.transforms as transforms
from torch.nn import DataParallel
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from backbone import mobilefacenet, cbam, attention
from preprocess_dataset import LFW


def loadModel(backbone_net, feature_dim, gpus, resume, root, dev_path, flip):
    if backbone_net == 'MobileFace':
        net = mobilefacenet.MobileFaceNet()
    elif backbone_net == 'CBAM_50':
        net = cbam.CBAMResNet(50, feature_dim=feature_dim, mode='ir')
    elif backbone_net == 'CBAM_50_SE':
        net = cbam.CBAMResNet(50, feature_dim=feature_dim, mode='ir_se')
    elif backbone_net == 'CBAM_100':
        net = cbam.CBAMResNet(100, feature_dim=feature_dim, mode='ir')
    elif backbone_net == 'CBAM_100_SE':
        net = cbam.CBAMResNet(100, feature_dim=feature_dim, mode='ir_se')
    elif backbone_net == 'CBAM_152':
        net = cbam.CBAMResNet(152, feature_dim=feature_dim, mode='ir')
    elif backbone_net == 'CBAM_152_SE':
        net = cbam.CBAMResNet(152, feature_dim=feature_dim, mode='ir_se')
    elif backbone_net == 'Attention_56':
        net = attention.ResidualAttentionNet_56(feature_dim=feature_dim)
    else:
        net = None
        print(backbone_net, ' is not available!')
        assert 1 == 0

    # gpu init
    multi_gpus = False
    if len(gpus.split(',')) > 1:
        multi_gpus = True
    os.environ['CUDA_VISIBLE_DEVICES'] = gpus
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    net.load_state_dict(torch.load(resume)['net_state_dict'])

    if multi_gpus:
        net = DataParallel(net).to(device)
    else:
        net = net.to(device)

    transform = transforms.Compose([
        transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # range [0.0, 1.0] -> [-1.0,1.0]
    ])
    lfw_dataset = LFW(root, dev_path, transform=transform, flip=flip)
    lfw_loader = DataLoader(lfw_dataset, batch_size=1, shuffle=False)

    return net.eval(), device, lfw_dataset, lfw_loader


def getFeatureFromTorch(net, device, data_loader, feature_save_path, is_hard, is_flip, start_idx=1):
    features_query = []
    features_avg = []
    with torch.no_grad():
        for imgs in tqdm(data_loader):
            feature_query = net(imgs[0].to(device)).cpu().numpy()
            feature_avg = [net(img.to(device)).cpu().numpy() for img in imgs[start_idx:]]
            feature_avg = np.concatenate(feature_avg, axis=0)
            feature_avg = np.mean(feature_avg, axis=0, keepdims=True)

            features_query.append(feature_query)
            features_avg.append(feature_avg)

    features_query = np.concatenate(features_query, axis=0)
    features_avg = np.concatenate(features_avg, axis=0)

    result = {'features_query': features_query, 'features_avg': features_avg}

    tokens = list(os.path.split(feature_save_path))
    if is_flip:
        tokens[-1] = "flip_" + tokens[-1]
    if is_hard:
        tokens.insert(1, "hard/")

    save_path = os.path.join(*tokens)
    scipy.io.savemat(save_path, result)

    return save_path


def main():
    args = argparser()
    n_models = len(args.backbone_net)

    for idx in range(n_models):
        for is_flip in [False, True]:
            net, device, lfw_dataset, lfw_loader = loadModel(args.backbone_net[idx], args.feature_dim[idx], args.gpus,
                                                             args.resume[idx], args.root, args.dev_path, is_flip)

            is_hard = "hard" in args.dev_path
            save_path = getFeatureFromTorch(net, device, lfw_loader, args.feature_save_path[idx], is_hard, is_flip)

            acc = predict(save_path)

            print(args.backbone_net[idx], acc)


def predict(feature_save_path):
    np.set_printoptions(precision=3, linewidth=1000)
    features_path = feature_save_path
    features = scipy.io.loadmat(features_path)
    features_query = features['features_query']
    features_avg = features['features_avg']

    num_features = features_query.shape[0]

    norm_query = np.linalg.norm(features_query, axis=1, keepdims=True)
    norm_avg = np.linalg.norm(features_avg, axis=1, keepdims=True)

    cos_dist = np.matmul(features_query, features_avg.T) / np.matmul(norm_query, norm_avg.T)

    preds = np.argmax(cos_dist, axis=1)

    acc = np.sum(preds == np.arange(0, num_features)) / num_features

    return acc


def argparser():
    parser = argparse.ArgumentParser(description='Testing LFW')
    parser.add_argument('--root', type=str, default='/data/FaceRecognition/securityAI/securityAI_round1_images',
                        help='The path of lfw data')
    parser.add_argument('--dev_path', type=str, default='/data/FaceRecognition/securityAI/securityAI_round1_dev.csv',
                        help='The path of lfw data')
    parser.add_argument('--backbone_net', type=list,
                        default=['MobileFace', 'CBAM_50', 'CBAM_50_SE', 'CBAM_100_SE', 'Attention_56', 'CBAM_100',
                                 'CBAM_152'],
                        help='MobileFace, CBAM_50, CBAM_50_SE, CBAM_100, CBAM_100_SE, Attention_56')
    parser.add_argument('--feature_dim', type=list,
                        default=[128, 512, 512, 512, 512, 512, 512],
                        help='feature dimension')
    parser.add_argument('--resume', type=list,
                        default=['../model/Mobile_MOBILEFACE_20190813_112144/Iter_054000_net.ckpt',
                                 '../model/Res50_IR_RES50_IR_20190821_181502/Iter_108000_net.ckpt',
                                 '../model/SERes50_IR_SERES50_IR_20190819_165550/Iter_120000_net.ckpt',
                                 '../model/SERes100_IR_SERES100_IR_20190820_161900/Iter_078000_net.ckpt',
                                 '../model/Attention_56_ATTENTION_56_20190822_164221/Iter_093000_net.ckpt',
                                 '../model/Res100_IR_RES100_IR_20190824_180052/Iter_111000_net.ckpt',
                                 '../model/Res152_IR_RES152_IR_20190904_094041/Iter_099000_net.ckpt', ],
                        help='The path pf save model')
    parser.add_argument('--feature_save_path', type=list,
                        default=['../result/features_attacked_Mobile.mat',
                                 '../result/features_attacked_Res50.mat',
                                 '../result/features_attacked_SERes50.mat',
                                 '../result/features_attacked_SERes100.mat',
                                 '../result/features_attacked_Attention56.mat',
                                 '../result/features_attacked_Res100.mat',
                                 '../result/features_attacked_Res152.mat', ],
                        help='The path of the extract features save, must be .mat file')
    parser.add_argument('--gpus', type=str, default='0', help='gpu list')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main()
