import logging
import os
from pickle import dump, load

import numpy as np


class AverageMeter(object):

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0


def get_log_filename(args):
    filename = "{h}x{w}, alpha={alpha}, {n_models}models, {epochs}epochs, lr={lr:.1e}, batch_size={bs}".format(
        alpha=args.alpha, n_models=len(args.backbone_net), epochs=args.epochs, lr=args.lr,
        bs=args.batch_size, h=args.masks_size[-2], w=args.masks_size[-1]
    )
    return filename


def init_log(output_dir, filename):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        filename=os.path.join(output_dir, filename),
                        filemode='w')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger('').addHandler(console)
    return logging


def init_random_state(args):
    if args.random_state != "":
        with open(args.random_state, 'rb') as f:
            random_state = load(f)
            np.random.set_state(random_state)
    else:
        with open("state/random_state.obj", 'wb') as f:
            random_state = np.random.get_state()
            dump(random_state, f)
