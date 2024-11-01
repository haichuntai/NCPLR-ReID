# -*- coding: utf-8 -*-
from __future__ import print_function, absolute_import
import argparse
import os.path as osp
import random
import numpy as np
import sys
import time
from datetime import timedelta

import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader

from clustercontrast import datasets
from clustercontrast import models
from clustercontrast.evaluators import Evaluator
from clustercontrast.utils.data import transforms as T
from clustercontrast.utils.data.preprocessor import Preprocessor
from clustercontrast.utils.logging import Logger
from clustercontrast.utils.serialization import load_checkpoint

start_epoch = best_mAP = 0

import datetime


def get_data(name, data_dir):
    root = osp.join(data_dir, name)
    dataset = datasets.create(name, root)
    return dataset


def get_test_loader(dataset, height, width, batch_size, workers, testset=None):
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    test_transformer = T.Compose([
        T.Resize((height, width), interpolation=3),
        T.ToTensor(),
        normalizer
    ])

    if testset is None:
        testset = list(set(dataset.query) | set(dataset.gallery))

    test_loader = DataLoader(
        Preprocessor(testset, root=dataset.images_dir, transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)

    return test_loader


def create_model(args):
    model = models.create(args.arch, num_features=args.features, norm=True, dropout=args.dropout,
                          num_classes=3000, pooling_type=args.pooling_type, use_part=args.use_part)
    # use CUDA
    model.cuda()
    model = nn.DataParallel(model)

    return model


def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True

    main_worker(args)


def main_worker(args):
    global start_epoch, best_mAP
    start_time = time.monotonic()

    cudnn.benchmark = True

    # sys.stdout = Logger(osp.join(args.logs_dir, 'log.txt'))
    sys.stdout = Logger(osp.join(args.logs_dir, 'log_{}.txt'.format(datetime.datetime.now().strftime('%Y-%m-%d|%H:%M:%S'))))
    # sys.stderr = Logger(osp.join(args.logs_dir, 'log_error_{}.txt'.format(datetime.datetime.now().strftime('%Y-%m-%d|%H:%M:%S'))))
    print("==========\nArgs:{}\n==========".format(args))

    # Create datasets
    iters = args.iters if (args.iters > 0) else None
    print("==> Load unlabeled dataset")
    dataset = get_data(args.dataset, args.data_dir)
    test_loader = get_test_loader(dataset, args.height, args.width, args.batch_size, args.workers)

    # Create model
    model = create_model(args)

    # Evaluator
    evaluator = Evaluator(model)


    # Load from checkpoint
    if args.resume:
        # checkpoint = load_checkpoint(osp.join(args.resume, 'model_best.pth.tar'))
        checkpoint = load_checkpoint(osp.join(args.resume, 'model_best.pth.tar'))
        model.load_state_dict(checkpoint['state_dict'],False)

    print('==> Test with the best model:')
    evaluator.evaluate(test_loader, dataset.query, dataset.gallery, cmc_flag=True, use_part=args.use_part)

    end_time = time.monotonic()
    print('Total running time: ', timedelta(seconds=end_time - start_time))



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Self-paced contrastive learning on unsupervised re-ID")
    # data
    parser.add_argument('-d', '--dataset', type=str, default='market1501',
                        choices=datasets.names())
    parser.add_argument('-b', '--batch-size', type=int, default=256)
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('--height', type=int, default=320, help="input height")
    parser.add_argument('--width', type=int, default=128, help="input width")
    parser.add_argument('--num-instances', type=int, default=16,
                        help="each minibatch consist of "
                             "(batch_size // num_instances) identities, and "
                             "each identity has num_instances instances, "
                             "default: 0 (NOT USE)")
    # cluster
    parser.add_argument('--eps', type=float, default=0.4,
                        help="max neighbor distance for DBSCAN")
    parser.add_argument('--eps-neighbor', type=float, default=0.2)
    parser.add_argument('--eps-partneighbor', type=float, default=0.2)
    parser.add_argument('--eps-negneighbor', type=float, default=0.7)
    parser.add_argument('--eps-neighbor-add', type=float, default=0.0)
    parser.add_argument('--eps-gap', type=float, default=0.02,
                        help="multi-scale criterion for measuring cluster reliability")
    parser.add_argument('--k1', type=int, default=30,
                        help="hyperparameter for jaccard distance")
    parser.add_argument('--k2', type=int, default=6,
                        help="hyperparameter for jaccard distance")

    # model
    parser.add_argument('-a', '--arch', type=str, default='resnet50_neighbor',
                        choices=models.names())
    parser.add_argument('--features', type=int, default=0)
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--momentum', type=float, default=0.1,
                        help="update momentum for the hybrid memory")
    # optimizer
    parser.add_argument('--lr', type=float, default=0.00035,
                        help="learning rate")
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--iters', type=int, default=200)
    parser.add_argument('--step-size', type=int, default=20)
    # training configs
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--print-freq', type=int, default=10)
    parser.add_argument('--eval-step', type=int, default=1)
    parser.add_argument('--temp', type=float, default=0.05,
                        help="temperature for scaling contrastive loss")
    # path
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, '../data'))
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'logs'))
    parser.add_argument('--pooling-type', type=str, default='gem')
    parser.add_argument('--use-hard', action="store_true")
    parser.add_argument('--use-mean', action="store_true")
    parser.add_argument('--use-rand', action="store_true")
    parser.add_argument('--no-cam',  action="store_true")

    parser.add_argument('--store-checkpoint',  action="store_true")
    parser.add_argument('--store-checkpoint-eval',  action="store_true")
    parser.add_argument('--ins-posoption', type=int, default=0)
    parser.add_argument('--ins-negoption', type=int, default=0)
    parser.add_argument('--ce-option', type=int, default=0)
    parser.add_argument('--cr-option', type=int, default=0)
    parser.add_argument('--extra-option', type=int, default=0)
    parser.add_argument('--extra-option-sub', type=int, default=0)
    parser.add_argument('--temp-dist', type=float, default=1.0)
    parser.add_argument('--loss-weight', type=float, default=1.0)
    parser.add_argument('--ins-weight',  action="store_true")
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--cmalpha', type=float, default=0.1)
    parser.add_argument('--topk-s', type=int, default=0)
    parser.add_argument('--topk-spart', type=int, default=0)
    parser.add_argument('--resume', type=str, default='', metavar='PATH')
    parser.add_argument('--resume-epoch', type=int, default=0)
    parser.add_argument('--ins-epoch', type=int, default=20)
    parser.add_argument('--dbi-value', type=float, default=1.3)
    parser.add_argument('--lambda2', type=float, default=0.2)
    parser.add_argument('--lambda1', type=float, default=0.2)
    parser.add_argument('--temp-KLlogits', type=float, default=1.0)
    parser.add_argument('--temp-KLloss', type=float, default=1.0)
    parser.add_argument('--refineloss-weight', type=float, default=1.0)
    
    parser.add_argument('--use-meanteacher',  action="store_true")
    parser.add_argument('--use-auxmemory',  action="store_true")
    parser.add_argument('--use-hardaux', action="store_true")
    parser.add_argument('--use-meanaux', action="store_true")
    parser.add_argument('--use-randaux', action="store_true")
    parser.add_argument('--use-insmemory',  action="store_true")
    parser.add_argument('--use-refinelabels',  action="store_true")
    parser.add_argument('--use-truelabel',  action="store_true")
    parser.add_argument('--use-part',  action="store_true")
    parser.add_argument('--num-parts', type=int, default=2, help="number of part")
    parser.add_argument('--use-mixdistneighbbor',  action="store_true")

    main()
