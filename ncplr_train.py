# -*- coding: utf-8 -*-
from __future__ import print_function, absolute_import
import argparse
import os.path as osp
import random
import numpy as np
import sys
import collections
import time
from datetime import timedelta

from sklearn.cluster import DBSCAN

import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader
import torch.nn.functional as F

from ncplr import datasets
from ncplr import models
from ncplr.models.cm import ClusterMemory
from ncplr.trainers import ncplrTrainer
from ncplr.evaluators import Evaluator, extract_features
from ncplr.utils.data import IterLoader
from ncplr.utils.data import transforms as T
from ncplr.utils.data.preprocessor import Preprocessor
from ncplr.utils.logging import Logger
from ncplr.utils.serialization import load_checkpoint, save_checkpoint
from ncplr.utils.faiss_rerank import compute_jaccard_distance
from ncplr.utils.data.sampler import RandomMultipleGallerySampler, RandomMultipleGallerySamplerNoCam

start_epoch = best_mAP = 0

def get_data(name, data_dir):
    root = osp.join(data_dir, name)
    dataset = datasets.create(name, root)
    return dataset

def get_train_loader(args, dataset, height, width, batch_size, workers,
                     num_instances, iters, trainset=None, cam=False):

    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    train_transformer = T.Compose([
        T.Resize((height, width), interpolation=3),
        T.RandomHorizontalFlip(p=0.5),
        T.Pad(10),
        T.RandomCrop((height, width)),
        T.ToTensor(),
        normalizer,
        T.RandomErasing(probability=0.5, mean=[0.485, 0.456, 0.406])
    ])

    train_set = sorted(dataset.train) if trainset is None else sorted(trainset)
    rmgs_flag = num_instances > 0
    if rmgs_flag:
        if cam:
            sampler = RandomMultipleGallerySampler(train_set, num_instances)
        else:
            sampler = RandomMultipleGallerySamplerNoCam(train_set, num_instances)
    else:
        sampler = None
    train_loader = IterLoader(
        DataLoader(Preprocessor(train_set, root=dataset.images_dir, transform=train_transformer, mutual=True),
                   batch_size=batch_size, num_workers=workers, sampler=sampler,
                   shuffle=not rmgs_flag, pin_memory=True, drop_last=True), length=iters)

    return train_loader


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
                          num_classes=3000, pooling_type=args.pooling_type)
    model_ema = models.create(args.arch, num_features=args.features, norm=True, dropout=args.dropout,
                          num_classes=3000, pooling_type=args.pooling_type)
    # use CUDA
    model.cuda()
    model = nn.DataParallel(model)
    model_ema.cuda()
    model_ema = nn.DataParallel(model_ema)
    return model, model_ema


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

    sys.stdout = Logger(osp.join(args.logs_dir, 'log.txt'))
    print("==========\nArgs:{}\n==========".format(args))

    # Create datasets
    iters = args.iters if (args.iters > 0) else None
    print("==> Load unlabeled dataset")
    dataset = get_data(args.dataset, args.data_dir)
    test_loader = get_test_loader(dataset, args.height, args.width, args.batch_size, args.workers)

    # Create model
    model, model_ema = create_model(args)

    # Evaluator
    evaluator = Evaluator(model)
    evaluator_ema = Evaluator(model_ema)

    # Load from checkpoint
    if args.resume:
        checkpoint = load_checkpoint(osp.join(args.resume, 'model_best.pth.tar'))
        model.load_state_dict(checkpoint['state_dict'])
        model_ema.load_state_dict(checkpoint['state_dict'])
        evaluator.evaluate(test_loader, dataset.query, dataset.gallery, cmc_flag=True)

    # Optimizer
    params = [{"params": [value]} for _, value in model.named_parameters() if value.requires_grad]
    optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.1)

    # Trainer
    trainer = ncplrTrainer(model, model_ema, ema_decay=args.ema_decay)

    for epoch in range(args.epochs):
        with torch.no_grad():
            print('==> Create pseudo labels for unlabeled data')
            cluster_loader = get_test_loader(dataset, args.height, args.width,
                                             args.batch_size, args.workers, testset=sorted(dataset.train))

            features, _, _ = extract_features(model, cluster_loader, print_freq=50)
            features = torch.cat([features[f].unsqueeze(0) for f, _, _ in sorted(dataset.train)], 0)
            features_ema, _, _ = extract_features(model_ema, cluster_loader, print_freq=50)
            features_ema = torch.cat([features_ema[f].unsqueeze(0) for f, _, _ in sorted(dataset.train)], 0)
            rerank_dist = compute_jaccard_distance(features, k1=args.k1, k2=args.k2)

            if epoch == 0:
                # DBSCAN cluster
                eps = args.eps
                print('Clustering criterion: eps: {:.3f}'.format(eps))
                cluster = DBSCAN(eps=eps, min_samples=4, metric='precomputed', n_jobs=-1)

            # select & cluster images as training set of this epochs
            pseudo_labels = cluster.fit_predict(rerank_dist)
            
            num_cluster = len(set(pseudo_labels)) - (1 if -1 in pseudo_labels else 0)
            trainer.num_cluster = num_cluster

            idxs = []
            for i, pl in enumerate(pseudo_labels):
                if pl.item() != -1:
                    idxs.append(i)

            # Search neighbors
            neighbors = []
            neighbor_dists = []
            for dist in rerank_dist[idxs][:, idxs]:
                neighbor = np.argwhere(dist < args.eps_neighbor).flatten().tolist()
                neighbor_dists.append(dist[neighbor].tolist())
                neighbors.append(neighbor)
            trainer.neighbors = neighbors
            trainer.neighbor_dists = neighbor_dists

        # Generate new dataset and calculate cluster centers
        @torch.no_grad()
        def generate_cluster_features(labels, features):
            centers = collections.defaultdict(list)
            un_used = 0
            for i, label in enumerate(labels):
                if label == -1:
                    un_used += 1
                    continue
                centers[labels[i]].append(features[i])
            print('number of unused instances: {}'.format(un_used))
            centers = [ torch.stack(centers[idx], dim=0).mean(0) for idx in sorted(centers.keys()) ]
            centers = torch.stack(centers, dim=0)
            return centers

        cluster_features = generate_cluster_features(pseudo_labels, features)
        cluster_features_ema = generate_cluster_features(pseudo_labels, features_ema)
        del cluster_loader, features, features_ema

        # Initial classifier
        centroids_g = F.normalize(cluster_features, p=2, dim=1)
        centroids_g_ema = F.normalize(cluster_features_ema, p=2, dim=1)
        model.module.classifier.weight.data[:num_cluster].copy_(centroids_g)
        model_ema.module.classifier.weight.data[:num_cluster].copy_(centroids_g_ema)

        # Create hybrid memory
        memory = ClusterMemory(num_cluster, temp=args.temp, momentum=args.momentum, use_hard=args.use_hard, alpha=args.alpha, 
                               temp_dist=args.temp_dist, lambda1=args.lambda1, lambda2=args.lambda2, eps_consistency=args.eps_neighbor-args.eps_neighbor_gap).cuda()
        memory.features = F.normalize(cluster_features, dim=1).cuda()

        trainer.memory = memory

        pseudo_labeled_dataset = []
        for i, ((fname, _, cid), label) in enumerate(zip(sorted(dataset.train), pseudo_labels)):
            if label != -1:
                pseudo_labeled_dataset.append((fname, label.item(), cid))
        print('==> Statistics for epoch {}: {} clusters'.format(epoch, num_cluster))

        train_loader = get_train_loader(args, dataset, args.height, args.width,
                                        args.batch_size, args.workers, args.num_instances, iters,
                                        trainset=pseudo_labeled_dataset, cam=args.cam)

        train_loader.new_epoch()
        trainer.train(epoch, train_loader, optimizer, print_freq=args.print_freq, 
                      train_iters=len(train_loader), neighbors=neighbors, p=args.p, rampup_max_value=args.rampup_value)
        
        if (epoch + 1) % args.eval_step == 0 or (epoch == args.epochs - 1):
            mAP = evaluator.evaluate(test_loader, dataset.query, dataset.gallery, cmc_flag=True)[1]
            mAP_ema = evaluator_ema.evaluate(test_loader, dataset.query, dataset.gallery, cmc_flag=True)[1]
            is_best = (mAP > best_mAP) or (mAP_ema > best_mAP)
            best_mAP = max(mAP, mAP_ema, best_mAP)
            save_checkpoint({
                'state_dict': model.state_dict(),
                'epoch': epoch + 1,
                'best_mAP': best_mAP,
            }, is_best, fpath=osp.join(args.logs_dir, 'model.pth.tar'))
            save_checkpoint({
                'state_dict': model_ema.state_dict(),
                'epoch': epoch + 1,
                'best_mAP': best_mAP,
            }, (is_best and (mAP <= mAP_ema)), fpath=osp.join(args.logs_dir, 'model_ema.pth.tar'))

            print('\n * Finished epoch {:3d}  model mAP: {:5.1%}  model_ema mAP: {:5.1%}  best: {:5.1%}{}\n'.
                  format(epoch, mAP, mAP_ema, best_mAP, ' *' if is_best else ''))

        lr_scheduler.step()

    print('==> Test with the best model:')
    checkpoint = load_checkpoint(osp.join(args.logs_dir, 'model_best.pth.tar'))
    model.load_state_dict(checkpoint['state_dict'])
    evaluator.evaluate(test_loader, dataset.query, dataset.gallery, cmc_flag=True)

    end_time = time.monotonic()
    print('Total running time: ', timedelta(seconds=end_time - start_time))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Self-paced contrastive learning on unsupervised re-ID")
    # data
    parser.add_argument('-d', '--dataset', type=str, default='market1501',
                        choices=datasets.names())
    parser.add_argument('-b', '--batch-size', type=int, default=256)
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('--height', type=int, default=256, help="input height")
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
    parser.add_argument('--eps-neighbor-gap', type=float, default=0.0,
                        help="different neighbor for measuring neighbour consistency regularization")
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
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--iters', type=int, default=200)
    parser.add_argument('--step-size', type=int, default=20)
    # training configs
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--print-freq', type=int, default=10)
    parser.add_argument('--eval-step', type=int, default=5)
    parser.add_argument('--temp', type=float, default=0.05,
                        help="temperature for scaling contrastive loss")
    parser.add_argument('--pooling-type', type=str, default='gem')
    parser.add_argument('--use-hard', action="store_true")
    parser.add_argument('--cam',  action="store_true")
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('-p', type=float, default=5.0)
    parser.add_argument('--ema-decay', type=float, default=0.99)
    parser.add_argument('--temp-dist', type=float, default=0.05)
    parser.add_argument('--resume', type=str, default='', metavar='PATH')
    parser.add_argument('--lambda1', type=float, default=1.0)
    parser.add_argument('--lambda2', type=float, default=1.0)
    parser.add_argument('--rampup-value', type=float, default=1.0)
    # path
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'data'))
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'logs'))
    
    
    main()