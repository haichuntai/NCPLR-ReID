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

from clustercontrast import datasets
from clustercontrast import models
from clustercontrast.models.cm import ClusterMemory, InstanceMemory, RefineLabels
from clustercontrast.trainers import ClusterContrastTrainer
from clustercontrast.evaluators import Evaluator, extract_features, extract_allfeatures
from clustercontrast.utils.data import IterLoader
from clustercontrast.utils.data import transforms as T
from clustercontrast.utils.data.preprocessor import Preprocessor
from clustercontrast.utils.logging import Logger
from clustercontrast.utils.serialization import load_checkpoint, save_checkpoint
from clustercontrast.utils.faiss_rerank import compute_jaccard_distance
from clustercontrast.utils.data.sampler import RandomMultipleGallerySampler, RandomMultipleGallerySamplerNoCam
from clustercontrast.utils.osutils import mkdir_if_missing

start_epoch = best_mAP = 0

import datetime
import math
from sklearn.metrics import pairwise_distances

def get_davies_bouldin(X, labels):
    n_clusters = np.unique(labels).shape[0]
    centroids = np.zeros((n_clusters, len(X[0])), dtype=float)
    s_i = np.zeros(n_clusters)
    for k in range(n_clusters):  # 遍历每一个簇
        x_in_cluster = X[labels == k]  # 取当前簇中的所有样本
        centroids[k] = np.mean(x_in_cluster, axis=0)  # 计算当前簇的簇中心
        s_i[k] = pairwise_distances(x_in_cluster, [centroids[k]]).mean()  #
    centroid_distances = pairwise_distances(centroids)  # [K,K]
    combined_s_i_j = s_i[:, None] + s_i  # [K,k]
    centroid_distances[centroid_distances == 0] = np.inf
    scores = np.max(combined_s_i_j / centroid_distances, axis=1)
    return torch.from_numpy(scores).cuda()


def get_data(name, data_dir):
    root = osp.join(data_dir, name)
    dataset = datasets.create(name, root)
    return dataset


def get_train_loader(args, dataset, height, width, batch_size, workers,
                     num_instances, iters, trainset=None, no_cam=False):

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
        if no_cam:
            sampler = RandomMultipleGallerySamplerNoCam(train_set, num_instances)
        else:
            sampler = RandomMultipleGallerySampler(train_set, num_instances)
    else:
        sampler = None
    train_loader = IterLoader(
        DataLoader(Preprocessor(train_set, root=dataset.images_dir, transform=train_transformer, use_meanteacher=args.use_meanteacher),
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
                          num_classes=3000, pooling_type=args.pooling_type, use_part=args.use_part)
    # use CUDA
    model.cuda()
    model = nn.DataParallel(model)

    if args.use_meanteacher:
        model_ema = models.create(args.arch, num_features=args.features, norm=True, dropout=args.dropout,
                                  num_classes=3000, pooling_type=args.pooling_type, use_part=args.use_part)
        model_ema.cuda()
        model_ema = nn.DataParallel(model_ema)
        return model, model_ema

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
    if args.use_meanteacher:
        model, model_ema = create_model(args)
    else:
        model = create_model(args)
        model_ema = None

    # Evaluator
    evaluator = Evaluator(model)
    if args.use_meanteacher:
        evaluator_ema = Evaluator(model_ema)


    # Load from checkpoint
    if args.resume:
        # checkpoint = load_checkpoint(osp.join(args.resume, 'model_pth/model_epoch{}.pth.tar'.format(args.resume_epoch)))
        checkpoint = load_checkpoint(osp.join(args.resume, 'model_best.pth.tar'))
        model.load_state_dict(checkpoint['state_dict'])
        if args.use_meanteacher:
            # checkpoint_ema = load_checkpoint(osp.join(args.resume, 'model_ema_pth/model_ema_epoch{}.pth.tar'.format(args.resume_epoch)))
            checkpoint_ema = load_checkpoint(osp.join(args.resume, 'model_best.pth.tar'))
            model_ema.load_state_dict(checkpoint_ema['state_dict'])
        # evaluator.evaluate(test_loader, dataset.query, dataset.gallery, cmc_flag=False)

    # Optimizer
    params = [{"params": [value]} for _, value in model.named_parameters() if value.requires_grad]
    optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.1)

    # Trainer
    trainer = ClusterContrastTrainer(model, encoder_ema=model_ema, use_meanteacher=args.use_meanteacher)
    
    init_cluster = True
    for epoch in range(args.resume_epoch, args.epochs):
        with torch.no_grad():
            print('==> Create pseudo labels for unlabeled data')
            cluster_loader = get_test_loader(dataset, args.height, args.width,
                                             args.batch_size, args.workers, testset=sorted(dataset.train))

            if args.use_part:
                features, features_up, features_down, _ = extract_allfeatures(model, cluster_loader, print_freq=50)
                features = torch.cat([features[f].unsqueeze(0) for f, _, _ in sorted(dataset.train)], 0)
                features_up = torch.cat([features_up[f].unsqueeze(0) for f, _, _ in sorted(dataset.train)], 0)
                features_down = torch.cat([features_down[f].unsqueeze(0) for f, _, _ in sorted(dataset.train)], 0)
                if args.use_meanteacher:
                    if epoch < args.ins_epoch:
                        features_ema = features
                        features_ema_up = features_up
                        features_ema_down = features_down
                    else:
                        features_ema, features_ema_up, features_ema_down, _= extract_allfeatures(model_ema, cluster_loader, print_freq=50)
                        features_ema = torch.cat([features_ema[f].unsqueeze(0) for f, _, _ in sorted(dataset.train)], 0)
                        features_ema_up = torch.cat([features_ema_up[f].unsqueeze(0) for f, _, _ in sorted(dataset.train)], 0)
                        features_ema_down = torch.cat([features_ema_down[f].unsqueeze(0) for f, _, _ in sorted(dataset.train)], 0)
            else:
                features, true_labels = extract_features(model, cluster_loader, print_freq=50)
                features = torch.cat([features[f].unsqueeze(0) for f, _, _ in sorted(dataset.train)], 0)
                if args.use_meanteacher:
                    if epoch < args.ins_epoch:
                        features_ema = features
                    else:
                        features_ema, _= extract_features(model_ema, cluster_loader, print_freq=50)
                        features_ema = torch.cat([features_ema[f].unsqueeze(0) for f, _, _ in sorted(dataset.train)], 0)
            
            mkdir_if_missing(osp.join(args.logs_dir, 'saved_features/'))
            torch.save(features, osp.join(args.logs_dir, 'saved_features/gfeature_epoch{}.pth'.format(epoch)))
            if args.use_part:
                torch.save(features_up, osp.join(args.logs_dir, 'saved_features/p1feature_epoch{}.pth'.format(epoch)))
                torch.save(features_down, osp.join(args.logs_dir, 'saved_features/p2feature_epoch{}.pth'.format(epoch)))

            # pseudo_labels = np.array([true_labels[f].item() for f, _, _ in sorted(dataset.train)])
            # np.save('true_labels_msmt17.npy', pseudo_labels)
            # exit(0)

            if args.use_truelabel:
                pseudo_labels = np.array([true_labels[f].item() for f, _, _ in sorted(dataset.train)])
                num_ins = len(pseudo_labels)
                rerank_dist = np.zeros((num_ins, num_ins))
            else:
                # if epoch == 0:
                #     rerank_dist = compute_jaccard_distance(features, k1=args.k1, k2=1)
                # rerank_dist = compute_jaccard_distance(features, k1=args.k1, k2=args.k2, neighbor_dist=rerank_dist)
                rerank_dist = compute_jaccard_distance(features, k1=args.k1, k2=args.k2)

                if args.use_part:
                    rerank_dist_up = compute_jaccard_distance(features_up, k1=args.k1, k2=args.k2)
                    rerank_dist_down = compute_jaccard_distance(features_down, k1=args.k1, k2=args.k2)
                    lambda1_up = args.lambda1
                    lambda1_down = args.lambda1
                    rerank_dist_cluster = (1.0-lambda1_up-lambda1_down)*rerank_dist + lambda1_up*rerank_dist_up + lambda1_down*rerank_dist_down
                    # rerank_dist_cluster = 0.7*rerank_dist + 0.2*rerank_dist_up + 0.1*rerank_dist_down
                else:
                    rerank_dist_cluster = rerank_dist
                # rerank_dist_cluster = rerank_dist
                    
                if init_cluster:
                    init_cluster = False
                    # DBSCAN cluster
                    eps = args.eps
                    print('Clustering criterion: eps: {:.3f}'.format(eps))
                    cluster = DBSCAN(eps=eps, min_samples=4, metric='precomputed', n_jobs=-1)
                # select & cluster images as training set of this epochs
                pseudo_labels = cluster.fit_predict(rerank_dist_cluster)

            pseudo_labels_idx = np.where(pseudo_labels != -1)[0]
            num_cluster = len(set(pseudo_labels)) - (1 if -1 in pseudo_labels else 0)
            trainer.num_cluster = num_cluster

            print('==> Statistics for epoch {}: {} clusters'.format(epoch, num_cluster))
            print('==> Statistics for epoch {}: {} uncluster instances'.format(epoch, len(pseudo_labels) - len(pseudo_labels_idx)))
            
            end_time = time.monotonic()
            print('Neighbor running time: ', timedelta(seconds=end_time - start_time))


            # # conpute BDI value and relabelins
            # cluster_dbi = get_davies_bouldin(np.array(features[pseudo_labels_idx]), pseudo_labels[pseudo_labels_idx])


        # generate new dataset and calculate cluster centers
        @torch.no_grad()
        def generate_cluster_features(labels, features):
            centers = collections.defaultdict(list)
            for i, label in enumerate(labels):
                if label == -1:
                    continue
                centers[labels[i]].append(features[i])

            centers = [
                torch.stack(centers[idx], dim=0).mean(0) for idx in sorted(centers.keys())
            ]

            centers = torch.stack(centers, dim=0)
            return centers

        cluster_features = generate_cluster_features(pseudo_labels, features)
        if args.use_part:
            cluster_features_up = generate_cluster_features(pseudo_labels, features_up)
            cluster_features_down = generate_cluster_features(pseudo_labels, features_down)
        if args.use_meanteacher:
            cluster_features_ema = generate_cluster_features(pseudo_labels, features_ema)
            if args.use_part:
                cluster_features_ema_up = generate_cluster_features(pseudo_labels, features_ema_up)
                cluster_features_ema_down = generate_cluster_features(pseudo_labels, features_ema_down)

        # Initial classifier
        centroids_g = F.normalize(cluster_features, p=2, dim=1) # torch.Size([313, 2048])
        model.module.classifier.weight.data[:num_cluster].copy_(centroids_g)
        if args.use_part:
            centroids_p1 = F.normalize(cluster_features_up, p=2, dim=1) # torch.Size([313, 2048])
            model.module.classifier2.weight.data[:num_cluster].copy_(centroids_p1)
            centroids_p2 = F.normalize(cluster_features_down, p=2, dim=1) # torch.Size([313, 2048])
            model.module.classifier3.weight.data[:num_cluster].copy_(centroids_p2)
        if args.use_meanteacher:
            centroids_g_ema = F.normalize(cluster_features_ema, p=2, dim=1)
            model_ema.module.classifier.weight.data[:num_cluster].copy_(centroids_g_ema)
            if args.use_part:
                centroids_ema_p1 = F.normalize(cluster_features_ema_up, p=2, dim=1) # torch.Size([313, 2048])
                model_ema.module.classifier2.weight.data[:num_cluster].copy_(centroids_ema_p1)
                centroids_ema_p2 = F.normalize(cluster_features_ema_down, p=2, dim=1) # torch.Size([313, 2048])
                model_ema.module.classifier3.weight.data[:num_cluster].copy_(centroids_ema_p2)
        
        # New Search neighbors
        eps_neighbor_epoch = args.eps_neighbor
        pseudo_labels_valid = torch.from_numpy(pseudo_labels[pseudo_labels_idx])
        pseudo_labels_valid_mat = (pseudo_labels_valid[:, None] == pseudo_labels_valid[None])

        if args.use_mixdistneighbbor:
            rerank_dist = rerank_dist_cluster
            rerank_dist_up = rerank_dist_cluster
            rerank_dist_down = rerank_dist_cluster

        neighbors = torch.zeros((pseudo_labels_idx.shape[0], pseudo_labels_idx.shape[0]), dtype=torch.bool) # torch.Size([3899])
        neighbor_dist = torch.tensor(rerank_dist[pseudo_labels_idx][:, pseudo_labels_idx])
        neighbors[neighbor_dist <= eps_neighbor_epoch] = True
        # neighbors[~pseudo_labels_valid_mat] == False

        if args.use_part:
            if args.eps_partneighbor == args.eps_neighbor:
                neighbors_up = neighbors
                neighbors_down = neighbors
                neighbor_dist_up = neighbor_dist
                neighbor_dist_down = neighbor_dist
            else:
                eps_partneighbor_epoch = args.eps_partneighbor
                neighbors_up = torch.zeros((pseudo_labels_idx.shape[0], pseudo_labels_idx.shape[0]), dtype=torch.bool)
                neighbors_down = torch.zeros((pseudo_labels_idx.shape[0], pseudo_labels_idx.shape[0]), dtype=torch.bool)
                neighbor_dist_up = torch.tensor(rerank_dist_up[pseudo_labels_idx][:, pseudo_labels_idx])
                neighbor_dist_down = torch.tensor(rerank_dist_down[pseudo_labels_idx][:, pseudo_labels_idx])
                neighbors_up[neighbor_dist_up <= eps_partneighbor_epoch] = True
                neighbors_down[neighbor_dist_down <= eps_partneighbor_epoch] = True
                # neighbors_up[~pseudo_labels_valid_mat] == False
                # neighbors_down[~pseudo_labels_valid_mat] == False

        end_time = time.monotonic()
        print('Neighbor running time: ', timedelta(seconds=end_time - start_time))

        # Create hybrid memory
        memory = ClusterMemory(model.module.num_features, num_cluster, temp=args.temp, momentum=args.momentum, \
                               use_hard=args.use_hard, use_mean=args.use_mean, use_rand=args.use_rand).cuda()
        memory.features = F.normalize(cluster_features, dim=1).cuda()
        # memory.cluster_dbi = cluster_dbi
        memory.neighbors = neighbors.cpu()
        memory.neighbor_dists = neighbor_dist.cpu()
        memory.use_part = args.use_part
        if args.use_part:
            memory.features_up = F.normalize(cluster_features_up, dim=1).cuda()
            memory.features_down = F.normalize(cluster_features_down, dim=1).cuda()

        if args.use_auxmemory:
            aux_memory = ClusterMemory(model.module.num_features, num_cluster, temp=args.temp, momentum=args.momentum, \
                                       use_hard=args.use_hardaux, use_mean=args.use_meanaux, use_rand=args.use_randaux).cuda()
            aux_memory.features = F.normalize(cluster_features, dim=1).cuda()
            # aux_memory.cluster_dbi = cluster_dbi
            aux_memory.neighbors = neighbors.cpu()
            aux_memory.neighbor_dists = neighbor_dist.cpu()
            aux_memory.use_part = args.use_part
            if args.use_part:
                aux_memory.features_up = F.normalize(cluster_features_up, dim=1).cuda()
                aux_memory.features_down = F.normalize(cluster_features_down, dim=1).cuda()




        if args.use_insmemory:
            ins_memory = InstanceMemory(model.module.num_features, num_cluster, temp=args.temp, momentum=args.momentum, \
                                        posoption=args.ins_posoption, negoption=args.ins_negoption).cuda()

            # ins item
            ins_memory.labels = torch.tensor(pseudo_labels).cuda()[pseudo_labels_idx]  # [12936]
            # ins_memory.cluster_dbi = cluster_dbi
            
            # select_insfeats
            all_indexes = torch.arange(len(pseudo_labels))
            selected_insfeats = []
            if args.use_part:
                selected_insfeats_up = []
                selected_insfeats_down = []
            selected_insfeats = []
            pseudo_labels_torch = torch.from_numpy(pseudo_labels) # torch.Size([12936])
            pseudo_labels_valid = torch.from_numpy(pseudo_labels[pseudo_labels_idx]) # torch.Size([564])
            sorted_unique_labels = torch.unique(pseudo_labels_valid, sorted=True) # torch.Size([77])
            assert sorted_unique_labels.max()+1 == sorted_unique_labels.size(0)
            for sl in sorted_unique_labels:
                inds = (pseudo_labels_torch == sl).nonzero().view(-1)
                rand_inds = torch.randint(0, len(inds), (args.num_instances,)) # torch.Size([16])
                if args.use_meanteacher:
                    selected_insfeats.extend(features_ema[inds][rand_inds])
                    if args.use_part:
                        selected_insfeats_up.extend(features_ema_up[inds][rand_inds])
                        selected_insfeats_down.extend(features_ema_down[inds][rand_inds])
                else:
                    selected_insfeats.extend(features[inds][rand_inds])
                    if args.use_part:
                        selected_insfeats_up.extend(features_up[inds][rand_inds])
                        selected_insfeats_down.extend(features_down[inds][rand_inds])
            selected_insfeats = torch.stack(selected_insfeats) # torch.Size([1232, 2048])
            ins_memory.features = F.normalize(selected_insfeats, dim=1).cuda()
            if args.use_part:
                selected_insfeats_up = torch.stack(selected_insfeats_up)
                selected_insfeats_down = torch.stack(selected_insfeats_down)
                ins_memory.features_up = F.normalize(selected_insfeats_up, dim=1).cuda()
                ins_memory.features_down = F.normalize(selected_insfeats_down, dim=1).cuda()
            ins_memory.num_samples = len(pseudo_labels_idx)
            ins_memory.neighbors = neighbors.cpu()
            ins_memory.neighbor_dists = neighbor_dist.cpu()

            ins_memory.use_part = args.use_part

            end_time = time.monotonic()
            print('Ins running time: ', timedelta(seconds=end_time - start_time))
        

        if args.use_refinelabels:
            refine_labels = RefineLabels(model.module.num_features, num_cluster, option=args.ce_option,
                                         temp_dist=args.temp_dist, alpha=args.alpha, cmalpha=args.cmalpha, \
                                         topk_s=args.topk_s, topk_spart=args.topk_spart, temp_KLlogits=args.temp_KLlogits, temp_KLloss=args.temp_KLloss).cuda()
            refine_labels.neighbors = neighbors.cpu()
            refine_labels.neighbor_dists = neighbor_dist.cpu()
            # refine_labels.cluster_dbi = cluster_dbi
            refine_labels.use_part = args.use_part
            if args.use_part:
                refine_labels.features_up = F.normalize(cluster_features_up, dim=1).cuda()
                refine_labels.features_down = F.normalize(cluster_features_down, dim=1).cuda()
                refine_labels.neighbors_up = neighbors_up.cpu()
                refine_labels.neighbor_dists_up = neighbor_dist_up.cpu()
                refine_labels.neighbors_down = neighbors_down.cpu()
                refine_labels.neighbor_dists_down = neighbor_dist_down.cpu()
            refine_labels.neighbor_temp = args.neighbor_temp
        

        del cluster_loader, features

        memory.lambda2 = args.lambda2
        trainer.memory = memory
        if args.use_auxmemory:
            aux_memory.lambda2 = args.lambda2
            trainer.aux_memory = aux_memory
        if args.use_insmemory:
            ins_memory.lambda2 = args.lambda2
            trainer.ins_memory = ins_memory
        if args.use_refinelabels:
            refine_labels.lambda2 = args.lambda2
            trainer.refine_labels = refine_labels
        trainer.dbi_value = args.dbi_value

        pseudo_labeled_dataset = []
        for i, ((fname, _, cid), label) in enumerate(zip(sorted(dataset.train), pseudo_labels)):
            if label != -1:
                pseudo_labeled_dataset.append((fname, label.item(), cid))

        train_loader = get_train_loader(args, dataset, args.height, args.width,
                                        args.batch_size, args.workers, args.num_instances, iters,
                                        trainset=pseudo_labeled_dataset, no_cam=args.no_cam)

        train_loader.new_epoch()
        trainer.refineloss_weight = args.refineloss_weight

        # if epoch == 20:
        #     print("===================init encoder_ema==================")
        #     trainer.encoder_ema.load_state_dict(trainer.encoder.state_dict())

        trainer.train(epoch, train_loader, optimizer, print_freq=args.print_freq, train_iters=len(train_loader), 
                      loss_weight=args.loss_weight, ins_weight=args.ins_weight, extra_option=args.extra_option, \
                      use_auxmemory=args.use_auxmemory, use_insmemory=args.use_insmemory, use_refine_labels=args.use_refinelabels, \
                      ins_epoch=args.ins_epoch, use_part=args.use_part)
        
        end_time = time.monotonic()
        print('Total train running time: ', timedelta(seconds=end_time - start_time))


        if args.use_meanteacher:
            if (epoch + 1) % args.eval_step == 0 or (epoch == args.epochs - 1):
                if args.store_checkpoint:
                    is_best = True
                    save_checkpoint({
                        'state_dict': model.state_dict(),
                        'epoch': epoch + 1,
                        'best_mAP': best_mAP,
                    }, is_best, fpath=osp.join(args.logs_dir, 'model_pth/model_epoch{}.pth.tar'.format(epoch + 1)))
                    save_checkpoint({
                        'state_dict': model_ema.state_dict(),
                        'epoch': epoch + 1,
                        'best_mAP': best_mAP,
                    }, is_best, fpath=osp.join(args.logs_dir, 'model_ema_pth/model_ema_epoch{}.pth.tar'.format(epoch + 1)))
                    if args.store_checkpoint_eval:
                        evaluator.evaluate(test_loader, dataset.query, dataset.gallery, cmc_flag=False, use_part=args.use_part)
                        evaluator_ema.evaluate(test_loader, dataset.query, dataset.gallery, cmc_flag=False, use_part=args.use_part)
                else:
                    mAP = evaluator.evaluate(test_loader, dataset.query, dataset.gallery, cmc_flag=False, use_part=args.use_part)
                    mAP_ema = evaluator_ema.evaluate(test_loader, dataset.query, dataset.gallery, cmc_flag=False, use_part=args.use_part)
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
        else:
            if (epoch + 1) % args.eval_step == 0 or (epoch == args.epochs - 1):
                if args.store_checkpoint:
                    is_best = True
                    save_checkpoint({
                        'state_dict': model.state_dict(),
                        'epoch': epoch + 1,
                        'best_mAP': best_mAP,
                    }, is_best, fpath=osp.join(args.logs_dir, 'model_pth/model_epoch{}.pth.tar'.format(epoch + 1)))
                    if args.store_checkpoint_eval:
                        if (epoch + 1) % 5 == 0:
                            evaluator.evaluate(test_loader, dataset.query, dataset.gallery, cmc_flag=False, use_part=args.use_part)
                else:
                    mAP = evaluator.evaluate(test_loader, dataset.query, dataset.gallery, cmc_flag=False, use_part=args.use_part)
                    is_best = (mAP > best_mAP)
                    best_mAP = max(mAP, best_mAP)
                    save_checkpoint({
                        'state_dict': model.state_dict(),
                        'epoch': epoch + 1,
                        'best_mAP': best_mAP,
                    }, is_best, fpath=osp.join(args.logs_dir, 'checkpoint.pth.tar'))

                    print('\n * Finished epoch {:3d}  model mAP: {:5.1%}  best: {:5.1%}{}\n'.
                        format(epoch, mAP, best_mAP, ' *' if is_best else ''))

        lr_scheduler.step()

    end_time = time.monotonic()
    print('Total train running time: ', timedelta(seconds=end_time - start_time))

    if args.store_checkpoint:
        print('==> Test with the all model:')
        for ei in [50, 49, 48]: 
            if args.use_meanteacher:
                checkpoint = load_checkpoint(osp.join(args.logs_dir, 'model_ema_pth/model_ema_epoch{}.pth.tar'.format(ei)))
                model.load_state_dict(checkpoint['state_dict'])
                evaluator.evaluate(test_loader, dataset.query, dataset.gallery, cmc_flag=True, use_part=args.use_part)  
            else:
                checkpoint = load_checkpoint(osp.join(args.logs_dir, 'model_pth/model_epoch{}.pth.tar'.format(ei)))
                model.load_state_dict(checkpoint['state_dict'])
                evaluator.evaluate(test_loader, dataset.query, dataset.gallery, cmc_flag=True, use_part=args.use_part)  

    else:
        print('==> Test with the best model:')
        checkpoint = load_checkpoint(osp.join(args.logs_dir, 'model_best.pth.tar'))
        model.load_state_dict(checkpoint['state_dict'])
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
    parser.add_argument('--eps-partneighbor', type=float, default=0.0)
    parser.add_argument('--eps-negneighbor', type=float, default=0.7)
    parser.add_argument('--eps-neighbor-add', type=float, default=0.0)
    parser.add_argument('--eps-gap', type=float, default=0.02,
                        help="multi-scale criterion for measuring cluster reliability")
    parser.add_argument('--k1', type=int, default=30,
                        help="hyperparameter for jaccard distance")
    parser.add_argument('--k2', type=int, default=6,
                        help="hyperparameter for jaccard distance")

    # model
    parser.add_argument('-a', '--arch', type=str, default='resnet50',
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
    parser.add_argument('--neighbor-temp', type=float, default=1.0)
    
    parser.add_argument('--use-meanteacher',  action="store_true")
    parser.add_argument('--use-auxmemory',  action="store_true")
    parser.add_argument('--use-hardaux', action="store_true")
    parser.add_argument('--use-meanaux', action="store_true")
    parser.add_argument('--use-randaux', action="store_true")
    parser.add_argument('--use-insmemory',  action="store_true")
    parser.add_argument('--use-refinelabels',  action="store_true")
    parser.add_argument('--use-truelabel',  action="store_true")
    parser.add_argument('--use-part',  action="store_true")
    parser.add_argument('--use-mixdistneighbbor',  action="store_true")

    main()
