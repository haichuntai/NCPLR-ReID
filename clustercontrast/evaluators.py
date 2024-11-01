from __future__ import print_function, absolute_import
import time
import collections
from collections import OrderedDict
import numpy as np
import torch
import random
import copy

from .evaluation_metrics import cmc, mean_ap
from .utils.meters import AverageMeter
from .utils.rerank import re_ranking
from .utils import to_torch


def extract_cnn_feature(model, inputs):
    inputs = to_torch(inputs).cuda()
    outputs = model(inputs)
    outputs = outputs.data.cpu()
    return outputs


def extract_features(model, data_loader, print_freq=50):
    model.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    features = OrderedDict()
    labels = OrderedDict()

    end = time.time()
    with torch.no_grad():
        for i, (imgs, fnames, pids, _, _) in enumerate(data_loader):
            data_time.update(time.time() - end)

            outputs = extract_cnn_feature(model, imgs)
            for fname, output, pid in zip(fnames, outputs, pids):
                features[fname] = output
                labels[fname] = pid

            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Extract Features: [{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      .format(i + 1, len(data_loader),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg))

    return features, labels















def extract_cnn_allfeature(model, inputs):
    inputs = to_torch(inputs).cuda()
    outputs, outputs_up, outputs_down = model(inputs)
    outputs = outputs.data.cpu()
    outputs_up = outputs_up.data.cpu()
    outputs_down = outputs_down.data.cpu()
    return outputs, outputs_up, outputs_down



def extract_allfeatures(model, data_loader, print_freq=50):
    model.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    features = OrderedDict()
    features_up = OrderedDict()
    features_down = OrderedDict()
    labels = OrderedDict()

    end = time.time()
    with torch.no_grad():
        for i, (imgs, fnames, pids, _, _) in enumerate(data_loader):
            data_time.update(time.time() - end)

            outputs, outputs_up, outputs_down = extract_cnn_allfeature(model, imgs)
            for fname, output, output_up, output_down, pid in zip(fnames, outputs, outputs_up, outputs_down, pids):
                features[fname] = output
                features_up[fname] = output_up
                features_down[fname] = output_down
                labels[fname] = pid

            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Extract Features: [{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      .format(i + 1, len(data_loader),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg))

    return features, features_up, features_down, labels

















def extract_cnn_allfeature_logits(model, inputs):
    inputs = to_torch(inputs).cuda()
    (outputs, outputs_up, outputs_down), (outputs_logits, outputs_up_logits, outputs_down_logits)  = model(inputs, return_logits=True)
    outputs = outputs.data.cpu()
    outputs_up = outputs_up.data.cpu()
    outputs_down = outputs_down.data.cpu()
    outputs_logits = outputs_logits.data.cpu()
    outputs_up_logits = outputs_up_logits.data.cpu()
    outputs_down_logits = outputs_down_logits.data.cpu()
    return (outputs, outputs_up, outputs_down), (outputs_logits, outputs_up_logits, outputs_down_logits)



def extract_allfeatures_logits(model, data_loader, print_freq=50):
    model.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    features = OrderedDict()
    features_up = OrderedDict()
    features_down = OrderedDict()
    features_logits = OrderedDict()
    features_up_logits = OrderedDict()
    features_down_logits = OrderedDict()
    labels = OrderedDict()

    end = time.time()
    with torch.no_grad():
        for i, (imgs, fnames, pids, _, _) in enumerate(data_loader):
            data_time.update(time.time() - end)

            (outputs, outputs_up, outputs_down), (outputs_logits, outputs_up_logits, outputs_down_logits) = extract_cnn_allfeature_logits(model, imgs)
            for fname, output, output_up, output_down, output_logit, output_up_logit, output_down_logit, pid in zip(fnames, outputs, outputs_up, outputs_down, outputs_logits, outputs_up_logits, outputs_down_logits, pids):
                features[fname] = output
                features_up[fname] = output_up
                features_down[fname] = output_down
                features_logits[fname] = output_logit
                features_up_logits[fname] = output_up_logit
                features_down_logits[fname] = output_down_logit
                labels[fname] = pid

            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Extract Features: [{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      .format(i + 1, len(data_loader),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg))

    return (features, features_up, features_down), (features_logits, features_up_logits, features_down_logits), labels


























def pairwise_distance(features, query=None, gallery=None):
    if query is None and gallery is None:
        n = len(features)
        x = torch.cat(list(features.values()))
        x = x.view(n, -1)
        dist_m = torch.pow(x, 2).sum(dim=1, keepdim=True) * 2
        dist_m = dist_m.expand(n, n) - 2 * torch.mm(x, x.t())
        return dist_m

    x = torch.cat([features[f].unsqueeze(0) for f, _, _ in query], 0)
    y = torch.cat([features[f].unsqueeze(0) for f, _, _ in gallery], 0)
    m, n = x.size(0), y.size(0)
    x = x.view(m, -1)
    y = y.view(n, -1)
    dist_m = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + \
           torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist_m.addmm_(1, -2, x, y.t())
    return dist_m, x.numpy(), y.numpy()


def evaluate_all(query_features, gallery_features, distmat, query=None, gallery=None,
                 query_ids=None, gallery_ids=None,
                 query_cams=None, gallery_cams=None,
                 cmc_topk=(1, 5, 10), cmc_flag=False):
    if query is not None and gallery is not None:
        query_ids = [pid for _, pid, _ in query]
        gallery_ids = [pid for _, pid, _ in gallery]
        query_cams = [cam for _, _, cam in query]
        gallery_cams = [cam for _, _, cam in gallery]
    else:
        assert (query_ids is not None and gallery_ids is not None
                and query_cams is not None and gallery_cams is not None)

    # Compute mean AP
    mAP = mean_ap(distmat, query_ids, gallery_ids, query_cams, gallery_cams)
    print('Mean AP: {:4.1%}'.format(mAP))

    if (not cmc_flag):
        return mAP

    cmc_configs = {
        'market1501': dict(separate_camera_set=False,
                           single_gallery_shot=False,
                           first_match_break=True),}
    cmc_scores = {name: cmc(distmat, query_ids, gallery_ids,
                            query_cams, gallery_cams, **params)
                  for name, params in cmc_configs.items()}

    print('CMC Scores:')
    for k in cmc_topk:
        print('  top-{:<4}{:12.1%}'.format(k, cmc_scores['market1501'][k-1]))
    return cmc_scores['market1501'], mAP


class Evaluator(object):
    def __init__(self, model):
        super(Evaluator, self).__init__()
        self.model = model

    def evaluate(self, data_loader, query, gallery, cmc_flag=False, rerank=False, use_part=False):
        if use_part:
            features, features_up, features_down, _ = extract_allfeatures(self.model, data_loader)
        else:
            features, _ = extract_features(self.model, data_loader)
        distmat, query_features, gallery_features = pairwise_distance(features, query, gallery)
        results = evaluate_all(query_features, gallery_features, distmat, query=query, gallery=gallery, cmc_flag=cmc_flag)

        if (not rerank):
            return results

        print('Applying person re-ranking ...')
        distmat_qq, _, _ = pairwise_distance(features, query, query)
        distmat_gg, _, _ = pairwise_distance(features, gallery, gallery)
        distmat = re_ranking(distmat.numpy(), distmat_qq.numpy(), distmat_gg.numpy())
        return evaluate_all(query_features, gallery_features, distmat, query=query, gallery=gallery, cmc_flag=cmc_flag)
