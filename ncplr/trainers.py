from __future__ import print_function, absolute_import
import time
from .utils.meters import AverageMeter
import torch
import numpy as np

def sigmoid_rampup(current, rampup_length, p, rampup_max_value):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        phase = 1.0 - current / rampup_length
        return min(rampup_max_value, float(np.exp(- p * phase * phase)))


class ncplrTrainer(object):
    def __init__(self, encoder, encoder_ema, memory=None, ema_decay=None):
        super(ncplrTrainer, self).__init__()
        self.encoder = encoder
        self.encoder_ema = encoder_ema
        self.memory = memory
        self.ema_decay = ema_decay

    def train(self, epoch, data_loader, optimizer, print_freq=10, train_iters=400, neighbors=None, p=None, rampup_max_value=1.0):
        self.encoder.train()
        self.encoder_ema.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()

        losses = AverageMeter()
        losses_nce = AverageMeter()
        losses_ce = AverageMeter()
        losses_KL = AverageMeter()

        end = time.time()
        for i in range(train_iters):
            # load data
            inputs = data_loader.next()
            data_time.update(time.time() - end)

            # process inputs
            inputs1, inputs2, labels, indexes, neighbors, neighbor_dists = self._parse_data(inputs)

            f_out1, f_out_logits1 = self.encoder(inputs1)
            with torch.no_grad():
                f_out2, f_out_logits2 = self.encoder_ema(inputs2)
            f_out_logits1 = f_out_logits1[:, :self.num_cluster]
            f_out_logits2 = f_out_logits2[:, :self.num_cluster]

            # compute loss with the hybrid memory
            loss_nce, loss_ce, loss_KL = self.memory((f_out1, f_out2), (f_out_logits1, f_out_logits2), labels, indexes, neighbors, neighbor_dists, sigmoid_rampup(epoch+1, 50, p, rampup_max_value))
            loss = ( loss_nce + loss_ce + loss_KL )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # updata teacher network
            self._update_ema_variables(self.encoder, self.encoder_ema, self.ema_decay, train_iters, epoch * len(data_loader) + i + 1)

            losses.update(loss.item())
            losses_nce.update(loss_nce.item())
            losses_ce.update(loss_ce.item())
            losses_KL.update(loss_KL.item())

            # print log
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss_nce {:.3f} ({:.3f})\t'
                      'Loss_ce {:.3f} ({:.3f})\t'
                      'Loss_KL {:.3f} ({:.3f})\t'
                      'Loss {:.3f} ({:.3f})'
                      .format(epoch, i + 1, len(data_loader),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses_nce.val, losses_nce.avg,
                              losses_ce.val, losses_ce.avg,
                              losses_KL.val, losses_KL.avg,
                              losses.val, losses.avg))

    def _parse_data(self, inputs):
        imgs, _, pids, _, indexes = inputs
        batch_neighbors = []
        batch_neighbor_dists = []
        for index in indexes:
            batch_neighbors.append(self.neighbors[index])
            batch_neighbor_dists.append(self.neighbor_dists[index])
        return imgs[0].cuda(), imgs[1].cuda(), pids.cuda(), indexes.cuda(), batch_neighbors, batch_neighbor_dists

    def _update_ema_variables(self, model, ema_model, ema_decay, train_iters, global_step, rampup_epoch=50):
        beta = train_iters*rampup_epoch*(1-ema_decay)
        ema_decay = min(1-beta/(global_step+beta-1), ema_decay)

        for (ema_name, ema_param), (model_name, param) in zip(ema_model.named_parameters(), model.named_parameters()):
          ema_param.data.mul_(ema_decay).add_(1 - ema_decay, param.data)

