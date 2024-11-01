from __future__ import print_function, absolute_import
import time
from .utils.meters import AverageMeter

import torch
from .models.cm import get_refinelabels

class ClusterContrastTrainer(object):
    def __init__(self, encoder, memory=None, encoder_ema=None, use_meanteacher=False):
        super(ClusterContrastTrainer, self).__init__()
        self.encoder = encoder
        self.memory = memory
        self.encoder_ema = encoder_ema
        self.use_meanteacher = use_meanteacher

    def train(self, epoch, data_loader, optimizer, print_freq=10, train_iters=400, loss_weight=1.0, extra_option=0, \
              use_auxmemory=False, use_insmemory=False, use_refine_labels=False, \
              ins_weight=False, ins_epoch=20, use_part=False):
        self.encoder.train()
        if self.use_meanteacher:
            self.encoder_ema.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()

        losses = AverageMeter()
        nce = AverageMeter()
        auxnce = AverageMeter()
        ins = AverageMeter()
        refine = AverageMeter()

        end = time.time()
        for iter in range(train_iters):
            # load data
            inputs = data_loader.next()
            data_time.update(time.time() - end)

            # process inputs
            if self.use_meanteacher:
                inputs, inputs_ema, labels, indexes = self._parse_data_ema(inputs)
            else:
                inputs, labels, indexes = self._parse_data(inputs)

            # forward
            if use_part:
                (f_out, f_out_up, f_out_down), (f_logits, f_logits_up, f_logits_down) = self.encoder(inputs)
                f_logits = f_logits[:, :self.num_cluster]
                f_logits_up = f_logits_up[:, :self.num_cluster]
                f_logits_down = f_logits_down[:, :self.num_cluster]
                if self.use_meanteacher:
                    with torch.no_grad():
                        (f_out_ema, f_out_ema_up, f_out_ema_down), (f_logits_ema, f_logits_ema_up, f_logits_ema_down) = self.encoder_ema(inputs_ema)
                    f_logits_ema = f_logits_ema[:, :self.num_cluster]
                    f_logits_ema_up = f_logits_ema_up[:, :self.num_cluster]
                    f_logits_ema_down = f_logits_ema_down[:, :self.num_cluster]
                    f_out = [(f_out, f_out_up, f_out_down), (f_out_ema, f_out_ema_up, f_out_ema_down)]
                    f_logits = [(f_logits, f_logits_up, f_logits_down), (f_logits_ema, f_logits_ema_up, f_logits_ema_down)]
                else:
                    f_out = (f_out, f_out_up, f_out_down)
                    f_logits = (f_logits, f_logits_up, f_logits_down)
            else:
                f_out, f_logits = self.encoder(inputs)
                f_logits = f_logits[:, :self.num_cluster]
                if self.use_meanteacher:
                    with torch.no_grad():
                        f_out_ema, f_logits_ema = self.encoder_ema(inputs_ema)
                    f_logits_ema = f_logits_ema[:, :self.num_cluster]
                    f_out = [f_out, f_out_ema]
                    f_logits = [f_logits, f_logits_ema]
            # print("f_out shape: {}".format(f_out.shape))
            # compute loss with the hybrid memory
            # loss = self.memory(f_out, indexes)


            if use_refine_labels:
                loss_refine, part_weight = self.refine_labels(f_out, f_logits, labels, indexes, extra_option=extra_option, epoch=epoch, use_meanteacher=self.use_meanteacher, extra_labels=None, loss_weight=loss_weight) # [cm_refinelabels, cmaux_refinelabels, ins_refinelabels])
            else:
                part_weight = None

            loss = 0 
            loss_nce = self.memory(f_out, f_logits, labels, indexes,  extra_option=extra_option, epoch=epoch, use_meanteacher=self.use_meanteacher, dbi_value=self.dbi_value, part_weight=part_weight)
            loss += loss_nce
            if use_auxmemory:
                loss_auxnce = self.aux_memory(f_out, f_logits, labels, indexes, extra_option=extra_option, epoch=epoch, use_meanteacher=self.use_meanteacher, dbi_value=self.dbi_value, part_weight=part_weight)
                loss = (loss + loss_auxnce) * 0.5
            if use_insmemory:
                if epoch >= ins_epoch:
                    loss_ins = self.ins_memory(f_out, f_logits, labels, indexes, extra_option=extra_option, epoch=epoch, use_meanteacher=self.use_meanteacher, dbi_value=self.dbi_value, part_weight=part_weight)
                    loss = (loss + loss_ins) * 0.5
            if use_refine_labels:
                loss += self.refineloss_weight * loss_refine

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # updata teacher network
            if self.use_meanteacher:
                self._update_ema_variables(self.encoder, self.encoder_ema, ema_decay=0.999)

            losses.update(loss.item())
            nce.update(loss_nce.item())
            auxnce.update(loss_auxnce.item() if use_auxmemory else 0)
            ins.update(loss_ins.item() if use_insmemory and epoch >= ins_epoch else 0)
            refine.update(loss_refine.item() if use_refine_labels else 0)

            # print log
            batch_time.update(time.time() - end)
            end = time.time()

            if (iter + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                    #   'Time {:.3f} ({:.3f})\t'
                    #   'Data {:.3f} ({:.3f})\t'
                      'nce {:.3f} ({:.3f})\t'
                      'auxnce {:.3f} ({:.3f})\t'
                      'ins {:.3f} ({:.3f})\t'
                      'ce {:.3f} ({:.3f})\t'
                      'Loss {:.3f} ({:.3f})'
                      .format(epoch, iter + 1, len(data_loader),
                            #   batch_time.val, batch_time.avg,
                            #   data_time.val, data_time.avg,
                              nce.val, nce.avg,
                              auxnce.val, auxnce.avg,
                              ins.val, ins.avg,
                              refine.val, refine.avg,
                              losses.val, losses.avg))

    def _parse_data(self, inputs):
        imgs, _, pids, _, indexes = inputs
        return imgs.cuda(), pids.cuda(), indexes.cuda()
    
    def _parse_data_ema(self, inputs):
        imgs, imgs_ema, _, pids, _, indexes = inputs
        return imgs.cuda(), imgs_ema.cuda(), pids.cuda(), indexes.cuda()

    def _forward(self, inputs):
        return self.encoder(inputs)

    def _update_ema_variables(self, model, ema_model, ema_decay):
        for (ema_name, ema_param), (model_name, param) in zip(ema_model.named_parameters(), model.named_parameters()):
          ema_param.data.mul_(ema_decay).add_(1 - ema_decay, param.data)

