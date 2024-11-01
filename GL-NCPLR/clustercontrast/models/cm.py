import collections
import numpy as np
from abc import ABC
import torch
import torch.nn.functional as F
from torch import nn, autograd

import math

class CM(autograd.Function):

    @staticmethod
    def forward(ctx, inputs, targets, features, momentum):
        ctx.features = features
        ctx.momentum = momentum
        ctx.save_for_backward(inputs, targets)
        outputs = inputs.mm(ctx.features.t())

        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        inputs, targets = ctx.saved_tensors
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(ctx.features)

        # momentum update
        # print('cm: {}'.format(targets.unique().size(0)))
        for x, y in zip(inputs, targets):
            ctx.features[y] = ctx.momentum * ctx.features[y] + (1. - ctx.momentum) * x
            ctx.features[y] /= ctx.features[y].norm()

        return grad_inputs, None, None, None


def cm(inputs, indexes, features, momentum=0.5):
    return CM.apply(inputs, indexes, features, torch.Tensor([momentum]).to(inputs.device))








class CM_Mean(autograd.Function):

    @staticmethod
    def forward(ctx, inputs, targets, features, momentum):
        ctx.features = features
        ctx.momentum = momentum
        ctx.save_for_backward(inputs, targets)
        outputs = inputs.mm(ctx.features.t())

        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        inputs, targets = ctx.saved_tensors
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(ctx.features)

        batch_centers = collections.defaultdict(list)
        for instance_feature, index in zip(inputs, targets.tolist()):
            batch_centers[index].append(instance_feature)

        for index, features in batch_centers.items():
            ctx.features[index] = ctx.features[index] * ctx.momentum + (1 - ctx.momentum) * torch.stack(features).mean(dim=0)
            ctx.features[index] /= ctx.features[index].norm()

        return grad_inputs, None, None, None


def cm_mean(inputs, indexes, features, momentum=0.5):
    return CM_Mean.apply(inputs, indexes, features, torch.Tensor([momentum]).to(inputs.device))












class CM_Rand(autograd.Function):

    @staticmethod
    def forward(ctx, inputs, targets, features, momentum):
        ctx.features = features
        ctx.momentum = momentum
        ctx.save_for_backward(inputs, targets)
        outputs = inputs.mm(ctx.features.t())

        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        inputs, targets = ctx.saved_tensors
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(ctx.features)

        # # momentum update
        # rand_indexes = torch.randperm(inputs.size(0)).cuda()
        # inputs, targets = inputs[rand_indexes], targets[rand_indexes]
        # for x, y in zip(inputs, targets):
        #     ctx.features[y] = ctx.momentum * ctx.features[y] + (1. - ctx.momentum) * x
        #     ctx.features[y] /= ctx.features[y].norm()

        # return grad_inputs, None, None, None

        batch_centers = collections.defaultdict(list)
        for instance_feature, index in zip(inputs, targets.tolist()):
            batch_centers[index].append(instance_feature)

        for index, features in batch_centers.items():
            rand_ind = torch.randint(0, len(features), (1,))
            ctx.features[index] = ctx.features[index] * ctx.momentum + (1 - ctx.momentum) * torch.stack(features)[rand_ind] # .mean(dim=0)
            ctx.features[index] /= ctx.features[index].norm()

        return grad_inputs, None, None, None


def cm_rand(inputs, indexes, features, momentum=0.5):
    return CM_Rand.apply(inputs, indexes, features, torch.Tensor([momentum]).to(inputs.device))










class CM_Hard(autograd.Function):

    @staticmethod
    def forward(ctx, inputs, targets, features, momentum):
        ctx.features = features
        ctx.momentum = momentum
        ctx.save_for_backward(inputs, targets)
        outputs = inputs.mm(ctx.features.t())

        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        inputs, targets = ctx.saved_tensors
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(ctx.features)

        batch_centers = collections.defaultdict(list)
        for instance_feature, index in zip(inputs, targets.tolist()):
            batch_centers[index].append(instance_feature)

        for index, features in batch_centers.items():
            distances = []
            for feature in features:
                distance = feature.unsqueeze(0).mm(ctx.features[index].unsqueeze(0).t())[0][0]
                distances.append(distance.cpu().numpy())

            median = np.argmin(np.array(distances))
            # median = np.argmax(np.array(distances))
            ctx.features[index] = ctx.features[index] * ctx.momentum + (1 - ctx.momentum) * features[median]
            ctx.features[index] /= ctx.features[index].norm()

        return grad_inputs, None, None, None


def cm_hard(inputs, indexes, features, momentum=0.5):
    return CM_Hard.apply(inputs, indexes, features, torch.Tensor([momentum]).to(inputs.device))










class HM(autograd.Function):

    @staticmethod
    def forward(ctx, inputs, indexes, features, momentum):
        ctx.features = features
        ctx.momentum = momentum
        ctx.save_for_backward(inputs, indexes)
        outputs = inputs.mm(ctx.features.t())

        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        inputs, indexes = ctx.saved_tensors
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(ctx.features)

        # momentum update
        for x, y in zip(inputs, indexes):
            ctx.features[y] = ctx.momentum * ctx.features[y] + (1. - ctx.momentum) * x  
            # ctx.features[y] = 0.1 * ctx.features[y] + (1. - 0.1) * x                       
                     
            ctx.features[y] /= ctx.features[y].norm()

        return grad_inputs, None, None, None


def hm(inputs, indexes, features, momentum=0.5):
    return HM.apply(inputs, indexes, features, torch.Tensor([momentum]).to(inputs.device))










class HM_ema(autograd.Function):

    @staticmethod
    def forward(ctx, inputs, inputs_ema, targets, features, indexes, momentum):
        ctx.features = features
        ctx.momentum = momentum
        ctx.indexes = indexes
        ctx.save_for_backward(inputs_ema, targets)
        outputs = inputs.mm(ctx.features.t())

        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        inputs, targets = ctx.saved_tensors
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(ctx.features)

        # print('hm: {}'.format(targets.unique().size(0)))
        # update feat
        for index in targets.unique():
            inds_for_feat = (index == targets).nonzero().squeeze()
            # assert inds_for_feat.size(0) == 16
            ctx.features[index*16: (index+1)*16] = inputs[inds_for_feat]
            ctx.features[index*16: (index+1)*16] /= ctx.features[index*16: (index+1)*16].norm()

        return grad_inputs, None, None, None, None, None


def hm_ema(inputs, inputs_ema, targets, features, indexes, momentum=0.5):
    return HM_ema.apply(inputs, inputs_ema, targets, features, indexes, torch.Tensor([momentum]).to(inputs.device))

















class ClusterMemory(nn.Module, ABC):
    def __init__(self, num_features, num_samples, temp=0.05, momentum=0.2, use_hard=False, use_mean=False, use_rand=False):
        super(ClusterMemory, self).__init__()
        self.num_features = num_features
        self.num_samples = num_samples

        self.momentum = momentum
        self.temp = temp
        self.use_hard = use_hard
        self.use_mean = use_mean
        self.use_rand = use_rand

        self.register_buffer('features', torch.zeros(num_samples, num_features))

        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, inputs_logits, targets, indexes, epsilon=1e-6, extra_option=0, epoch=None, use_meanteacher=False, dbi_value=1.3, part_weight=None):
        if self.use_part:
            if use_meanteacher:
                inputs_ema, inputs_ema_up, inputs_ema_down = inputs[1]
                inputs_logits_ema, inputs_logits_ema_up, inputs_logits_ema_down = inputs_logits[1]
                inputs, inputs_up, inputs_down = inputs[0]
                inputs_logits, inputs_logits_up, inputs_logits_down = inputs_logits[0]
            else:
                inputs, inputs_up, inputs_down = inputs
                inputs_logits, inputs_logits_up, inputs_logits_down = inputs_logits
            inputs = F.normalize(inputs, dim=1).cuda()
            inputs_up = F.normalize(inputs_up, dim=1).cuda()
            inputs_down = F.normalize(inputs_down, dim=1).cuda()
            if use_meanteacher:
                inputs_ema = F.normalize(inputs_ema, dim=1).cuda()
                inputs_ema_up = F.normalize(inputs_ema_up, dim=1).cuda()
                inputs_ema_down = F.normalize(inputs_ema_down, dim=1).cuda()
        else:
            if use_meanteacher:
                inputs_ema = inputs[1]
                inputs_logits_ema = inputs_logits[1]
                inputs = inputs[0]
                inputs_logits = inputs_logits[0]
            inputs = F.normalize(inputs, dim=1).cuda()
            if use_meanteacher:
                inputs_ema = F.normalize(inputs_ema, dim=1).cuda()
        
        if self.use_hard:
            outputs = cm_hard(inputs, targets, self.features, self.momentum)
            if self.use_part:
                outputs_up = cm_hard(inputs_up, targets, self.features_up, self.momentum)
                outputs_down = cm_hard(inputs_down, targets, self.features_down, self.momentum)
        elif self.use_mean:
            outputs = cm_mean(inputs, targets, self.features, self.momentum)
            if self.use_part:
                outputs_up = cm_mean(inputs_up, targets, self.features_up, self.momentum)
                outputs_down = cm_mean(inputs_down, targets, self.features_down, self.momentum)
        elif self.use_rand:
            outputs = cm_rand(inputs, targets, self.features, self.momentum)
            if self.use_part:
                outputs_up = cm_rand(inputs_up, targets, self.features_up, self.momentum)
                outputs_down = cm_rand(inputs_down, targets, self.features_down, self.momentum)
        else:
            outputs = cm(inputs, targets, self.features, self.momentum)
            if self.use_part:
                outputs_up = cm(inputs_up, targets, self.features_up, self.momentum)
                outputs_down = cm(inputs_down, targets, self.features_down, self.momentum)
        outputs /= self.temp
        if self.use_part:
            outputs_up /= self.temp
            outputs_down /= self.temp

        # torch.Size([256, 77])
        loss = F.cross_entropy(outputs, targets)
        if self.use_part:
            if extra_option == 1:
                loss_up = (F.cross_entropy(outputs_up, targets, reduction='none') * part_weight[0]).sum() / part_weight[0].sum()
                loss_down = (F.cross_entropy(outputs_down, targets, reduction='none') * part_weight[1]).sum() / part_weight[1].sum()
            else:
                loss_up = F.cross_entropy(outputs_up, targets)
                loss_down = F.cross_entropy(outputs_down, targets)
            lambda2 = self.lambda2
            loss = (1-lambda2)*loss + lambda2 *loss_up + lambda2 *loss_down
            # loss = (1-lambda2)*loss + lambda2 *loss_up
            # loss = (1-lambda2)*loss + lambda2 *loss_down


        return loss

        




    














class InstanceMemory(nn.Module, ABC):
    def __init__(self, num_features, num_samples, temp=0.05, momentum=0.2, posoption=0, negoption=0):
        super(InstanceMemory, self).__init__()
        self.num_features = num_features
        self.num_samples = num_samples

        self.momentum = momentum
        self.temp = temp

        print("========posoption: {}, negoption: {} =========".format(posoption, negoption))
        self.posoption = posoption
        self.negoption = negoption

        self.register_buffer('features', torch.zeros(num_samples, num_features))
        self.register_buffer('labels', torch.zeros(num_samples))

    def forward(self, inputs, inputs_logits, targets, indexes, epsilon=1e-6, extra_option=0, epoch=None, use_meanteacher=False, momentum=0.0, dbi_value=1.3, part_weight=None):
        if self.use_part:
            if use_meanteacher:
                inputs_ema, inputs_ema_up, inputs_ema_down = inputs[1]
                inputs_logits_ema, inputs_logits_ema_up, inputs_logits_ema_down = inputs_logits[1]
                inputs, inputs_up, inputs_down = inputs[0]
                inputs_logits, inputs_logits_up, inputs_logits_down = inputs_logits[0]
            else:
                inputs, inputs_up, inputs_down = inputs
                inputs_logits, inputs_logits_up, inputs_logits_down = inputs_logits
            inputs = F.normalize(inputs, dim=1).cuda()
            inputs_up = F.normalize(inputs_up, dim=1).cuda()
            inputs_down = F.normalize(inputs_down, dim=1).cuda()
            if use_meanteacher:
                inputs_ema = F.normalize(inputs_ema, dim=1).cuda()
                inputs_ema_up = F.normalize(inputs_ema_up, dim=1).cuda()
                inputs_ema_down = F.normalize(inputs_ema_down, dim=1).cuda()
        else:
            if use_meanteacher:
                inputs_ema = inputs[1]
                inputs_logits_ema = inputs_logits[1]
                inputs = inputs[0]
                inputs_logits = inputs_logits[0]
            inputs = F.normalize(inputs, dim=1).cuda()
            if use_meanteacher:
                inputs_ema = F.normalize(inputs_ema, dim=1).cuda()
        

        # ins memory
        outputs_ins = hm_ema(inputs, inputs_ema, targets, self.features, indexes, momentum=momentum)
        if self.use_part:
            outputs_ins_up = hm_ema(inputs_up, inputs_ema_up, targets, self.features_up, indexes, momentum=momentum)
            outputs_ins_down = hm_ema(inputs_down, inputs_ema_down, targets, self.features_down, indexes, momentum=momentum)
        outputs_ins /= self.temp
        if self.use_part:
            outputs_ins_up /= self.temp
            outputs_ins_down /= self.temp

        B = outputs_ins.size(0)

        assert (self.labels[indexes] == targets).all()
        labels = self.labels.clone()
        

        outputs_ins = torch.exp(outputs_ins) # torch.Size([256, 3903])
        masked_sim_insmemory = torch.zeros(B, 2).float().cuda()
        sim2insmemory = outputs_ins
        if self.use_part:
            outputs_ins_up = torch.exp(outputs_ins_up) # torch.Size([256, 3903])
            masked_sim_insmemory_up = torch.zeros(B, 2).float().cuda()
            sim2insmemory_up = outputs_ins_up

            outputs_ins_down = torch.exp(outputs_ins_down) # torch.Size([256, 3903])
            masked_sim_insmemory_down = torch.zeros(B, 2).float().cuda()
            sim2insmemory_down = outputs_ins_down

        batch_sim = torch.exp(inputs.mm(inputs_ema.t())/self.temp) # torch.Size([256, 256])
        if self.use_part:
            batch_sim_up = torch.exp(inputs_up.mm(inputs_ema_up.t())/self.temp)
            batch_sim_down = torch.exp(inputs_down.mm(inputs_ema_down.t())/self.temp)
    
        for i, index_i in enumerate(indexes):
            # 挑选正样本
            ind_b = (targets[i] == targets)
            sim_i_pos = batch_sim[i][ind_b].min()
            if self.use_part:
                sim_i_pos_up = batch_sim_up[i][ind_b].min()
                sim_i_pos_down = batch_sim_down[i][ind_b].min()
            # 挑选负样本
            topk_neg = 256
            s_inds = torch.ones(sim2insmemory[i].shape[0], dtype=torch.bool)
            s_inds[targets[i]*16: (targets[i]+1)*16] = False
            sim_i_neg = sim2insmemory[i][s_inds]
            sim_i_neg, k_indices = sim_i_neg.topk(k=topk_neg, dim=0, largest=True, sorted=True)
            if self.use_part:
                sim_i_neg_up = sim2insmemory_up[i][s_inds]
                sim_i_neg_up, k_indices = sim_i_neg_up.topk(k=topk_neg, dim=0, largest=True, sorted=True)
                sim_i_neg_down = sim2insmemory_down[i][s_inds]
                sim_i_neg_down, k_indices = sim_i_neg_down.topk(k=topk_neg, dim=0, largest=True, sorted=True)

            masked_sim_insmemory[i][0] = sim_i_pos
            masked_sim_insmemory[i][1] = sim_i_neg.sum()
            if self.use_part:
                masked_sim_insmemory_up[i][0] = sim_i_pos_up
                masked_sim_insmemory_up[i][1] = sim_i_neg_up.sum()
                
                masked_sim_insmemory_down[i][0] = sim_i_pos_down
                masked_sim_insmemory_down[i][1] = sim_i_neg_down.sum()

        targets_insmemory = torch.zeros(B, dtype=targets.dtype).cuda()

        masked_sums = masked_sim_insmemory.sum(1, keepdim=True) + epsilon
        masked_sim_insmemory = masked_sim_insmemory/masked_sums
        ins_loss =  F.nll_loss(torch.log(masked_sim_insmemory+1e-6), targets_insmemory)
        if self.use_part:
            masked_sums_up = masked_sim_insmemory_up.sum(1, keepdim=True) + epsilon
            masked_sim_insmemory_up = masked_sim_insmemory_up/masked_sums_up

            masked_sums_down = masked_sim_insmemory_down.sum(1, keepdim=True) + epsilon
            masked_sim_insmemory_down = masked_sim_insmemory_down/masked_sums_down

            if extra_option == 1:
                ins_loss_up =  (F.nll_loss(torch.log(masked_sim_insmemory_up+1e-6), targets_insmemory, reduction='none') * part_weight[0]).sum() / part_weight[0].sum()
                ins_loss_down =  (F.nll_loss(torch.log(masked_sim_insmemory_down+1e-6), targets_insmemory, reduction='none') * part_weight[1]).sum() / part_weight[1].sum()
            else:
                ins_loss_up =  F.nll_loss(torch.log(masked_sim_insmemory_up+1e-6), targets_insmemory)
                ins_loss_down =  F.nll_loss(torch.log(masked_sim_insmemory_down+1e-6), targets_insmemory)

            lambda2 = self.lambda2
            ins_loss = (1.0-lambda2)*ins_loss + lambda2 *ins_loss_up + lambda2 *ins_loss_down
            # ins_loss = ins_loss + lambda2 *ins_loss_up + lambda2 *ins_loss_down

        return ins_loss














class RefineLabels(nn.Module, ABC):
    def __init__(self, num_features, num_samples, alpha=0.0, cmalpha=0.0, temp_dist=0.05, option=0, topk_s=0, topk_spart=0, temp_KLlogits=0, temp_KLloss=0):
        super(RefineLabels, self).__init__()
        self.num_features = num_features
        self.num_samples = num_samples

        self.alpha = alpha
        self.cmalpha = cmalpha
        self.temp_dist = temp_dist
        self.topk_s = topk_s
        self.topk_spart = topk_spart
        self.temp_KLlogits = temp_KLlogits
        self.temp_KLloss = temp_KLloss
        print("========ce option: {}=========".format(option))
        self.option = option
        self.softmax = nn.Softmax(dim=1)
        self.logsoftmax = nn.LogSoftmax(dim=1)
        # self.KLDivLoss_mean = nn.KLDivLoss(reduction='batchmean')
        self.KLDivLoss = nn.KLDivLoss(reduction='none')

    def forward(self, inputs, inputs_logits, targets, indexes, epsilon=1e-6, extra_option=0, epoch=None, use_meanteacher=False, dbi_value=1.3, extra_labels=None, loss_weight=1.0):
        if self.use_part:
            if use_meanteacher:
                inputs_ema, inputs_ema_up, inputs_ema_down = inputs[1]
                inputs_logits_ema, inputs_logits_ema_up, inputs_logits_ema_down = inputs_logits[1]
                inputs, inputs_up, inputs_down = inputs[0]
                inputs_logits, inputs_logits_up, inputs_logits_down = inputs_logits[0]
            else:
                inputs, inputs_up, inputs_down = inputs
                inputs_logits, inputs_logits_up, inputs_logits_down = inputs_logits
            inputs = F.normalize(inputs, dim=1).cuda()
            inputs_up = F.normalize(inputs_up, dim=1).cuda()
            inputs_down = F.normalize(inputs_down, dim=1).cuda()
            if use_meanteacher:
                inputs_ema = F.normalize(inputs_ema, dim=1).cuda()
                inputs_ema_up = F.normalize(inputs_ema_up, dim=1).cuda()
                inputs_ema_down = F.normalize(inputs_ema_down, dim=1).cuda()
        else:
            if use_meanteacher:
                inputs_ema = inputs[1]
                inputs_logits_ema = inputs_logits[1]
                inputs = inputs[0]
                inputs_logits = inputs_logits[0]
            inputs = F.normalize(inputs, dim=1).cuda()
            if use_meanteacher:
                inputs_ema = F.normalize(inputs_ema, dim=1).cuda()

        refined_targets = get_refinelabels(inputs_logits, targets, indexes, self.neighbors, self.neighbor_dists, ce_alpha=self.alpha, option=self.option, topk_s=self.topk_s, neighbor_temp=self.neighbor_temp)
        loss_ce = (-refined_targets * self.logsoftmax(inputs_logits)).sum(1).mean()

        if extra_option == 2:
            targets_onehot = torch.zeros_like(inputs_logits).scatter_(1, targets.unsqueeze(1), 1)
            logits_neighbors = get_refinelabels(inputs_logits, targets, indexes, self.neighbors, self.neighbor_dists, ce_alpha=0.0, option=self.option, neighbor_temp=self.neighbor_temp)
            
            # refined_targets = self.alpha * targets_onehot + (1-self.alpha) * logits_neighbors.detach()
            # loss_ce = (-refined_targets * self.logsoftmax(inputs_logits)).sum(1).mean()

            
            # , topk_s=self.topk_s
            # , topk_s=self.topk_spart
            # , topk_s=self.topk_spart

            interaction_factor = 1.0
            if self.use_part:
                logits_neighbors_up = get_refinelabels(inputs_logits_up, targets, indexes, self.neighbors_up, self.neighbor_dists_up, ce_alpha=0.0, option=self.option, neighbor_temp=self.neighbor_temp)
                logits_neighbors_down = get_refinelabels(inputs_logits_down, targets, indexes, self.neighbors_down, self.neighbor_dists_down, ce_alpha=0.0, option=self.option, neighbor_temp=self.neighbor_temp)
                
                # KL_up = self.KLDivLoss(torch.log(logits_neighbors), logits_neighbors_up).sum(dim=1) # torch.Size([256])
                # KL_down = self.KLDivLoss(torch.log(logits_neighbors), logits_neighbors_down).sum(dim=1)
                # KL散度中以第二个参数为参考
                # 以全局特征为参考，得到全局特征logits
                KL_up = self.KLDivLoss(torch.log(logits_neighbors_up), logits_neighbors).sum(dim=1) # torch.Size([256])
                KL_down = self.KLDivLoss(torch.log(logits_neighbors_down), logits_neighbors).sum(dim=1)
                temp_KL = self.temp_KLlogits
                weight_up = torch.exp(-KL_up/temp_KL)[:, None] * interaction_factor
                weight_down = torch.exp(-KL_down/temp_KL)[:, None] * interaction_factor
                

                # 整体特征的邻域
                logits_neighbors_new = (logits_neighbors + weight_up * logits_neighbors_up + weight_down * logits_neighbors_down) / (1 + weight_up + weight_down) # torch.Size([256, 814])
                # logits_neighbors_new = (logits_neighbors + weight_up * logits_neighbors_up) / (1 + weight_up) # torch.Size([256, 814])
                # logits_neighbors_new = (logits_neighbors + weight_down * logits_neighbors_down) / (1 + weight_down) # torch.Size([256, 814])
                
                # logits_neighbors_new = logits_neighbors
                #对加权之后的全局特征的logits进行topk选取
                if self.topk_s != 0:
                    # 只选取前K个类进行伪标签的优化
                    k_vals, g_k_indices = logits_neighbors_new.topk(k=self.topk_s, dim=1, largest=True, sorted=True) # torch.Size([256, 2]), torch.Size([256, 2])
                    g_mask = (logits_neighbors_new<k_vals.index_select(dim=1, index=torch.tensor(self.topk_s-1).cuda())).bool()
                    logits_neighbors_new[logits_neighbors_new<k_vals.index_select(dim=1, index=torch.tensor(self.topk_s-1).cuda())] *= 0
                refined_targets = self.alpha * targets_onehot + (1-self.alpha) * logits_neighbors_new.detach()
                loss_ce = (-refined_targets * self.logsoftmax(inputs_logits)).sum(1).mean()
                
                # 以局部特征为参考，得到局部特征的logits
                KL_up = self.KLDivLoss(torch.log(logits_neighbors), logits_neighbors_up).sum(dim=1) # torch.Size([256])
                KL_down = self.KLDivLoss(torch.log(logits_neighbors), logits_neighbors_down).sum(dim=1)
                temp_KL = self.temp_KLlogits
                weight_up = torch.exp(-KL_up/temp_KL)[:, None] * interaction_factor
                weight_down = torch.exp(-KL_down/temp_KL)[:, None] * interaction_factor

                logits_neighbors_up_new = (logits_neighbors_up + weight_up*logits_neighbors) / (1 + weight_up)
                logits_neighbors_down_new = (logits_neighbors_down + weight_down*logits_neighbors) / (1 + weight_down)

                #对加权之后的局部特征的logits进行topk选取
                if self.topk_spart != 0:
                    # k_vals, k_indices = logits_neighbors_up_new.topk(k=self.topk_spart, dim=1, largest=True, sorted=True)
                    # logits_neighbors_up_new[logits_neighbors_up_new<k_vals.index_select(dim=1, index=torch.tensor(self.topk_spart-1).cuda())] *= 0
                    # k_vals, k_indices = logits_neighbors_down_new.topk(k=self.topk_spart, dim=1, largest=True, sorted=True)
                    # logits_neighbors_down_new[logits_neighbors_down_new<k_vals.index_select(dim=1, index=torch.tensor(self.topk_spart-1).cuda())] *= 0
                    logits_neighbors_up_new[g_mask] *= 0
                    logits_neighbors_down_new[g_mask] *= 0

                
                refined_targets_up = self.cmalpha * targets_onehot + (1-self.cmalpha) * logits_neighbors_up_new.detach()
                # refined_targets_up = targets_onehot.detach()
                refined_targets_down = self.cmalpha * targets_onehot + (1-self.cmalpha) * logits_neighbors_down_new.detach()
                # refined_targets_down = targets_onehot.detach()

                loss_ce_up = (-refined_targets_up * self.logsoftmax(inputs_logits_up)).sum(1).mean()            
                loss_ce_down = (-refined_targets_down * self.logsoftmax(inputs_logits_down)).sum(1).mean()

                lambda2 = self.lambda2
                loss_ce = (1.0-lambda2)*loss_ce + lambda2 *loss_ce_up + lambda2 *loss_ce_down
                # loss_ce = (1.0-lambda2)*loss_ce + lambda2 *loss_ce_up
                # loss_ce = (1.0-lambda2)*loss_ce + lambda2 *loss_ce_down
            
                return loss_ce, None
        return loss_ce, None

        # part_option = self.option
        # if extra_option == 1:
        #     logits_neighbors = get_refinelabels(inputs_logits, targets, indexes, self.neighbors, self.neighbor_dists, ce_alpha=0.0, option=self.option, topk_s=self.topk_spart)
        #     logits_neighbors_up = get_refinelabels(inputs_logits_up, targets, indexes, self.neighbors_up, self.neighbor_dists_up, ce_alpha=0.0, option=part_option, topk_s=self.topk_spart)
        #     logits_neighbors_down = get_refinelabels(inputs_logits_down, targets, indexes, self.neighbors_down, self.neighbor_dists_down, ce_alpha=0.0, option=part_option, topk_s=self.topk_spart)
            
        #     # KL_up = self.KLDivLoss(torch.log(logits_neighbors), logits_neighbors_up).sum(dim=1) # torch.Size([256])
        #     # KL_down = self.KLDivLoss(torch.log(logits_neighbors), logits_neighbors_down).sum(dim=1)
        #     KL_up = self.KLDivLoss(torch.log(logits_neighbors_up), logits_neighbors).sum(dim=1) # torch.Size([256])
        #     KL_down = self.KLDivLoss(torch.log(logits_neighbors_down), logits_neighbors).sum(dim=1)

        #     temp_KL = self.temp_KLloss
        #     weight_up = torch.exp(-KL_up/temp_KL)
        #     weight_down = torch.exp(-KL_down/temp_KL)
        #     part_weight = [weight_up, weight_down]
        # else:
        #     part_weight = None


        
        # if self.use_part:
        #     part_topk_s = self.topk_spart
        #     refined_targets_up = get_refinelabels(inputs_logits_up, targets, indexes, self.neighbors_up, self.neighbor_dists_up, ce_alpha=self.cmalpha, option=part_option, topk_s=part_topk_s)
        #     refined_targets_down = get_refinelabels(inputs_logits_down, targets, indexes, self.neighbors_down, self.neighbor_dists_down, ce_alpha=self.cmalpha, option=part_option, topk_s=part_topk_s)
        #     if extra_option == 1:
        #         loss_ce_up = ((-refined_targets_up * self.logsoftmax(inputs_logits_up)).sum(1) * part_weight[0]).sum() / part_weight[0].sum()
        #         loss_ce_down = ((-refined_targets_down * self.logsoftmax(inputs_logits_down)).sum(1) * part_weight[1]).sum() / part_weight[1].sum()
        #     else:
        #         loss_ce_up = (-refined_targets_up * self.logsoftmax(inputs_logits_up)).sum(1).mean()
        #         loss_ce_down = (-refined_targets_down * self.logsoftmax(inputs_logits_down)).sum(1).mean()
        #     lambda2 = self.lambda2
        #     loss_ce = (1.0-lambda2)*loss_ce  + lambda2 *loss_ce_down #+ lambda2 *loss_ce_up #
        #     # loss_ce = loss_ce + lambda2 *loss_ce_up + lambda2 *loss_ce_down
        #     # loss_ce = (loss_ce + (loss_ce_up + loss_ce_down)*0.5) * loss_weight
            


        # return loss_ce, part_weight



        
        
        


        refined_targets = get_refinelabels(inputs_logits, targets, indexes, self.neighbors, self.neighbor_dists, ce_alpha=self.alpha, option=self.option, topk_s=self.topk_s)
        loss_ce = (-refined_targets * self.logsoftmax(inputs_logits)).sum(1).mean()

        if self.use_part:
            refined_targets_up = get_refinelabels(inputs_logits_up, targets, indexes, self.neighbors, self.neighbor_dists, ce_alpha=self.cmalpha, option=self.option, topk_s=self.topk_s)
            loss_ce_up = (-refined_targets_up * self.logsoftmax(inputs_logits_up)).sum(1).mean()

            refined_targets_down = get_refinelabels(inputs_logits_down, targets, indexes, self.neighbors, self.neighbor_dists, ce_alpha=self.cmalpha, option=self.option, topk_s=self.topk_s)
            loss_ce_down = (-refined_targets_down * self.logsoftmax(inputs_logits_down)).sum(1).mean()

            lambda2 = 0.15
            loss_ce = (1.0-lambda2)*loss_ce + lambda2 *loss_ce_up + lambda2 *loss_ce_down

        return loss_ce
        


        # if extra_option == 1:
        #     # ce_alpha = self.alpha+0.3 - 0.3*(math.exp(-epoch/10))
        #     ce_alpha = self.alpha + (1-self.alpha)*math.pow(epoch/50, 2)
        # elif extra_option == 2:
        #     ce_alpha = self.alpha + math.pow(epoch/50, 2)*0.1
        # elif extra_option == 3:
        #     ce_alpha = self.alpha + math.pow(epoch/50, 2)*0.2
        # elif extra_option == 4:
        #     ce_alpha = self.alpha + math.pow(epoch/50, 2)*0.3
        # else:
        #     ce_alpha = self.alpha
        

        # if extra_labels is not None:
        #     if epoch < 20:
        #         weight_label = [0.05, 0.05, 0.1]
        #     else:
        #         weight_label = [0.025, 0.025, 0.05]
        #     sum = 0
        #     mix_labels = torch.zeros_like(logits_neighbors).cuda()
        #     for extra_i, extra_label in enumerate(extra_labels):
        #         if extra_label is not None:
        #             mix_labels += extra_label * weight_label[extra_i]
        #             sum += weight_label[extra_i]
        #     mix_labels += logits_neighbors * (1-sum)
        #     logits_neighbors = mix_labels


        ce_alpha = self.alpha
        refined_targets = ce_alpha * targets_onehot + (1-ce_alpha) * logits_neighbors.detach()
        
        cmalpha = self.cmalpha
        cm_refined_targets = cmalpha * targets_onehot + (1-cmalpha) * logits_neighbors.detach()
        
        # if extra_option == 1:
        #     loss = 0
        #     targets_unique = targets.unique()
        #     class_num = targets_unique.size(0)
        #     assert targets_unique.size(0) == 16
        #     for target in targets_unique:
        #         inds_target = (target == targets).nonzero().flatten()
        #         loss_target = 0
        #         ins_num = inds_target.size(0)
        #         assert ins_num == 16
        #         for i in inds_target:
        #             if self.cluster_dbi[targets[i]] <= dbi_value and torch.argmax(inputs_logits[i]) != targets[i]:
        #                 loss_target += (-cm_refined_targets[i] * self.logsoftmax(inputs_logits[i][None])[0]).sum()
        #             else:
        #                 loss_target += (-refined_targets[i] * self.logsoftmax(inputs_logits[i][None])[0]).sum()
        #         loss_target /= ins_num
        #         loss += loss_target
        #     loss /= class_num
        #     return loss, cm_refined_targets
        # elif extra_option == 2:
        #     loss = 0
        #     targets_unique = targets.unique()
        #     class_num = targets_unique.size(0)
        #     assert targets_unique.size(0) == 16
        #     for target in targets_unique:
        #         inds_target = (target == targets).nonzero().flatten()
        #         loss_target = 0
        #         ins_num = inds_target.size(0)
        #         assert ins_num == 16
        #         for i in inds_target:
        #             if self.cluster_dbi[targets[i]] <= dbi_value and torch.argmax(inputs_logits[i]) != targets[i]:
        #                 loss_target += (-refined_targets[i] * self.logsoftmax(inputs_logits[i][None])[0]).sum()
        #             else:
        #                 loss_target += (-cm_refined_targets[i] * self.logsoftmax(inputs_logits[i][None])[0]).sum()
        #         loss_target /= ins_num
        #     loss /= class_num
        #     return loss, cm_refined_targets
        
        loss_ce = (-refined_targets * self.logsoftmax(inputs_logits)).sum(1).mean()
        
        return loss_ce, cm_refined_targets
    








def get_refinelabels(inputs_logits, targets, indexes, neighbors, neighbor_dists, ce_alpha=0.1, option=0, topk_s=0, neighbor_temp=1.0):
    # CE refine
    targets_onehot = torch.zeros_like(inputs_logits).scatter_(1, targets.unsqueeze(1), 1)

    unique_indexes = indexes.unique()
    batch2unique_inds = torch.tensor([ (ui == indexes).nonzero()[0] for ui in unique_indexes ]).cuda() # torch.Size([97])
    inputs_logits_unique = inputs_logits[batch2unique_inds]
    batch_neighbors = neighbors[indexes][:, unique_indexes].cuda()
    if option == 1:
        batch_neighbors_dists = neighbor_dists[indexes][:, unique_indexes].cuda()

    logits_neighbors = []
    for i in range(inputs_logits.size(0)):
        neighbors_batch_i = batch_neighbors[i].nonzero().flatten()
        i_neighbor = inputs_logits_unique.index_select(dim=0, index=neighbors_batch_i) # torch.Size([8, 311])
        
        if topk_s != 0:
            # 只选取前K个类进行伪标签的优化
            k_vals, k_indices = i_neighbor.topk(k=topk_s, dim=1, largest=True, sorted=True)
            i_neighbor[i_neighbor<k_vals.index_select(dim=1, index=torch.tensor(topk_s-1).cuda())] *= 0

        if option == 0:
            i_neighbor = nn.Softmax(dim=1)(i_neighbor).mean(dim=0)
        elif option == 1:
            i_neighbor_dist = batch_neighbors_dists[i][neighbors_batch_i]
            i_neighbor = (nn.Softmax(dim=1)(i_neighbor) * torch.softmax(i_neighbor_dist/neighbor_temp, dim=0)[:, None]).sum(dim=0)
        logits_neighbors.append(i_neighbor)
    logits_neighbors = torch.stack(logits_neighbors) # torch.Size([256, 311])
    
    refined_targets = ce_alpha * targets_onehot + (1-ce_alpha) * logits_neighbors.detach()

    return refined_targets