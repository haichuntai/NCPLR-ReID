import collections
import numpy as np
from abc import ABC
import torch
import torch.nn.functional as F
from torch import nn, autograd

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
        for x, y in zip(inputs, targets):
            ctx.features[y] = ctx.momentum * ctx.features[y] + (1. - ctx.momentum) * x
            ctx.features[y] /= ctx.features[y].norm()

        return grad_inputs, None, None, None


def cm(inputs, indexes, features, momentum=0.5):
    return CM.apply(inputs, indexes, features, torch.Tensor([momentum]).to(inputs.device))


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
            ctx.features[index] = ctx.features[index] * ctx.momentum + (1 - ctx.momentum) * features[median]
            ctx.features[index] /= ctx.features[index].norm()

        return grad_inputs, None, None, None


def cm_hard(inputs, indexes, features, momentum=0.5):
    return CM_Hard.apply(inputs, indexes, features, torch.Tensor([momentum]).to(inputs.device))

class ClusterMemory(nn.Module, ABC):
    def __init__(self, num_samples, num_features=2048, temp=0.05, momentum=0.2, use_hard=False, alpha=None,  temp_dist=None, lambda1=None, lambda2=None, eps_consistency=None):
        super(ClusterMemory, self).__init__()
        self.num_features = num_features
        self.num_samples = num_samples

        self.momentum = momentum
        self.temp = temp
        self.use_hard = use_hard
        self.alpha = alpha
        self.temp_dist = temp_dist
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.eps_consistency = eps_consistency

        print('alpha:{}'.format(alpha))

        self.softmax = nn.Softmax(dim=1)
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.KLDivLoss = nn.KLDivLoss(reduction='batchmean')

        self.register_buffer('features', torch.zeros(num_samples, num_features))

    def forward(self, inputs, inputs_logits, targets, indexes, neighbors, neighbor_dists, rampup):
        inputs1 = F.normalize(inputs[0], dim=1).cuda()
        if self.use_hard:
            outputs = cm_hard(inputs1, targets, self.features, self.momentum)
        else:
            outputs = cm(inputs1, targets, self.features, self.momentum)
        outputs /= self.temp
        loss_nce = F.cross_entropy(outputs, targets)

        targets_onehot = torch.zeros_like(inputs_logits[0]).scatter_(1, targets.unsqueeze(1), 1)
        idx2bidx = collections.defaultdict(list)
        for i, idx in enumerate(indexes):
            idx = int(idx)
            idx2bidx[idx] = i

        logits_neighbors1 = torch.FloatTensor().cuda()
        logits_neighbors1_KL = torch.FloatTensor().cuda()
        for neighbor, neighbor_dist in zip(neighbors, neighbor_dists):
            i_neighbor1 = torch.Tensor().cuda()
            i_neighbor1_KL = torch.Tensor().cuda()
            neighbor_dist_valid = []
            for i, nb in enumerate(neighbor):
                if nb in indexes:
                    if neighbor_dist[i] <= self.eps_consistency:
                        i_neighbor1_KL = torch.cat([i_neighbor1_KL, inputs_logits[0][idx2bidx[nb]].unsqueeze(0)], dim=0)
                    i_neighbor1 = torch.cat([i_neighbor1, inputs_logits[0][idx2bidx[nb]].unsqueeze(0)], dim=0) # student
                    i_neighbor1 = torch.cat([i_neighbor1, inputs_logits[1][idx2bidx[nb]].unsqueeze(0)], dim=0) # teacher
                    neighbor_dist_valid.extend([neighbor_dist[i], neighbor_dist[i]])
            
            neighbor_dist_valid_softmax = torch.softmax(torch.Tensor((np.array(neighbor_dist_valid))/self.temp_dist), dim=0).cuda()
            mean_i_neighbor1 = (self.softmax(i_neighbor1)*(neighbor_dist_valid_softmax.unsqueeze(1).expand_as(i_neighbor1))).sum(dim=0)
            logits_neighbors1 = torch.cat([logits_neighbors1, mean_i_neighbor1.unsqueeze(0)], dim=0)

            mean_i_neighbor1_KL = self.softmax(i_neighbor1_KL).mean(dim=0)
            logits_neighbors1_KL = torch.cat([logits_neighbors1_KL, mean_i_neighbor1_KL.unsqueeze(0)], dim=0)

        refined_targets1 = self.alpha * targets_onehot + (1-self.alpha) * logits_neighbors1.detach()
        loss_ce = (-refined_targets1 * self.logsoftmax(inputs_logits[0])).sum(1).mean()

        loss_KL = self.KLDivLoss(self.logsoftmax(inputs_logits[1]).detach(), logits_neighbors1_KL)

        return (loss_nce, self.lambda1*loss_ce, self.lambda2*rampup*loss_KL)
