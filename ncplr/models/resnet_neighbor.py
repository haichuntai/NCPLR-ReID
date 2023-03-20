from __future__ import absolute_import
from torch import nn
from torch.nn import functional as F
from torch.nn import init
import torchvision
from .pooling import build_pooling_layer

__all__ = ['ResNet_neighbor', 'resnet18_neighbor', 'resnet34_neighbor', 'resnet50_neighbor', 'resnet101_neighbor',
           'resnet152_neighbor']

class ResNet_neighbor(nn.Module):
    __factory = {
        18: torchvision.models.resnet18,
        34: torchvision.models.resnet34,
        50: torchvision.models.resnet50,
        101: torchvision.models.resnet101,
        152: torchvision.models.resnet152,
    }

    def __init__(self, depth, pretrained=True, cut_at_pooling=False,
                 num_features=0, norm=False, dropout=0, num_classes=0, pooling_type='avg'):
        super(ResNet_neighbor, self).__init__()
        self.pretrained = pretrained
        self.depth = depth
        # Construct base (pretrained) resnet
        if depth not in ResNet_neighbor.__factory:
            raise KeyError("Unsupported depth:", depth)
        resnet = ResNet_neighbor.__factory[depth](pretrained=pretrained)
        resnet.layer4[0].conv2.stride = (1,1)
        resnet.layer4[0].downsample[0].stride = (1,1)

        self.num_classes = num_classes

        self.base = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4)
        
        if pooling_type == 'avg':
            print('pooling-type: avg')
            self.gap = nn.AdaptiveAvgPool2d(1)
        elif pooling_type == 'gem':
            print('pooling-type: gem')
            self.gap = build_pooling_layer('gem')

        # global feature classifiers
        self.bnneck = nn.BatchNorm1d(2048)
        init.constant_(self.bnneck.weight, 1)
        init.constant_(self.bnneck.bias, 0)
        self.bnneck.bias.requires_grad_(False)

        self.classifier = nn.Linear(2048, self.num_classes, bias=False)
        init.normal_(self.classifier.weight, std=0.001)

        if not pretrained:
            self.reset_params()

    def forward(self, x, cls=1):
        x = self.base(x)

        f_g = self.gap(x)
        f_g = f_g.view(x.size(0), -1)
        f_g = self.bnneck(f_g)

        if self.training is False:
            f_g = F.normalize(f_g)
            return f_g

        logits_g = self.classifier(f_g)

        return f_g, logits_g

    def reset_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)


def resnet18_neighbor(**kwargs):
    return ResNet_neighbor(18, **kwargs)


def resnet34_neighbor(**kwargs):
    return ResNet_neighbor(34, **kwargs)


def resnet50_neighbor(**kwargs):
    return ResNet_neighbor(50, **kwargs)


def resnet101_neighbor(**kwargs):
    return ResNet_neighbor(101, **kwargs)


def resnet152_neighbor(**kwargs):
    return ResNet_neighbor(152, **kwargs)



