# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 00:08:07 2019

@author: zehaojin
"""
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

import torch.nn as nn
import torch.utils.model_zoo as model_zoo


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, num_classes_=[1,56,56]):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.num_classes = num_classes
        self.num_classes_ = num_classes_

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=1)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1)

        number_features = 256
        kernel_size = 3
        final_conv = nn.Conv2d(number_features, self.num_classes, kernel_size=kernel_size)

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            final_conv,
            nn.AvgPool2d(kernel_size=self.num_classes_[1]-(kernel_size-1),stride=1)
        )



        final_conv_ = nn.Conv2d(number_features, self.num_classes_[0], kernel_size=1)
        self.classifier_ = nn.Sequential(
            nn.Dropout(p=0.5),
            final_conv_
        )

        final_conv_recon = nn.Conv2d(number_features, self.num_classes_[0], kernel_size=1)
        self.classifier_recon = nn.Sequential(
            nn.Dropout(p=0.5),
            final_conv_recon
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x_macro = self.classifier(x)
        x_micro = self.classifier_(x)
        x_recon = self.classifier_recon(x)

        # DEBUG
        #print(x_macro.size())
        #print(x_micro.size())

        x_macro = x_macro.squeeze()
        x_micro = x_micro.squeeze()
        x_recon = x_recon.squeeze()


        # DEBUG
        #print(x_macro.size())
        #print(x_micro.size())
        return x_macro, x_micro, x_recon


def resnetDMS(pretrained=False, **kwargs):
    """
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model

def resnetDMSV2(pretrained=False, **kwargs):
    """
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [4, 4, 4, 4], **kwargs)
    return model


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model


import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict

__all__ = ['DenseNet', 'densenet121', 'densenet169', 'densenet201', 'densenet161']


model_urls = {
    'densenet121': 'https://download.pytorch.org/models/densenet121-a639ec97.pth',
    'densenet169': 'https://download.pytorch.org/models/densenet169-b2777c0a.pth',
    'densenet201': 'https://download.pytorch.org/models/densenet201-c1103571.pth',
    'densenet161': 'https://download.pytorch.org/models/densenet161-8d451a50.pth',
}


def densenetDMS(pretrained=False,**kwargs):
    r"""Densenet-DMS model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = DenseNet(num_init_features=96, growth_rate=48, block_config=(3, 3, 3, 3),
                     **kwargs)
    return model

def densenetDMSV2(pretrained=False,**kwargs):
    r"""Densenet-DMSV2 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = DenseNet(num_init_features=96, growth_rate=48, block_config=(6, 6, 6, 6),
                     **kwargs)
    return model

def densenet121(pretrained=False, **kwargs):
    r"""Densenet-121 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = DenseNet(num_init_features=64, growth_rate=32, block_config=(6, 12, 24, 16),
                     **kwargs)
    if pretrained:
        # '.'s are no longer allowed in module names, but pervious _DenseLayer
        # has keys 'norm.1', 'relu.1', 'conv.1', 'norm.2', 'relu.2', 'conv.2'.
        # They are also in the checkpoints in model_urls. This pattern is used
        # to find such keys.
        pattern = re.compile(
            r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
        state_dict = model_zoo.load_url(model_urls['densenet121'])
        for key in list(state_dict.keys()):
            res = pattern.match(key)
            if res:
                new_key = res.group(1) + res.group(2)
                state_dict[new_key] = state_dict[key]
                del state_dict[key]
        model.load_state_dict(state_dict)
    return model


def densenet169(pretrained=False, **kwargs):
    r"""Densenet-169 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = DenseNet(num_init_features=64, growth_rate=32, block_config=(6, 12, 32, 32),
                     **kwargs)
    if pretrained:
        # '.'s are no longer allowed in module names, but pervious _DenseLayer
        # has keys 'norm.1', 'relu.1', 'conv.1', 'norm.2', 'relu.2', 'conv.2'.
        # They are also in the checkpoints in model_urls. This pattern is used
        # to find such keys.
        pattern = re.compile(
            r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
        state_dict = model_zoo.load_url(model_urls['densenet169'])
        for key in list(state_dict.keys()):
            res = pattern.match(key)
            if res:
                new_key = res.group(1) + res.group(2)
                state_dict[new_key] = state_dict[key]
                del state_dict[key]
        model.load_state_dict(state_dict)
    return model


def densenet201(pretrained=False, **kwargs):
    r"""Densenet-201 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = DenseNet(num_init_features=64, growth_rate=32, block_config=(6, 12, 48, 32),
                     **kwargs)
    if pretrained:
        # '.'s are no longer allowed in module names, but pervious _DenseLayer
        # has keys 'norm.1', 'relu.1', 'conv.1', 'norm.2', 'relu.2', 'conv.2'.
        # They are also in the checkpoints in model_urls. This pattern is used
        # to find such keys.
        pattern = re.compile(
            r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
        state_dict = model_zoo.load_url(model_urls['densenet201'])
        for key in list(state_dict.keys()):
            res = pattern.match(key)
            if res:
                new_key = res.group(1) + res.group(2)
                state_dict[new_key] = state_dict[key]
                del state_dict[key]
        model.load_state_dict(state_dict)
    return model


def densenet161(pretrained=False, **kwargs):
    r"""Densenet-161 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = DenseNet(num_init_features=96, growth_rate=48, block_config=(6, 12, 36, 24),
                     **kwargs)
    if pretrained:
        # '.'s are no longer allowed in module names, but pervious _DenseLayer
        # has keys 'norm.1', 'relu.1', 'conv.1', 'norm.2', 'relu.2', 'conv.2'.
        # They are also in the checkpoints in model_urls. This pattern is used
        # to find such keys.
        pattern = re.compile(
            r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
        state_dict = model_zoo.load_url(model_urls['densenet161'])
        for key in list(state_dict.keys()):
            res = pattern.match(key)
            if res:
                new_key = res.group(1) + res.group(2)
                state_dict[new_key] = state_dict[key]
                del state_dict[key]
        model.load_state_dict(state_dict)
    return model


class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
                        growth_rate, kernel_size=1, stride=1, bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                        kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)


class _Transition(nn.Sequential):
    def __init__(self, i, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class DenseNet(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_
    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
    """

    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0, num_classes=1000,num_classes_=[1,224,224]):

        super(DenseNet, self).__init__()
        self.num_classes = num_classes
        self.num_classes_ = num_classes_

        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(i,num_input_features=num_features, num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))

        # Final convolution is initialized differently form the rest
        number_features = num_features
        kernel_size = 3
        final_conv = nn.Conv2d(number_features, self.num_classes, kernel_size=kernel_size)

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            final_conv,
            nn.AvgPool2d(kernel_size=self.num_classes_[1]-(kernel_size-1),stride=1)
        )

        final_conv_ = nn.Conv2d(number_features, self.num_classes_[0], kernel_size=1)
        self.classifier_ = nn.Sequential(
            nn.Dropout(p=0.5),
            final_conv_
        )

        final_conv_recon = nn.Conv2d(number_features, self.num_classes_[0], kernel_size=1)
        self.classifier_recon = nn.Sequential(
            nn.Dropout(p=0.5),
            final_conv_recon
        )

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x_macro = self.classifier(x)
        x_micro = self.classifier_(x)
        x_recon = self.classifier_recon(x)

        # DEBUG
        #print(x_macro.size())
        #print(x_micro.size())

        x_macro = x_macro.squeeze()
        x_micro = x_micro.squeeze()
        x_recon = x_recon.squeeze()


        # DEBUG
        #print(x_macro.size())
        #print(x_micro.size())
        return x_macro, x_micro, x_recon


import math
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.utils.model_zoo as model_zoo


__all__ = ['SqueezeNet', 'squeezenet1_0', 'squeezenet1_1']


model_urls = {
    'squeezenet1_0': 'https://download.pytorch.org/models/squeezenet1_0-a815701f.pth',
    'squeezenet1_1': 'https://download.pytorch.org/models/squeezenet1_1-f364aa15.pth',
}


class Fire(nn.Module):

    def __init__(self, inplanes, squeeze_planes,
                 expand1x1_planes, expand3x3_planes):
        super(Fire, self).__init__()
        self.inplanes = inplanes
        self.squeeze = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1)
        self.squeeze_activation = nn.ReLU(inplace=True)
        self.expand1x1 = nn.Conv2d(squeeze_planes, expand1x1_planes,
                                   kernel_size=1)
        self.expand1x1_activation = nn.ReLU(inplace=True)
        self.expand3x3 = nn.Conv2d(squeeze_planes, expand3x3_planes,
                                   kernel_size=3, padding=1)
        self.expand3x3_activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.squeeze_activation(self.squeeze(x))
        return torch.cat([
            self.expand1x1_activation(self.expand1x1(x)),
            self.expand3x3_activation(self.expand3x3(x))
        ], 1)

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class SqueezeNet(nn.Module):

    def __init__(self, version=1.0, num_classes=1000, num_classes_=[1,224,224]):
        super(SqueezeNet, self).__init__()

        self.num_classes = num_classes
        self.num_classes_ = num_classes_

        if version != 2.0:
            self.features = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, stride=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
                Fire(64, 16, 64, 64),
                Fire(128, 16, 64, 64),
                Fire(128, 32, 128, 128),
                Fire(256, 32, 128, 128),
                Fire(256, 48, 128, 128),
                Fire(256, 64, 192, 192),
                Fire(384, 64, 256, 256),
                Fire(512, 64, 256, 256),
                Fire(512, 64, 256, 256)
            )
            number_features = 512
        elif version == 2.0:
            self.features = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, stride=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
                Fire(64, 16, 64, 64),
                Fire(128, 16, 64, 64),
                Fire(128, 32, 128, 128),
                Fire(256, 32, 128, 128),
                Fire(256, 48, 128, 128),
                Fire(256, 64, 192, 192),
                Fire(384, 64, 256, 256),
                Fire(512, 64, 256, 256),
                Fire(512, 64, 512, 512),
                Fire(1024, 64, 512, 512),
                Fire(1024, 64, 512, 512)
            )
            number_features = 1024

        # Final convolution is initialized differently form the rest
        kernel_size = 3
        final_conv = nn.Conv2d(number_features, self.num_classes, kernel_size=kernel_size)

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            final_conv,
            nn.AvgPool2d(kernel_size=self.num_classes_[1]-(kernel_size-1),stride=1)
        )

        final_conv_ = nn.Conv2d(number_features, self.num_classes_[0], kernel_size=1)
        self.classifier_ = nn.Sequential(
            nn.Dropout(p=0.5),
            final_conv_
        )

        final_conv_recon = nn.Conv2d(number_features, self.num_classes_[0], kernel_size=1)
        self.classifier_recon = nn.Sequential(
            nn.Dropout(p=0.5),
            final_conv_recon
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m is final_conv:
                    init.normal_(m.weight, mean=0.0, std=0.01)
                else:
                    init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)

        x_macro = self.classifier(x)
        x_micro = self.classifier_(x)
        x_recon = self.classifier_recon(x)

        # DEBUG
        #print(x_macro.size())
        #print(x_micro.size())

        x_macro = x_macro.squeeze()
        x_micro = x_micro.squeeze()
        x_recon = x_recon.squeeze()


        # DEBUG
        #print(x_macro.size())
        #print(x_micro.size())
        return x_macro, x_micro, x_recon

def squeezenetDMS(pretrained=False, **kwargs):
    r"""
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = SqueezeNet(version=1.1, **kwargs)
    return model

def squeezenetDMSV2(pretrained=False, **kwargs):
    r"""
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = SqueezeNet(version=2.0, **kwargs)
    return model

def squeezenet1_0(pretrained=False, **kwargs):
    r"""SqueezeNet model architecture from the `"SqueezeNet: AlexNet-level
    accuracy with 50x fewer parameters and <0.5MB model size"
    <https://arxiv.org/abs/1602.07360>`_ paper.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = SqueezeNet(version=1.0, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['squeezenet1_0']))
    return model


def squeezenet1_1(pretrained=False, **kwargs):
    r"""SqueezeNet 1.1 model from the `official SqueezeNet repo
    <https://github.com/DeepScale/SqueezeNet/tree/master/SqueezeNet_v1.1>`_.
    SqueezeNet 1.1 has 2.4x less computation and slightly fewer parameters
    than SqueezeNet 1.0, without sacrificing accuracy.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = SqueezeNet(version=1.1, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['squeezenet1_1']))
    return model



import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from  torch.autograd import Variable
import torch.optim as optim
from torchvision import datasets, transforms

import os
import sys
import pandas as pd
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage

'''
import argparse


# Define argument
parser = argparse.ArgumentParser()
parser.add_argument('--net', dest='net',default='squeezenet',type=str,help='choose between [squeezenet,densenet,resnet,combinenet]')
parser.add_argument('--n_grid', dest='n_grid',default=7,type=int,help='n_grid_asdfghjkl')
parser.add_argument('--train_final_layer', dest='train_final_layer',default=True,type=bool,help='whether or not to train the final layer of each model')
args = parser.parse_args()
'''

class args():
    def __init__():
        return None
args.net='resnet'
args.n_grid=7
args.train_final_layer=True


np.random.seed(999)
TRAINING_SAMPLES = 50
TESTING_SAMPLES = 10

folder = 'void_project/data/'
model_path = 'void_project/saved_model'
glo_batch_size = 10
test_num_batch = 50
N_GRID = 224
n_grid = 7

show_test_imgs = False
data_transform = transforms.Compose([
            transforms.ToTensor(), # scale to [0,1] and convert to tensor
            ])
target_transform = torch.Tensor



class VoidsDataset(Dataset): # torch.utils.data.Dataset
    def __init__(self, root_dir, train=True, transform=None, target_transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set
        self.train_file = 'yTrain_count.npy'#'data_train'
        self.test_file = 'yTest_count.npy'#'data_test'

        if self.train:
            self.path = os.path.join(self.root_dir, self.train_file)
            self.df = np.load(self.path)
        else:
            self.path = os.path.join(self.root_dir, self.test_file)
            self.df = np.load(self.path)


    def __getitem__(self, index):
        y=self.df[index]

        if self.train:
            img_path = self.root_dir+'xTrain_count.npy'
        else:
            img_path = self.root_dir+'xTest_count.npy'
        image=np.load(img_path)[index]
        x=np.zeros((3, 224, 224))
        for i in range(3):
            x[i,:,:]+=image
        #x=image
        return x, y


    def __len__(self):
        return self.df.shape[0]



if __name__ == '__main__':
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    model_names = ['squeezenet','densenet','resnet']
    #models = {'squeezenet':SqueezeNet.squeezenetDMSV2, 'densenet':DenseNet.densenetDMSV2,'resnet': ResNet.resnetDMSV2}
    models = {'squeezenet':squeezenetDMSV2, 'densenet':densenetDMSV2,'resnet': resnetDMSV2}
    save_fils = {'squeezenet':'squeezeDMSV2','densenet':'densenetDMSV2','resnet': 'resnetDMSV2'}
    n_grid = args.n_grid

    if not os.path.exists(model_path):
        os.mkdir(model_path)

    # number of output features
    macro_lensing = 2
    micro_lensing = [1, 56 , 56] # the same output feature size applies to source recon as well

    if args.net != "combinenet":
        net = models[args.net](pretrained=False,num_classes=macro_lensing,num_classes_=micro_lensing)
        print("Total number of trainable parameters: ", (count_parameters(net)))

    else:
        net = CombineNet(num_models=3)
        nets = [0]*3
        for i in range(3):
            if args.train_final_layer:
                nets[i] = torch.load(model_path+save_fils[model_names[i]] + '_' + str(n_grid) + '_combine')
            else:
                nets[i] = torch.load(model_path+save_fils[model_names[i]] + '_' + str(n_grid))

            nets[i].cuda()
            for param in nets[i].parameters():
                param.requires_grad = False

            if args.train_final_layer:
                for param in nets[i].classifier.parameters():
                    param.requires_grad = True

                for param in nets[i].classifier_.parameters():
                    param.requires_grad = True

                for param in nets[i].classifier_recon.parameters():
                    param.requires_grad = True
        print("Total number of trainable parameters: ", (count_parameters(net)+sum([count_parameters(model) for model in nets])))

    loss_KL = nn.KLDivLoss(reduction='none')
    loss_mse = nn.MSELoss(reduction='none')
    loss_mae = nn.SmoothL1Loss(reduction='none')
    loss_bce = nn.BCEWithLogitsLoss(reduction='none')

    net.cuda()
    optimizer = optim.Adam(net.parameters(), lr = 1e-4)
    best_accuracy = float("inf")



    if not os.path.exists('void_project/data/xTrain_count.npy'):
        print('no training files')

    print('Traning on dataset xTrain yTrain')

    for epoch in range(20):

        net.train()
        total_loss = 0.0
        total_counter = 0

        train_loader = torch.utils.data.DataLoader(VoidsDataset(folder, train=True, transform=data_transform, target_transform=target_transform),
                batch_size = glo_batch_size, shuffle = True)

        for batch_idx, (x,y) in enumerate(train_loader):
            x, y = x.float(), y.float()
            x,y = Variable(x).cuda(),Variable(y).cuda()

            optimizer.zero_grad()

            if args.net != "combinenet":
                X = x
            else:
                output_macro_list = [0]*3
                output_subhalo_list = [0]*3
                output_source_list = [0]*3
                for i in range(3):
                    output_macro_list[i],output_subhalo_list[i],output_source_list[i]  = nets[i](x)
                    output_macro_list[i] = output_macro_list[i].unsqueeze(1).unsqueeze(2)
                    output_subhalo_list[i] = output_subhalo_list[i].unsqueeze(1)
                    output_source_list[i] = output_source_list[i].unsqueeze(1)
                X = [0]*3
                X[0] = torch.cat(output_macro_list,dim=1)
                X[1] = torch.cat(output_subhalo_list,dim=1)
                X[2] = torch.cat(output_source_list,dim=1)

            #print(X.shape)
            output = net(X)

            # Calculate Losses
            loss_y = loss_mse(output[0], y)
            loss_y=loss_y.sum(0)
            #print(loss_y.shape,loss_y.dtype)
            #loss = torch.mean(loss_y)
            loss=loss_y[0]/(1.20006426e+01*glo_batch_size) + loss_y[1]/(1.45742838e+05*glo_batch_size)

            #stdfactor=torch.tensor([12.02084904,6871.38375189])

            #loss = torch.sum(loss_y/stdfactor)

            total_loss += loss.item()
            total_counter += 1

            loss.backward()
            optimizer.step()


        print(epoch, 'Train loss (averge per batch wise):', total_loss/(total_counter))
        try:
            with torch.no_grad():
                net.eval()
                total_loss = 0.0
                total_counter = 0

                test_loader = torch.utils.data.DataLoader(VoidsDataset(folder, train=False, transform=data_transform, target_transform=target_transform),
                    batch_size = glo_batch_size, shuffle = True)

                for batch_idx,(x,y)in enumerate(test_loader):
                    x, y = x.float(), y.float()
                    x,y = Variable(x).cuda(),Variable(y).cuda()

                    if args.net != "combinenet":
                        X = x
                    else:
                        output_macro_list = [0]*3
                        output_subhalo_list = [0]*3
                        output_source_list = [0]*3
                        for i in range(3):
                            output_macro_list[i],output_subhalo_list[i],output_source_list[i]  = nets[i](data)
                            output_macro_list[i] = output_macro_list[i].unsqueeze(1).unsqueeze(2)
                            output_subhalo_list[i] = output_subhalo_list[i].unsqueeze(1)
                            output_source_list[i] = output_source_list[i].unsqueeze(1)

                        X = [0]*3
                        X[0] = torch.cat(output_macro_list,dim=1)
                        X[1] = torch.cat(output_subhalo_list,dim=1)
                        X[2] = torch.cat(output_source_list,dim=1)

                    output = net(X)

                    # Calculate Losses
                    loss_y = loss_mse(output[0], y)
                    loss_y=loss_y.sum(0)
                    #print(loss_y.shape,loss_y.dtype)
                    #loss = torch.mean(loss_y)
                    loss=loss_y[0]/(1.20006426e+01*glo_batch_size) + loss_y[1]/(1.45742838e+05*glo_batch_size)

                    total_loss += loss.item()
                    total_counter += 1

                    if batch_idx % test_num_batch == 0 and batch_idx != 0:
                        break

                print(epoch, 'Test loss (averge per batch wise):', total_loss/(total_counter))
        except:
            pass

        if total_loss/(total_counter) < best_accuracy:
            best_accuracy = total_loss/(total_counter)
            torch.save(net, model_path + args.net + '_' + str(n_grid)+'_count1124')

            if args.net == "combinenet":
                for i in range(len(nets)):
                    torch.save(nets[i], model_path + save_fils[model_names[i]] + '_' + str(n_grid) + '_combine')

            print("saved to file.")
