import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import os
import numpy as np

import layers


model_urls = {'resnet18':
             'https://download.pytorch.org/models/resnet18-5c106cde.pth'}
# number of resnet blocks per layer
resnet_layers = {18: [2, 2, 2, 2]}


def conv3x3(in_planes, out_planes, stride=1, pad=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=pad, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, circpad=False):
        super(BasicBlock, self).__init__()
        if circpad:
            pad = 0
            self.circpad = layers.CircularPad(1)
        else:
            pad = 1
            self.circpad = None
        self.conv1 = conv3x3(inplanes, planes, stride, pad=pad)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, pad=pad)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        if self.circpad is not None:
            x = self.circpad(x)
        out = self.conv1(x)

        out = self.bn1(out)
        out = self.relu(out)

        if self.circpad is not None:
            out = self.circpad(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(residual)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, circpad=False):
        super(Bottleneck, self).__init__()
        # NOT IMPLEMENTED!
        assert not circpad
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
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


class GroupConvBlock(nn.Module):
    def __init__(self, inplanes, planes, init,
                 n_elements=12,
                 n_homogeneous=0,
                 use_bn=True,
                 activation=nn.ReLU,
                 support=None,
                 homogeneous_output=True):
        super(GroupConvBlock, self).__init__()
        self.use_bn = use_bn
        self.conv1 = layers.homogeneous_or_group_conv(inplanes, planes,
                                                      init=init,
                                                      n_elements=n_elements,
                                                      n_homogeneous=n_homogeneous,
                                                      support=support,
                                                      homogeneous_output=homogeneous_output)
        self.bn1 = nn.BatchNorm2d(planes)
        try:
            self.relu = activation(inplace=True)
        except TypeError:
            # inplace not supported
            self.relu = activation()

    def forward(self, x):
        out = self.conv1(x)
        if self.use_bn:
            out = self.bn1(out)
        out = self.relu(out)

        return out


def get_pool_fn(viewpool):
    assert 'max' in viewpool or 'avg' in viewpool
    pool_fn = ((lambda *a, **kwa: torch.max(*a, **kwa)[0])
               if 'max' in viewpool
               else torch.mean)
    return pool_fn


class ResNetMVGCNN(nn.Module):
    """
    Args:
      full_homogeneous: if True, all convs are on homogeneous space; else only first is.
    """

    def __init__(self, block, channels, gconv_channels,
                 num_classes=3,
                 view_dropout=False,
                 n_group_elements=12,
                 n_homogeneous=0,
                 bn_after_gconv=True,
                 gconv_activation=nn.ReLU,
                 gconv_support=None,
                 viewpool='avg',
                 n_fc_before_gconv=False,
                 circpad=False,
                 full_homogeneous=True):
        assert gconv_channels
        self.inplanes = 64
        self.view_dropout = view_dropout
        self.viewpool = viewpool
        super(ResNetMVGCNN, self).__init__()
        if circpad:
            pad1, pad3 = 0, 0
            self.circpad1 = layers.CircularPad(1)
            self.circpad3 = layers.CircularPad(3)
        else:
            pad1, pad3 = 1, 3
            self.circpad1, self.circpad3 = None, None
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7,
                               stride=2, padding=pad3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=pad1)
        self.layer1 = self._make_layer(block, 64, channels[0], circpad=circpad)
        self.layer2 = self._make_layer(block, 128, channels[1],
                                       stride=2, circpad=circpad)
        self.layer3 = self._make_layer(block, 256, channels[2],
                                       stride=2, circpad=circpad)
        self.layer4 = self._make_layer(block, 512, channels[3],
                                       stride=2, circpad=circpad)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.final_descriptor = layers.Identity()
        self.initial_group = layers.Identity()
        self.gcnn_fc = nn.Linear(
            gconv_channels[-1] * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if n_fc_before_gconv > 0:
            nlayers = [512] + [gconv_channels[0]
                               for _ in range(n_fc_before_gconv)]
            fc_before_conv = []
            for i, (chin, chout) in enumerate(zip(nlayers, nlayers[1:])):
                fc_before_conv.append(nn.Linear(chin, chout))
                if i != len(nlayers) - 2:
                    # last mlp layer is linear
                    fc_before_conv.append(torch.nn.ReLU(inplace=True))
            self.fc_before_conv = nn.Sequential(*fc_before_conv)
            gconv_channels = [gconv_channels[0], *gconv_channels]
        else:
            self.fc_before_conv = None
            gconv_channels = [512, *gconv_channels]

        n_homogeneous_per_layer = np.array(
            [n_homogeneous for _ in gconv_channels])
        support_per_layer = [gconv_support for _ in gconv_channels]
        if not full_homogeneous and n_homogeneous > 0:
            n_homogeneous_per_layer[1:] = 0
            support_per_layer[0] = None
        self.gc_layers = [GroupConvBlock(chin, chout,
                                         init='he',
                                         n_elements=n_group_elements,
                                         n_homogeneous=nh,
                                         use_bn=bn_after_gconv,
                                         activation=gconv_activation,
                                         support=sup,
                                         homogeneous_output=full_homogeneous)
                          for chin, chout, nh, sup in zip(gconv_channels,
                                                          gconv_channels[1:],
                                                          n_homogeneous_per_layer,
                                                          support_per_layer)]
        self.gconv = nn.Sequential(*self.gc_layers)

    def _make_layer(self, block, planes, blocks, stride=1, circpad=False):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample,
                            circpad=circpad))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes,
                                circpad=circpad))

        return nn.Sequential(*layers)

    def forward(self, x):
        if self.view_dropout:
            nviews = x.shape[1]
            # number of views to keep
            nkeep = np.random.randint(
                self.view_dropout[0], self.view_dropout[1])
            idx = np.random.choice(nviews, nkeep, replace=False)
            x = x[:, idx]

        shp = x.shape
        x = x.view((shp[0] * shp[1], *shp[2:]))
        if self.circpad3 is not None:
            x = self.circpad3(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if self.circpad1 is not None:
            x = self.circpad1(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)

        if self.fc_before_conv is not None:
            x = self.fc_before_conv(x[..., 0, 0])

        # batch, views, ...
        x = x.view((shp[0], shp[1], x.shape[1]))

        # reintroduce missing views (make their features zero)
        # TODO: could we use geodesic average instead?
        if self.view_dropout:
            new_x = torch.zeros((x.shape[0], nviews, *x.shape[2:]),
                                requires_grad=False,
                                device=x.device)
            new_x[:, idx] = x
            x = new_x

        # make (batch, channels, views, 1)
        x = x.transpose(1, 2).unsqueeze(-1)
        x = self.initial_group(x)

        # apply G-CNN
        x = self.gconv(x)[..., 0]

        # final pooling
        pool_fn = get_pool_fn(self.viewpool)
        if 'verylate' not in self.viewpool:
            x = pool_fn(x, dim=2)
            x = self.final_descriptor(x)
            x = self.gcnn_fc(x)
        else:
            self.final_descriptor(pool_fn(x, dim=2))
            x = self.gcnn_fc(x.transpose(1, 2))
            x = pool_fn(x, dim=1)

        return x


def resnet_pretrained(model, depth):
    pretrained_dict = model_zoo.load_url(model_urls[f'resnet{depth}'])
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items()
                       if k in model_dict and v.shape == model_dict[k].shape}
    print('Loading {} inputs from pretrained model...'
          .format(len(pretrained_dict)))
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)


def resnet18mvgcnn(**kwargs):
    """Constructs a MVGCNN based on ResNet-18 model."""
    return resnet_mvgcnn(18, **kwargs)


def resnet_mvgcnn(depth, pretrained=False, **kwargs):
    """Constructs a MVGCNN based on ResNet-18 model."""
    model = ResNetMVGCNN(BasicBlock,
                         resnet_layers[depth],
                         **kwargs)
    if pretrained:
        pretrained_dict = model_zoo.load_url(
            model_urls['resnet{}'.format(depth)])
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items()
                           if k in model_dict and v.shape == model_dict[k].shape}
        print('Loading {} inputs from pretrained model...'
              .format(len(pretrained_dict)))
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    return model
