import os
import numpy as np
import torch
from torch import nn
import scipy.io as sio
import constants as cts


def get_cyclic12_idx():
    idx = np.zeros([12, 12])
    for i in range(12):
        for j in range(12):
            idx[i, j] = (i + j) % 12
    return idx


def get_ico60_idx():
    matt = sio.loadmat(os.path.join(os.path.dirname(__file__),
                                    'cayley.mat'))
    idx = matt['multi'] - 1
    return idx


class group_conv(nn.Module):
    """ Group Convolution of (icosahedral or 12-element cyclical group)"""

    def __init__(self, inplane, outplane,
                 init='id', n_elements=12, support=None):
        assert init in ['he', 'id']
        super(group_conv, self).__init__()
        if n_elements == 12:
            self.idx = get_cyclic12_idx()
        elif n_elements == 60:
            self.idx = get_ico60_idx()
        if support:
            self.idx = self.idx[:, support]
            n_elements = len(support)

        self.inplane = inplane
        self.outplane = outplane
        self.gc = nn.Conv2d(inplane,
                            outplane,
                            kernel_size=[1, n_elements])
        if init == 'he':
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight,
                                            nonlinearity='relu')
                    m.bias.data = torch.zeros(m.bias.data.shape)
        elif init == 'id':
            ww = torch.cat([torch.ones([1, 1]),
                            torch.zeros([1, n_elements - 1])],
                           dim=1)

            self.gc.weight.data = torch.zeros((self.gc.weight.data.shape))
            self.gc.bias.data = torch.zeros((self.gc.bias.data.shape))
            for m in range(inplane):
                self.gc.weight.data[m, m, :, :] = ww

    def forward(self, x):
        x = x[:, :, self.idx, 0]
        x = self.gc(x)

        return x


class homogeneous_conv(nn.Module):
    """ Homogeneous space convolution.

    Args:
      n_group: number of elements of the group
      n_homogeneous: number of elements on the homogeneous space
      support: indices of the filter support (wrt homogeneous space)
      homogeneous_output: pool back to homogeneous space if True; else stay on the group
    """

    def __init__(self, inplane, outplane, n_homogeneous=12,
                 init='id', n_group=60, support=None, homogeneous_output=True):
        assert n_group == 60
        assert n_homogeneous in [12, 20]
        super(homogeneous_conv, self).__init__()
        self.homogeneous_output = homogeneous_output
        classes = cts.homogeneous_tables[n_group][n_homogeneous]['classes']
        ids = cts.homogeneous_tables[n_group][n_homogeneous]['ids']

        if support is not None:
            ids = [ids[s] for s in support]
        self.conv = group_conv(inplane, outplane,
                               init=init,
                               n_elements=n_group,
                               support=ids)
        id2tri = {i: np.where(classes == i)[0][0]
                  for i in range(n_group)}
        self.conv.idx = np.vectorize(id2tri.__getitem__)(self.conv.idx)

        if homogeneous_output:
            pool_matrix = np.zeros((n_homogeneous, n_group))
            for i in range(n_homogeneous):
                pool_matrix[i][classes[i]] = 1. / classes[i].shape[0]
            self.pool = torch.Tensor(pool_matrix)

    def forward(self, x):
        x = self.conv(x)
        if self.homogeneous_output:
            # group-pooling back to # of original views
            x = self.pool.to(device=x.device) @ x

        return x


def homogeneous_or_group_conv(inplanes, outplanes, init, n_homogeneous,
                              n_elements=12, support=None, homogeneous_output=True):
    args = [inplanes, outplanes]
    kwargs = dict(init=init, support=support)
    if n_homogeneous > 0:
        return homogeneous_conv(*args,
                                n_homogeneous=n_homogeneous,
                                n_group=n_elements,
                                homogeneous_output=homogeneous_output,
                                **kwargs)
    else:
        return group_conv(*args, n_elements=n_elements, **kwargs)


class CircularPad(nn.Module):
    def __init__(self, pad):
        super(CircularPad, self).__init__()
        self.pad = pad
        self.zeropad = torch.nn.modules.padding.ConstantPad2d(
            (pad, pad, 0, 0), 0)

    def forward(self, x):
        x = torch.cat([x[..., -self.pad:, :], x, x[..., :self.pad, :]], dim=-2)
        x = self.zeropad(x)
        return x


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x
