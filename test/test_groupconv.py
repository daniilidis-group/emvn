import numpy as np
import scipy

import torch
from pathlib import Path

import sys
path = Path(__file__).parent.parent / 'emvn'
sys.path.append(str(path))

import layers
import constants as cts


def run_groupconv(n_elements, support=None):
  chin, chout, bs = 4, 8, 10
  layer = layers.group_conv(chin, chout,
                            init='he',
                            n_elements=n_elements,
                            support=support)
  inp = torch.rand(bs, chin, n_elements, 1)
  out = layer(inp)
  if n_elements == 12:
    cayley = layers.get_cyclic12_idx()
  elif n_elements == 60:
    cayley = layers.get_ico60_idx()

  # test equivariance for all elements of the group
  for i in range(1, n_elements):
    rot_inp = inp[..., cayley[i], :]
    rot_out = layer(rot_inp)
    assert rot_out.shape == (bs, chout, n_elements, 1)
    gt_out = out[..., cayley[i], :]
    assert torch.allclose(gt_out, rot_out)
    assert not torch.allclose(gt_out, out)


def test_groupconv12_equivariance():
  run_groupconv(12)


def test_groupconv12_localized_equivariance():
  run_groupconv(12, [0, 1])
  run_groupconv(12, [0, 1, 2])


def test_groupconv60_equivariance():
  run_groupconv(60)


def test_groupconv60_localized_equivariance():
  run_groupconv(60, [0, 1])
  run_groupconv(60, [0, 1, 2])
  run_groupconv(60, [0, 8, 1, 15, 12, 25, 21, 19, 29, 7, 11, 20, 4])
  run_groupconv(60, [0, 8, 1, 15, 12, 25])
  

def run_hconv(n_homogeneous, support=None):
  """Run homogeneous convolution"""
  chin, chout, bs = 4, 8, 10
  n_group = 60
  layer = layers.homogeneous_conv(chin, chout,
                                  init='he',
                                  n_group=n_group,
                                  n_homogeneous=n_homogeneous,
                                  support=support)
  inp = torch.rand(bs, chin, n_homogeneous, 1)
  out = layer(inp)

  # TODO: find one known permutation instead of using the precomputed table
  # get cayley from another layer w/ full support!
  cayley = layers.homogeneous_conv(1, 1,
                                   n_group=n_group,
                                   n_homogeneous=n_homogeneous,
                                   support=None).conv.idx
  # test equivariance for all elements of the group
  for i in range(1, n_group):
    rot_inp = inp[..., cayley[i], :]
    rot_out = layer(rot_inp)
    assert rot_out.shape == (bs, chout, n_homogeneous, 1)
    gt_out = out[..., cayley[i], :]
    assert torch.allclose(gt_out, rot_out, atol=1e-6)
    assert not torch.allclose(gt_out, out)


def test_homogeneousconv60_equivariance():
  run_hconv(20)
  run_hconv(12)
  run_hconv(20, support=list(range(10)))
  run_hconv(12, support=list(range(6)))


def test_homogeneousconv60_id():
  inp = torch.rand(3, 4, 20, 1)
  layer = layers.homogeneous_conv(4, 4,
                                  init='id',
                                  n_group=60,
                                  n_homogeneous=20)
  out = layer(inp)
  torch.allclose(out, inp)
  
