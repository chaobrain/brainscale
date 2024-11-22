# Copyright 2024 BDP Ecosystem Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# -*- coding: utf-8 -*-

from __future__ import annotations

import brainstate as bst

__all__ = [
    # activation functions
    'ReLU', 'RReLU', 'Hardtanh', 'ReLU6', 'Sigmoid', 'Hardsigmoid',
    'Tanh', 'SiLU', 'Mish', 'Hardswish', 'ELU', 'CELU', 'SELU', 'GLU', 'GELU',
    'Hardshrink', 'LeakyReLU', 'LogSigmoid', 'Softplus', 'Softshrink', 'PReLU',
    'Softsign', 'Tanhshrink', 'Softmin', 'Softmax', 'Softmax2d', 'LogSoftmax',

    # dropout
    'Dropout', 'Dropout1d', 'Dropout2d', 'Dropout3d', 'AlphaDropout', 'FeatureAlphaDropout',

    # others
    'Identity', 'SpikeBitwise',
]


class ReLU(bst.nn.ReLU):
    __doc__ = bst.nn.ReLU.__doc__
    __module__ = 'brainscale.nn'


class RReLU(bst.nn.RReLU):
    __doc__ = bst.nn.RReLU.__doc__
    __module__ = 'brainscale.nn'


class Hardtanh(bst.nn.Hardtanh):
    __doc__ = bst.nn.Hardtanh.__doc__
    __module__ = 'brainscale.nn'


class ReLU6(bst.nn.ReLU6):
    __doc__ = bst.nn.ReLU6.__doc__
    __module__ = 'brainscale.nn'


class Sigmoid(bst.nn.Sigmoid):
    __doc__ = bst.nn.Sigmoid.__doc__
    __module__ = 'brainscale.nn'


class Hardsigmoid(bst.nn.Hardsigmoid):
    __doc__ = bst.nn.Hardsigmoid.__doc__
    __module__ = 'brainscale.nn'


class Tanh(bst.nn.Tanh):
    __doc__ = bst.nn.Tanh.__doc__
    __module__ = 'brainscale.nn'


class SiLU(bst.nn.SiLU):
    __doc__ = bst.nn.SiLU.__doc__
    __module__ = 'brainscale.nn'


class Mish(bst.nn.Mish):
    __doc__ = bst.nn.Mish.__doc__
    __module__ = 'brainscale.nn'


class Hardswish(bst.nn.Hardswish):
    __doc__ = bst.nn.Hardswish.__doc__
    __module__ = 'brainscale.nn'


class ELU(bst.nn.ELU):
    __doc__ = bst.nn.ELU.__doc__
    __module__ = 'brainscale.nn'


class CELU(bst.nn.CELU):
    __doc__ = bst.nn.CELU.__doc__
    __module__ = 'brainscale.nn'


class SELU(bst.nn.SELU):
    __doc__ = bst.nn.SELU.__doc__
    __module__ = 'brainscale.nn'


class GLU(bst.nn.GLU):
    __doc__ = bst.nn.GLU.__doc__
    __module__ = 'brainscale.nn'


class GELU(bst.nn.GELU):
    __doc__ = bst.nn.GELU.__doc__
    __module__ = 'brainscale.nn'


class Hardshrink(bst.nn.Hardshrink):
    __doc__ = bst.nn.Hardshrink.__doc__
    __module__ = 'brainscale.nn'


class LeakyReLU(bst.nn.LeakyReLU):
    __doc__ = bst.nn.LeakyReLU.__doc__
    __module__ = 'brainscale.nn'


class LogSigmoid(bst.nn.LogSigmoid):
    __doc__ = bst.nn.LogSigmoid.__doc__
    __module__ = 'brainscale.nn'


class Softplus(bst.nn.Softplus):
    __doc__ = bst.nn.Softplus.__doc__
    __module__ = 'brainscale.nn'


class Softshrink(bst.nn.Softshrink):
    __doc__ = bst.nn.Softshrink.__doc__
    __module__ = 'brainscale.nn'


class PReLU(bst.nn.PReLU):
    __doc__ = bst.nn.PReLU.__doc__
    __module__ = 'brainscale.nn'


class Softsign(bst.nn.Softsign):
    __doc__ = bst.nn.Softsign.__doc__
    __module__ = 'brainscale.nn'


class Tanhshrink(bst.nn.Tanhshrink):
    __doc__ = bst.nn.Tanhshrink.__doc__
    __module__ = 'brainscale.nn'


class Softmin(bst.nn.Softmin):
    __doc__ = bst.nn.Softmin.__doc__
    __module__ = 'brainscale.nn'


class Softmax(bst.nn.Softmax):
    __doc__ = bst.nn.Softmax.__doc__
    __module__ = 'brainscale.nn'


class Softmax2d(bst.nn.Softmax2d):
    __doc__ = bst.nn.Softmax2d.__doc__
    __module__ = 'brainscale.nn'


class LogSoftmax(bst.nn.LogSoftmax):
    __doc__ = bst.nn.LogSoftmax.__doc__
    __module__ = 'brainscale.nn'


class Dropout(bst.nn.Dropout):
    __doc__ = bst.nn.Dropout.__doc__
    __module__ = 'brainscale.nn'


class Dropout1d(bst.nn.Dropout1d):
    __doc__ = bst.nn.Dropout1d.__doc__
    __module__ = 'brainscale.nn'


class Dropout2d(bst.nn.Dropout2d):
    __doc__ = bst.nn.Dropout2d.__doc__
    __module__ = 'brainscale.nn'


class Dropout3d(bst.nn.Dropout3d):
    __doc__ = bst.nn.Dropout3d.__doc__
    __module__ = 'brainscale.nn'


class AlphaDropout(bst.nn.AlphaDropout):
    __doc__ = bst.nn.AlphaDropout.__doc__
    __module__ = 'brainscale.nn'


class FeatureAlphaDropout(bst.nn.FeatureAlphaDropout):
    __doc__ = bst.nn.FeatureAlphaDropout.__doc__
    __module__ = 'brainscale.nn'


class Identity(bst.nn.Identity):
    __doc__ = bst.nn.Identity.__doc__
    __module__ = 'brainscale.nn'


class SpikeBitwise(bst.nn.SpikeBitwise):
    __doc__ = bst.nn.SpikeBitwise.__doc__
    __module__ = 'brainscale.nn'
