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


import brainstate

__all__ = [
    # activation functions
    'ReLU', 'RReLU', 'Hardtanh', 'ReLU6', 'Sigmoid', 'Hardsigmoid',
    'Tanh', 'SiLU', 'Mish', 'Hardswish', 'ELU', 'CELU', 'SELU', 'GLU', 'GELU',
    'Hardshrink', 'LeakyReLU', 'LogSigmoid', 'Softplus', 'Softshrink', 'PReLU',
    'Softsign', 'Tanhshrink', 'Softmin', 'Softmax', 'Softmax2d', 'LogSoftmax',

    # dropout
    'Dropout', 'Dropout1d', 'Dropout2d', 'Dropout3d',

    # others
    'Identity', 'SpikeBitwise',
]


class ReLU(brainstate.nn.ReLU):
    __doc__ = brainstate.nn.ReLU.__doc__
    __module__ = 'brainscalon.nn'


class RReLU(brainstate.nn.RReLU):
    __doc__ = brainstate.nn.RReLU.__doc__
    __module__ = 'brainscalon.nn'


class Hardtanh(brainstate.nn.Hardtanh):
    __doc__ = brainstate.nn.Hardtanh.__doc__
    __module__ = 'brainscalon.nn'


class ReLU6(brainstate.nn.ReLU6):
    __doc__ = brainstate.nn.ReLU6.__doc__
    __module__ = 'brainscalon.nn'


class Sigmoid(brainstate.nn.Sigmoid):
    __doc__ = brainstate.nn.Sigmoid.__doc__
    __module__ = 'brainscalon.nn'


class Hardsigmoid(brainstate.nn.Hardsigmoid):
    __doc__ = brainstate.nn.Hardsigmoid.__doc__
    __module__ = 'brainscalon.nn'


class Tanh(brainstate.nn.Tanh):
    __doc__ = brainstate.nn.Tanh.__doc__
    __module__ = 'brainscalon.nn'


class SiLU(brainstate.nn.SiLU):
    __doc__ = brainstate.nn.SiLU.__doc__
    __module__ = 'brainscalon.nn'


class Mish(brainstate.nn.Mish):
    __doc__ = brainstate.nn.Mish.__doc__
    __module__ = 'brainscalon.nn'


class Hardswish(brainstate.nn.Hardswish):
    __doc__ = brainstate.nn.Hardswish.__doc__
    __module__ = 'brainscalon.nn'


class ELU(brainstate.nn.ELU):
    __doc__ = brainstate.nn.ELU.__doc__
    __module__ = 'brainscalon.nn'


class CELU(brainstate.nn.CELU):
    __doc__ = brainstate.nn.CELU.__doc__
    __module__ = 'brainscalon.nn'


class SELU(brainstate.nn.SELU):
    __doc__ = brainstate.nn.SELU.__doc__
    __module__ = 'brainscalon.nn'


class GLU(brainstate.nn.GLU):
    __doc__ = brainstate.nn.GLU.__doc__
    __module__ = 'brainscalon.nn'


class GELU(brainstate.nn.GELU):
    __doc__ = brainstate.nn.GELU.__doc__
    __module__ = 'brainscalon.nn'


class Hardshrink(brainstate.nn.Hardshrink):
    __doc__ = brainstate.nn.Hardshrink.__doc__
    __module__ = 'brainscalon.nn'


class LeakyReLU(brainstate.nn.LeakyReLU):
    __doc__ = brainstate.nn.LeakyReLU.__doc__
    __module__ = 'brainscalon.nn'


class LogSigmoid(brainstate.nn.LogSigmoid):
    __doc__ = brainstate.nn.LogSigmoid.__doc__
    __module__ = 'brainscalon.nn'


class Softplus(brainstate.nn.Softplus):
    __doc__ = brainstate.nn.Softplus.__doc__
    __module__ = 'brainscalon.nn'


class Softshrink(brainstate.nn.Softshrink):
    __doc__ = brainstate.nn.Softshrink.__doc__
    __module__ = 'brainscalon.nn'


class PReLU(brainstate.nn.PReLU):
    __doc__ = brainstate.nn.PReLU.__doc__
    __module__ = 'brainscalon.nn'


class Softsign(brainstate.nn.Softsign):
    __doc__ = brainstate.nn.Softsign.__doc__
    __module__ = 'brainscalon.nn'


class Tanhshrink(brainstate.nn.Tanhshrink):
    __doc__ = brainstate.nn.Tanhshrink.__doc__
    __module__ = 'brainscalon.nn'


class Softmin(brainstate.nn.Softmin):
    __doc__ = brainstate.nn.Softmin.__doc__
    __module__ = 'brainscalon.nn'


class Softmax(brainstate.nn.Softmax):
    __doc__ = brainstate.nn.Softmax.__doc__
    __module__ = 'brainscalon.nn'


class Softmax2d(brainstate.nn.Softmax2d):
    __doc__ = brainstate.nn.Softmax2d.__doc__
    __module__ = 'brainscalon.nn'


class LogSoftmax(brainstate.nn.LogSoftmax):
    __doc__ = brainstate.nn.LogSoftmax.__doc__
    __module__ = 'brainscalon.nn'


class Dropout(brainstate.nn.Dropout):
    __doc__ = brainstate.nn.Dropout.__doc__
    __module__ = 'brainscalon.nn'


class Dropout1d(brainstate.nn.Dropout1d):
    __doc__ = brainstate.nn.Dropout1d.__doc__
    __module__ = 'brainscalon.nn'


class Dropout2d(brainstate.nn.Dropout2d):
    __doc__ = brainstate.nn.Dropout2d.__doc__
    __module__ = 'brainscalon.nn'


class Dropout3d(brainstate.nn.Dropout3d):
    __doc__ = brainstate.nn.Dropout3d.__doc__
    __module__ = 'brainscalon.nn'


class Identity(brainstate.nn.Identity):
    __doc__ = brainstate.nn.Identity.__doc__
    __module__ = 'brainscalon.nn'


class SpikeBitwise(brainstate.nn.SpikeBitwise):
    __doc__ = brainstate.nn.SpikeBitwise.__doc__
    __module__ = 'brainscalon.nn'
