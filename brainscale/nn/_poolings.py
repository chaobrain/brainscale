# Copyright 2024 BrainX Ecosystem Limited. All Rights Reserved.
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
    'Flatten', 'Unflatten',

    'AvgPool1d', 'AvgPool2d', 'AvgPool3d',
    'MaxPool1d', 'MaxPool2d', 'MaxPool3d',

    'AdaptiveAvgPool1d', 'AdaptiveAvgPool2d', 'AdaptiveAvgPool3d',
    'AdaptiveMaxPool1d', 'AdaptiveMaxPool2d', 'AdaptiveMaxPool3d',
]


class Flatten(brainstate.nn.Flatten):
    __doc__ = brainstate.nn.Flatten.__doc__
    __module__ = 'brainscale.nn'


class Unflatten(brainstate.nn.Unflatten):
    __doc__ = brainstate.nn.Unflatten.__doc__
    __module__ = 'brainscale.nn'


class AvgPool1d(brainstate.nn.AvgPool1d):
    __doc__ = brainstate.nn.AvgPool1d.__doc__
    __module__ = 'brainscale.nn'


class AvgPool2d(brainstate.nn.AvgPool2d):
    __doc__ = brainstate.nn.AvgPool2d.__doc__
    __module__ = 'brainscale.nn'


class AvgPool3d(brainstate.nn.AvgPool3d):
    __doc__ = brainstate.nn.AvgPool3d.__doc__
    __module__ = 'brainscale.nn'


class MaxPool1d(brainstate.nn.MaxPool1d):
    __doc__ = brainstate.nn.MaxPool1d.__doc__
    __module__ = 'brainscale.nn'


class MaxPool2d(brainstate.nn.MaxPool2d):
    __doc__ = brainstate.nn.MaxPool2d.__doc__
    __module__ = 'brainscale.nn'


class MaxPool3d(brainstate.nn.MaxPool3d):
    __doc__ = brainstate.nn.MaxPool3d.__doc__
    __module__ = 'brainscale.nn'


class AdaptiveAvgPool1d(brainstate.nn.AdaptiveAvgPool1d):
    __doc__ = brainstate.nn.AdaptiveAvgPool1d.__doc__
    __module__ = 'brainscale.nn'


class AdaptiveAvgPool2d(brainstate.nn.AdaptiveAvgPool2d):
    __doc__ = brainstate.nn.AdaptiveAvgPool2d.__doc__
    __module__ = 'brainscale.nn'


class AdaptiveAvgPool3d(brainstate.nn.AdaptiveAvgPool3d):
    __doc__ = brainstate.nn.AdaptiveAvgPool3d.__doc__
    __module__ = 'brainscale.nn'


class AdaptiveMaxPool1d(brainstate.nn.AdaptiveMaxPool1d):
    __doc__ = brainstate.nn.AdaptiveMaxPool1d.__doc__
    __module__ = 'brainscale.nn'


class AdaptiveMaxPool2d(brainstate.nn.AdaptiveMaxPool2d):
    __doc__ = brainstate.nn.AdaptiveMaxPool2d.__doc__
    __module__ = 'brainscale.nn'


class AdaptiveMaxPool3d(brainstate.nn.AdaptiveMaxPool3d):
    __doc__ = brainstate.nn.AdaptiveMaxPool3d.__doc__
    __module__ = 'brainscale.nn'
