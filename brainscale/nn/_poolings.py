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
    'Flatten', 'Unflatten',

    'AvgPool1d', 'AvgPool2d', 'AvgPool3d',
    'MaxPool1d', 'MaxPool2d', 'MaxPool3d',

    'AdaptiveAvgPool1d', 'AdaptiveAvgPool2d', 'AdaptiveAvgPool3d',
    'AdaptiveMaxPool1d', 'AdaptiveMaxPool2d', 'AdaptiveMaxPool3d',
]


class Flatten(bst.nn.Flatten):
    __doc__ = bst.nn.Flatten.__doc__
    __module__ = 'brainscale.nn'


class Unflatten(bst.nn.Unflatten):
    __doc__ = bst.nn.Unflatten.__doc__
    __module__ = 'brainscale.nn'


class AvgPool1d(bst.nn.AvgPool1d):
    __doc__ = bst.nn.AvgPool1d.__doc__
    __module__ = 'brainscale.nn'


class AvgPool2d(bst.nn.AvgPool2d):
    __doc__ = bst.nn.AvgPool2d.__doc__
    __module__ = 'brainscale.nn'


class AvgPool3d(bst.nn.AvgPool3d):
    __doc__ = bst.nn.AvgPool3d.__doc__
    __module__ = 'brainscale.nn'


class MaxPool1d(bst.nn.MaxPool1d):
    __doc__ = bst.nn.MaxPool1d.__doc__
    __module__ = 'brainscale.nn'


class MaxPool2d(bst.nn.MaxPool2d):
    __doc__ = bst.nn.MaxPool2d.__doc__
    __module__ = 'brainscale.nn'


class MaxPool3d(bst.nn.MaxPool3d):
    __doc__ = bst.nn.MaxPool3d.__doc__
    __module__ = 'brainscale.nn'


class AdaptiveAvgPool1d(bst.nn.AdaptiveAvgPool1d):
    __doc__ = bst.nn.AdaptiveAvgPool1d.__doc__
    __module__ = 'brainscale.nn'


class AdaptiveAvgPool2d(bst.nn.AdaptiveAvgPool2d):
    __doc__ = bst.nn.AdaptiveAvgPool2d.__doc__
    __module__ = 'brainscale.nn'


class AdaptiveAvgPool3d(bst.nn.AdaptiveAvgPool3d):
    __doc__ = bst.nn.AdaptiveAvgPool3d.__doc__
    __module__ = 'brainscale.nn'


class AdaptiveMaxPool1d(bst.nn.AdaptiveMaxPool1d):
    __doc__ = bst.nn.AdaptiveMaxPool1d.__doc__
    __module__ = 'brainscale.nn'


class AdaptiveMaxPool2d(bst.nn.AdaptiveMaxPool2d):
    __doc__ = bst.nn.AdaptiveMaxPool2d.__doc__
    __module__ = 'brainscale.nn'


class AdaptiveMaxPool3d(bst.nn.AdaptiveMaxPool3d):
    __doc__ = bst.nn.AdaptiveMaxPool3d.__doc__
    __module__ = 'brainscale.nn'
