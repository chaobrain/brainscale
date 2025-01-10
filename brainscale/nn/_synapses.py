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

from brainscale._etrace_concepts import ETraceState

__all__ = [
    # synapse models
    'Expon', 'Alpha', 'DualExpon', 'STP', 'STD',
]


class Expon(bst.nn.Expon):
    __doc__ = bst.nn.Expon.__doc__
    __module__ = 'brainscale.nn'

    def init_state(self, batch_size: int = None, **kwargs):
        self.g = ETraceState(bst.init.param(self.g_initializer, self.varshape, batch_size))


class Alpha(bst.nn.Alpha):
    __doc__ = bst.nn.Alpha.__doc__
    __module__ = 'brainscale.nn'

    def init_state(self, batch_size: int = None, **kwargs):
        self.g = ETraceState(bst.init.param(self.g_initializer, self.varshape, batch_size))
        self.h = ETraceState(bst.init.param(self.g_initializer, self.varshape, batch_size))


class DualExpon(bst.nn.DualExpon):
    __doc__ = bst.nn.DualExpon.__doc__
    __module__ = 'brainscale.nn'

    def init_state(self, batch_size: int = None, **kwargs):
        self.g_rise = ETraceState(bst.init.param(self.g_initializer, self.varshape, batch_size))
        self.g_decay = ETraceState(bst.init.param(self.g_initializer, self.varshape, batch_size))


class STP(bst.nn.STP):
    __doc__ = bst.nn.STP.__doc__
    __module__ = 'brainscale.nn'

    def init_state(self, batch_size: int = None, **kwargs):
        self.x = ETraceState(bst.init.param(bst.init.Constant(1.), self.varshape, batch_size))
        self.u = ETraceState(bst.init.param(bst.init.Constant(self.U), self.varshape, batch_size))


class STD(bst.nn.STD):
    __doc__ = bst.nn.STD.__doc__
    __module__ = 'brainscale.nn'

    def init_state(self, batch_size: int = None, **kwargs):
        self.x = ETraceState(bst.init.param(bst.init.Constant(1.), self.varshape, batch_size))
