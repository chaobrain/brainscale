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
import brainpy
from brainscale._etrace_concepts import ETraceState

__all__ = [
    # synapse models
    'Expon', 'Alpha', 'DualExpon', 'STP', 'STD',
]


class Expon(brainpy.Expon):
    __doc__ = brainpy.Expon.__doc__
    __module__ = 'brainscale.nn'

    def init_state(self, batch_size: int = None, **kwargs):
        self.g = ETraceState(brainstate.init.param(self.g_initializer, self.varshape, batch_size))


class Alpha(brainpy.Alpha):
    __doc__ = brainpy.Alpha.__doc__
    __module__ = 'brainscale.nn'

    def init_state(self, batch_size: int = None, **kwargs):
        self.g = ETraceState(brainstate.init.param(self.g_initializer, self.varshape, batch_size))
        self.h = ETraceState(brainstate.init.param(self.g_initializer, self.varshape, batch_size))


class DualExpon(brainpy.DualExpon):
    __doc__ = brainpy.DualExpon.__doc__
    __module__ = 'brainscale.nn'

    def init_state(self, batch_size: int = None, **kwargs):
        self.g_rise = ETraceState(brainstate.init.param(self.g_initializer, self.varshape, batch_size))
        self.g_decay = ETraceState(brainstate.init.param(self.g_initializer, self.varshape, batch_size))


class STP(brainpy.STP):
    __doc__ = brainpy.STP.__doc__
    __module__ = 'brainscale.nn'

    def init_state(self, batch_size: int = None, **kwargs):
        self.x = ETraceState(brainstate.init.param(brainstate.init.Constant(1.), self.varshape, batch_size))
        self.u = ETraceState(brainstate.init.param(brainstate.init.Constant(self.U), self.varshape, batch_size))


class STD(brainpy.STD):
    __doc__ = brainpy.STD.__doc__
    __module__ = 'brainscale.nn'

    def init_state(self, batch_size: int = None, **kwargs):
        self.x = ETraceState(brainstate.init.param(brainstate.init.Constant(1.), self.varshape, batch_size))
