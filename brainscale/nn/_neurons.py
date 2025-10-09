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

import brainpy
from braintools import init

from brainscale._etrace_concepts import ETraceState

__all__ = [
    # neuron models
    'IF', 'LIF', 'ALIF',
]


# IF = brainpy.IF
# LIF = brainpy.LIF
# ALIF = brainpy.ALIF


class IF(brainpy.IF):
    __module__ = 'brainscale.nn'
    __doc__ = brainpy.IF.__doc__

    def init_state(self, batch_size: int = None, **kwargs):
        self.V = ETraceState(init.param(self.V_initializer, self.varshape, batch_size))


class LIF(brainpy.LIF):
    __module__ = 'brainscale.nn'
    __doc__ = brainpy.LIF.__doc__

    def init_state(self, batch_size: int = None, **kwargs):
        self.V = ETraceState(init.param(self.V_initializer, self.varshape, batch_size))


class ALIF(brainpy.ALIF):
    __doc__ = brainpy.ALIF.__doc__
    __module__ = 'brainscale.nn'

    def init_state(self, batch_size: int = None, **kwargs):
        self.V = ETraceState(init.param(self.V_initializer, self.varshape, batch_size))
        self.a = ETraceState(init.param(self.a_initializer, self.varshape, batch_size))
