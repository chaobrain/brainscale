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

import brainstate
from brainstate import init

from brainscalon._etrace_concepts import ETraceState

__all__ = [
    # neuron models
    'IF', 'LIF', 'ALIF',
]


class IF(brainstate.nn.IF):
    __module__ = 'brainscalon.nn'
    __doc__ = brainstate.nn.IF.__doc__

    def init_state(self, batch_size: int = None, **kwargs):
        self.V = ETraceState(init.param(self.V_initializer, self.varshape, batch_size))


class LIF(brainstate.nn.LIF):
    __module__ = 'brainscalon.nn'
    __doc__ = brainstate.nn.LIF.__doc__

    def init_state(self, batch_size: int = None, **kwargs):
        self.V = ETraceState(init.param(self.V_initializer, self.varshape, batch_size))


class ALIF(brainstate.nn.ALIF):
    __doc__ = brainstate.nn.ALIF.__doc__
    __module__ = 'brainscalon.nn'

    def init_state(self, batch_size: int = None, **kwargs):
        self.V = ETraceState(init.param(self.V_initializer, self.varshape, batch_size))
        self.a = ETraceState(init.param(self.a_initializer, self.varshape, batch_size))
