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
from brainstate import init

__all__ = [
    # neuron models
    'IF', 'LIF', 'ALIF',
]


class IF(bst.nn.IF):
    __module__ = 'brainscale.nn'
    __doc__ = bst.nn.IF.__doc__

    def init_state(self, batch_size: int = None, **kwargs):
        self.V = ETraceState(init.param(self.V_initializer, self.varshape, batch_size))


class LIF(bst.nn.LIF):
    __module__ = 'brainscale.nn'
    __doc__ = bst.nn.LIF.__doc__

    def init_state(self, batch_size: int = None, **kwargs):
        self.V = ETraceState(init.param(self.V_initializer, self.varshape, batch_size))


class ALIF(bst.nn.ALIF):
    __doc__ = bst.nn.ALIF.__doc__
    __module__ = 'brainscale.nn'

    def init_state(self, batch_size: int = None, **kwargs):
        self.V = ETraceState(init.param(self.V_initializer, self.varshape, batch_size))
        self.a = ETraceState(init.param(self.a_initializer, self.varshape, batch_size))
