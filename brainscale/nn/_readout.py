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

import numbers
from typing import Callable, Optional

import brainstate as bst
import brainunit as u
import jax
import jax.numpy as jnp
from brainstate import init, surrogate, nn

from brainscale._etrace_concepts import ETraceParamOp, ETraceState
from brainscale._etrace_operators import MatMulETraceOp
from brainscale._typing import Size, ArrayLike, Spike

__all__ = [
    'LeakyRateReadout',
    'LeakySpikeReadout',
]


class LeakyRateReadout(nn.Module):
    """
    Leaky dynamics for the read-out module used in the Real-Time Recurrent Learning.
    """
    __module__ = 'brainscale.nn'

    def __init__(
        self,
        in_size: Size,
        out_size: Size,
        tau: ArrayLike = 5. * u.ms,
        w_init: Callable = init.KaimingNormal(),
        r_initializer: Callable = init.ZeroInit(),
        name: Optional[str] = None,
    ):
        super().__init__(name=name)

        # parameters
        self.in_size = (in_size,) if isinstance(in_size, numbers.Integral) else tuple(in_size)
        self.out_size = (out_size,) if isinstance(out_size, numbers.Integral) else tuple(out_size)
        self.tau = init.param(tau, self.in_size)
        self.decay = jnp.exp(-bst.environ.get_dt() / self.tau)
        self.r_initializer = r_initializer

        # weights
        weight = init.param(w_init, (self.in_size[0], self.out_size[0]))
        self.weight_op = ETraceParamOp(weight, MatMulETraceOp())

    def init_state(self, batch_size=None, **kwargs):
        self.r = ETraceState(init.param(self.r_initializer, self.out_size, batch_size))

    def reset_state(self, batch_size=None, **kwargs):
        self.r.value = init.param(self.r_initializer, self.out_size, batch_size)

    def update(self, x):
        r = self.decay * self.r.value + self.weight_op.execute(x)
        self.r.value = r
        return r


class LeakySpikeReadout(nn.LeakySpikeReadout):
    """Integrate-and-fire neuron model."""

    __module__ = 'brainscale.nn'

    def __init__(
        self,
        in_size: Size,
        keep_size: bool = False,
        tau: ArrayLike = 5. * u.ms,
        V_th: ArrayLike = 1. * u.mV,
        w_init: Callable = init.KaimingNormal(unit=u.mV),
        V_initializer: Callable = init.ZeroInit(unit=u.mV),
        spk_fun: Callable = surrogate.ReluGrad(),
        spk_reset: str = 'soft',
        name: str = None,
    ):
        super().__init__(in_size, name=name,  spk_fun=spk_fun, spk_reset=spk_reset)

        # parameters
        self.tau = init.param(tau, self.varshape)
        self.V_th = init.param(V_th, self.varshape)
        self.V_initializer = V_initializer

        # weights
        weight = init.param(w_init, (self.in_size[0], self.out_size[0]))
        self.weight_op = ETraceParamOp(weight, MatMulETraceOp())

    def init_state(self, batch_size, **kwargs):
        self.V = ETraceState(init.param(self.V_initializer, self.varshape, batch_size))

    def reset_state(self, batch_size, **kwargs):
        self.V.value = init.param(self.V_initializer, self.varshape, batch_size)

    @property
    def spike(self):
        return self.get_spike(self.V.value)

    def get_spike(self, V):
        v_scaled = (V - self.V_th) / self.V_th
        return self.spk_fun(v_scaled)

    def update(self, spike: Spike) -> Spike:
        # reset
        last_V = self.V.value
        last_spike = self.get_spike(last_V)
        V_th = self.V_th if self.spk_reset == 'soft' else jax.lax.stop_gradient(last_V)
        V = last_V - V_th * last_spike
        # membrane potential
        dv = lambda v, x: (-v + self.sum_current_inputs(x, v)) / self.tau
        V = nn.exp_euler_step(dv, V, self.weight_op.execute(spike))
        V = self.sum_delta_inputs(V)
        self.V.value = V
        return self.get_spike(V)
