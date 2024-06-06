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

from typing import Callable, Optional

import brainstate as bst
import brainstate.nn as nn
import jax
import jax.numpy as jnp
from brainstate import surrogate, init

from ._etrace_concepts import ETraceVar
from .typing import DTypeLike, ArrayLike, Current, Spike, Size

__all__ = [
  # neuron models
  'IF', 'LIF', 'ALIF',

  # synapse models
  'Expon', 'STP', 'STD',
]


class IF(bst.nn.Neuron):
  """Integrate-and-fire neuron model."""
  __module__ = 'brainscale'

  def __init__(
      self,
      in_size: Size,
      keep_size: bool = False,
      tau: ArrayLike = 5.,
      V_th: ArrayLike = 1.,
      spk_fun: Callable = surrogate.ReluGrad(),
      spk_dtype: DTypeLike = None,
      spk_reset: str = 'soft',
      mode: bst.mixin.Mode = None,
      name: str = None,
  ):
    super().__init__(in_size, keep_size=keep_size, name=name, mode=mode,
                     spk_fun=spk_fun, spk_dtype=spk_dtype, spk_reset=spk_reset)

    # parameters
    self.tau = init.param(tau, self.varshape)
    self.V_th = init.param(V_th, self.varshape)

  def dv(self, v, t, x):
    x = self.sum_current_inputs(v, init=x)
    return (-v + x) / self.tau

  def init_state(self, batch_size: int = None, **kwargs):
    self.V = ETraceVar(init.param(jnp.zeros, self.varshape, batch_size))

  def get_spike(self, V=None):
    V = self.V.value if V is None else V
    v_scaled = (V - self.V_th) / self.V_th
    return self.spk_fun(v_scaled)

  def update(self, x: Current = 0.):
    # reset
    last_V = self.V.value
    last_spike = self.get_spike(self.V.value)
    V_th = self.V_th if self.spk_reset == 'soft' else jax.lax.stop_gradient(last_V)
    V = last_V - V_th * last_spike
    # membrane potential
    V = nn.exp_euler_step(self.dv, V, None, x) + self.sum_delta_inputs()
    self.V.value = V
    return self.get_spike(V)


class LIF(bst.nn.Neuron):
  """Leaky integrate-and-fire neuron model."""
  __module__ = 'brainscale'

  def __init__(
      self,
      in_size: Size,
      keep_size: bool = False,
      tau: ArrayLike = 5.,
      V_th: ArrayLike = 1.,
      V_reset: ArrayLike = 0.,
      V_rest: ArrayLike = 0.,
      spk_fun: Callable = surrogate.ReluGrad(),
      spk_dtype: DTypeLike = None,
      spk_reset: str = 'soft',
      mode: bst.mixin.Mode = None,
      name: str = None,
  ):
    super().__init__(in_size, keep_size=keep_size, name=name, mode=mode, spk_fun=spk_fun,
                     spk_dtype=spk_dtype, spk_reset=spk_reset)

    # parameters
    self.tau = init.param(tau, self.varshape)
    self.V_th = init.param(V_th, self.varshape)
    self.V_rest = init.param(V_rest, self.varshape)
    self.V_reset = init.param(V_reset, self.varshape)

  def dv(self, v, t, x):
    x = self.sum_current_inputs(v, init=x)
    return (-v + self.V_rest + x) / self.tau

  def init_state(self, batch_size: int = None, **kwargs):
    self.V = ETraceVar(init.param(init.Constant(self.V_reset), self.varshape, batch_size))

  def get_spike(self, V=None):
    V = self.V.value if V is None else V
    v_scaled = (V - self.V_th) / self.V_th
    return self.spk_fun(v_scaled)

  def update(self, x: Current = 0.):
    last_v = self.V.value
    lst_spk = self.get_spike(last_v)
    V_th = self.V_th if self.spk_reset == 'soft' else jax.lax.stop_gradient(last_v)
    V = last_v - (V_th - self.V_reset) * lst_spk
    # membrane potential
    V = nn.exp_euler_step(self.dv, V, None, x) + self.sum_delta_inputs()
    self.V.value = V
    return self.get_spike(V)


class ALIF(bst.nn.Neuron):
  """Adaptive Leaky Integrate-and-Fire (LIF) neuron model."""
  __module__ = 'brainscale'

  def __init__(
      self,
      in_size: Size,
      keep_size: bool = False,
      tau: ArrayLike = 5.,
      tau_a: ArrayLike = 100.,
      V_th: ArrayLike = 1.,
      beta: ArrayLike = 0.1,
      spk_fun: Callable = surrogate.ReluGrad(),
      spk_dtype: DTypeLike = None,
      spk_reset: str = 'soft',
      mode: bst.mixin.Mode = None,
      name: str = None,
  ):
    super().__init__(in_size, keep_size=keep_size, name=name, mode=mode, spk_fun=spk_fun,
                     spk_dtype=spk_dtype, spk_reset=spk_reset)

    # parameters
    self.tau = init.param(tau, self.varshape)
    self.tau_a = init.param(tau_a, self.varshape)
    self.V_th = init.param(V_th, self.varshape)
    self.beta = init.param(beta, self.varshape)

  def dv(self, v, t, x):
    x = self.sum_current_inputs(v, init=x)
    return (-v + x) / self.tau

  def da(self, a, t):
    return -a / self.tau_a

  def init_state(self, batch_size: int = None, **kwargs):
    self.V = ETraceVar(init.param(init.Constant(0.), self.varshape, batch_size))
    self.a = ETraceVar(init.param(init.Constant(0.), self.varshape, batch_size))

  def get_spike(self, V=None, a=None):
    V = self.V.value if V is None else V
    a = self.a.value if a is None else a
    v_scaled = (V - self.V_th - self.beta * a) / self.V_th
    return self.spk_fun(v_scaled)

  def update(self, x: Current = 0.):
    last_v = self.V.value
    last_a = self.a.value
    lst_spk = self.get_spike(last_v, last_a)
    V_th = self.V_th if self.spk_reset == 'soft' else jax.lax.stop_gradient(last_v)
    V = last_v - V_th * lst_spk
    a = last_a + lst_spk
    # membrane potential
    V = nn.exp_euler_step(self.dv, V, bst.environ.get('t'), x)
    a = nn.exp_euler_step(self.da, a, bst.environ.get('t'))
    self.V.value = V + self.sum_delta_inputs()
    self.a.value = a
    return self.get_spike(self.V.value, self.a.value)


class Expon(bst.nn.Synapse):
  r"""Exponential decay synapse model.

  Args:
    tau: float. The time constant of decay. [ms]
    %s
  """
  __module__ = 'brainscale'

  def __init__(
      self,
      size: Size,
      keep_size: bool = False,
      name: Optional[str] = None,
      mode: Optional[bst.mixin.Mode] = None,
      tau: ArrayLike = 8.0,
  ):
    super().__init__(name=name,
                     mode=mode,
                     size=size,
                     keep_size=keep_size)

    # parameters
    self.tau = init.param(tau, self.varshape)

  def dg(self, g, t):
    return -g / self.tau

  def init_state(self, batch_size: int = None, **kwargs):
    self.g = ETraceVar(init.param(init.Constant(0.), self.varshape, batch_size))

  def update(self, x: Spike = None):
    self.g.value = nn.exp_euler_step(self.dg, self.g.value, bst.environ.get('t'))
    if x is not None:
      self.align_post_input_add(x)
    return self.g.value

  def align_post_input_add(self, x: Spike):
    self.g.value += x

  def return_info(self):
    return self.g


class STP(bst.nn.Synapse):
  r"""Synaptic output with short-term plasticity.

  %s

  Args:
    tau_f: float, ArrayType, Callable. The time constant of short-term facilitation.
    tau_d: float, ArrayType, Callable. The time constant of short-term depression.
    U: float, ArrayType, Callable. The fraction of resources used per action potential.
    %s
  """
  __module__ = 'brainscale'

  def __init__(
      self,
      size: Size,
      keep_size: bool = False,
      name: Optional[str] = None,
      mode: Optional[bst.mixin.Mode] = None,
      U: ArrayLike = 0.15,
      tau_f: ArrayLike = 1500.,
      tau_d: ArrayLike = 200.,
  ):
    super().__init__(name=name,
                     mode=mode,
                     size=size,
                     keep_size=keep_size)

    # parameters
    self.tau_f = init.param(tau_f, self.varshape)
    self.tau_d = init.param(tau_d, self.varshape)
    self.U = init.param(U, self.varshape)

  def init_state(self, batch_size: int = None, **kwargs):
    self.x = ETraceVar(init.param(init.Constant(1.), self.varshape, batch_size))
    self.u = ETraceVar(init.param(init.Constant(self.U), self.varshape, batch_size))

  def du(self, u, t):
    return self.U - u / self.tau_f

  def dx(self, x, t):
    return (1 - x) / self.tau_d

  def update(self, pre_spike):
    t = bst.environ.get('t')
    u = nn.exp_euler_step(self.du, self.u.value, t)
    x = nn.exp_euler_step(self.dx, self.x.value, t)

    # --- original code:
    #   if pre_spike.dtype == jax.numpy.bool_:
    #     u = bm.where(pre_spike, u + self.U * (1 - self.u), u)
    #     x = bm.where(pre_spike, x - u * self.x, x)
    #   else:
    #     u = pre_spike * (u + self.U * (1 - self.u)) + (1 - pre_spike) * u
    #     x = pre_spike * (x - u * self.x) + (1 - pre_spike) * x

    # --- simplified code:
    u = u + pre_spike * self.U * (1 - self.u.value)
    x = x - pre_spike * u * self.x.value

    self.u.value = u
    self.x.value = x
    return u * x


class STD(bst.nn.Synapse):
  r"""Synaptic output with short-term depression.

  %s

  Args:
    tau: float, ArrayType, Callable. The time constant of recovery of the synaptic vesicles.
    U: float, ArrayType, Callable. The fraction of resources used per action potential.
    %s
  """
  __module__ = 'brainscale'

  def __init__(
      self,
      size: Size,
      keep_size: bool = False,
      name: Optional[str] = None,
      mode: Optional[bst.mixin.Mode] = None,
      # synapse parameters
      tau: ArrayLike = 200.,
      U: ArrayLike = 0.07,
  ):
    super().__init__(name=name,
                     mode=mode,
                     size=size,
                     keep_size=keep_size)

    # parameters
    self.tau = init.param(tau, self.varshape)
    self.U = init.param(U, self.varshape)

  def dx(self, x, t):
    return (1 - x) / self.tau

  def init_state(self, batch_size: int = None, **kwargs):
    self.x = ETraceVar(init.param(init.Constant(1.), self.varshape, batch_size))

  def update(self, pre_spike: Spike):
    t = bst.environ.get('t')
    x = nn.exp_euler_step(self.dx, self.x.value, t)

    # --- original code:
    # self.x.value = bm.where(pre_spike, x - self.U * self.x, x)

    # --- simplified code:
    self.x.value = x - pre_spike * self.U * self.x.value

    return self.x.value

  def return_info(self):
    return self.x
