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


import unittest
from functools import partial, reduce
from typing import Callable

import braincore as bc
import jax
import jax.numpy as jnp
from braintools import init

import brainscale as nn


class _IF_Delta_Dense_Layer(bc.Module):
  def __init__(
      self,
      n_in,
      n_rec,
      tau_mem: nn.typing.ArrayLike = 5.,
      V_th: nn.typing.ArrayLike = 1.,
      spk_reset: str = 'soft',
      rec_init=init.KaimingNormal(),
      ff_init=init.KaimingNormal()
  ):
    super().__init__()
    self.neu = nn.IF(n_rec, tau=tau_mem, spk_reset=spk_reset, V_th=V_th)
    w_init = jnp.concat([ff_init([n_in, n_rec]), rec_init([n_rec, n_rec])], axis=0)
    self.syn = nn.Linear(n_in + n_rec, n_rec, w_init=w_init)

  def update(self, spk):
    spk = jnp.concat([spk, self.neu.spike], axis=-1)
    return self.neu(self.syn(spk))


class TestDiagOn2(unittest.TestCase):

  def test_if_delta_etrace_update_On2(self):
    for mode in [
      bc.mixin.Batching(),
      bc.mixin.Training(),
      bc.mixin.JointMode(bc.mixin.Batching(), bc.mixin.Training()),
    ]:
      bc.environ.set(mode=mode)

      n_in, n_rec = 4, 10
      snn = _IF_Delta_Dense_Layer(n_in, n_rec)
      snn = bc.init_states(snn)
      algorithm = nn.DiagOn2Algorithm(snn)

  def test_non_batched_On2_algorithm(self):
    bc.environ.set(mode=bc.mixin.Training())

    n_in, n_rec = 4, 10
    snn = _IF_Delta_Dense_Layer(n_in, n_rec)
    snn = bc.init_states(snn)

    algorithm = nn.DiagOn2Algorithm(snn)
    algorithm.compile_graph(jax.ShapeDtypeStruct((n_in,), bc.environ.dftype()))

    def run_snn(i, inp_spk):
      with bc.environ.context(i=i, t=i * bc.environ.get_dt()):
        out = algorithm.update_model_and_etrace(inp_spk)
      return out

    nt = 100
    inputs = jnp.asarray(bc.random.rand(nt, n_in) < 0.2, dtype=bc.environ.dftype())
    outs = bc.transform.for_loop(run_snn, jnp.arange(nt), inputs)
    print(outs.shape)
    self.assertEqual(outs.shape, (nt, n_rec))

  def test_batched_On2_algorithm(self):
    bc.environ.set(mode=bc.mixin.JointMode(bc.mixin.Training(),
                                           bc.mixin.Batching()))

    n_batch = 16
    n_in, n_rec = 4, 10
    snn = _IF_Delta_Dense_Layer(n_in, n_rec)
    snn = bc.init_states(snn, n_batch)

    algorithm = nn.DiagOn2Algorithm(snn)
    algorithm.compile_graph(jax.ShapeDtypeStruct((n_batch, n_in), bc.environ.dftype()))

    def run_snn(i, inp_spk):
      with bc.environ.context(i=i, t=i * bc.environ.get_dt()):
        out = algorithm.update_model_and_etrace(inp_spk)
      return out

    nt = 100
    inputs = jnp.asarray(bc.random.rand(nt, n_batch, n_in) < 0.2, dtype=bc.environ.dftype())
    outs = bc.transform.for_loop(run_snn, jnp.arange(nt), inputs)
    print(outs.shape)
    self.assertEqual(outs.shape, (nt, n_batch, n_rec))


class _LIF_STDExpCu_Dense_Layer(bc.Module):
  """
  The RTRL layer with LIF neurons and dense connected STD-based exponential current synapses.
  """

  def __init__(
      self, n_in, n_rec, inp_std=False, tau_mem=5., tau_syn=10., V_th=1., tau_std=500.,
      spk_fun: Callable = bc.surrogate.ReluGrad(),
      spk_reset: str = 'soft',
      rec_init: Callable = init.KaimingNormal(),
      ff_init: Callable = init.KaimingNormal()
  ):
    super().__init__()
    self.neu = nn.LIF(n_rec, tau=tau_mem, spk_fun=spk_fun, spk_reset=spk_reset, V_th=V_th)
    self.std = nn.STD(n_rec, tau=tau_std, U=0.1)
    if inp_std:
      self.std_inp = nn.STD(n_in, tau=tau_std, U=0.1)
    else:
      self.std_inp = None

    self.syn = bc.HalfProjAlignPostMg(
      comm=nn.Linear(n_in + n_rec, n_rec, jnp.concat([ff_init([n_in, n_rec]), rec_init([n_rec, n_rec])], axis=0)),
      syn=nn.Expon.delayed(size=n_rec, tau=tau_syn),
      out=nn.CUBA.delayed(),
      post=self.neu
    )

  def update(self, inp_spk):
    if self.std_inp is not None:
      inp_spk = self.std_inp(inp_spk) * inp_spk
    last_spk = self.neu.spike
    inp = jnp.concat([inp_spk, last_spk * self.std(last_spk)], axis=-1)
    self.syn(inp)
    self.neu()
    return self.neu.spike


class _LIF_STPExpCu_Dense_Layer(bc.Module):
  def __init__(
      self,
      n_in, n_rec, inp_stp=False, tau_mem=5., tau_syn=10., V_th=1.,
      spk_fun: Callable = bc.surrogate.ReluGrad(),
      spk_reset: str = 'soft',
      rec_init: Callable = init.KaimingNormal(),
      ff_init: Callable = init.KaimingNormal()
  ):
    super().__init__()
    self.inp_stp = inp_stp
    self.neu = nn.LIF(n_rec, tau=tau_mem, spk_fun=spk_fun, spk_reset=spk_reset, V_th=V_th)
    self.stp = nn.STP(n_rec, tau_f=500., tau_d=100.)
    if inp_stp:
      self.stp_inp = nn.STP(n_in, tau_f=500., tau_d=100.)

    self.syn = bc.HalfProjAlignPostMg(
      comm=nn.Linear(n_in + n_rec, n_rec, jnp.concat([ff_init([n_in, n_rec]), rec_init([n_rec, n_rec])])),
      syn=nn.Expon.delayed(size=n_rec, tau=tau_syn),
      out=nn.CUBA.delayed(),
      post=self.neu
    )

  def update(self, inp_spk):
    if self.inp_stp:
      inp_spk = self.stp_inp(inp_spk) * inp_spk
    last_spk = self.neu.spike
    inp = jnp.concat([inp_spk, last_spk * self.stp(last_spk)], axis=-1)
    self.syn(inp)
    self.neu()
    return self.neu.spike


class TestDiagGrad(unittest.TestCase):
  def _hidden_to_hidden(self, model, other_vals, pre_hidden_vals, inputs, out_hidden: str = None):
    etrace_vars, other_states = model.states().split(nn.ETraceVar)
    etrace_vars.assign_values(pre_hidden_vals)
    other_states.assign_values(other_vals)
    with nn.stop_param_gradients():
      model(inputs)
    if out_hidden is not None:
      return etrace_vars[out_hidden].value
    else:
      return etrace_vars.to_dict_values()

  def _whether_collective_and_independent_are_same(self, model: bc.Module, inputs):
    states = model.states()
    state_vals = states.to_dict_values()
    etrace_vars, other_states = states.split(nn.ETraceVar)
    etrace_grads = jax.tree.map(bc.random.randn_like, etrace_vars.to_dict_values())

    # collective solve
    fun = partial(self._hidden_to_hidden, model, other_states.to_dict_values(), inputs=inputs)
    _, new_grads1 = jax.jvp(fun, (etrace_vars.to_dict_values(),), (etrace_grads,))
    states.assign_values(state_vals)

    # independent solve
    new_grads2 = dict()
    for key in etrace_vars:
      fun = partial(self._hidden_to_hidden, model, other_states.to_dict_values(), inputs=inputs, out_hidden=key)
      jac = bc.transform.vector_grad(fun, argnums=0)(etrace_vars.to_dict_values())
      states.assign_values(state_vals)

      leaves = jax.tree.leaves(jax.tree.map(jnp.multiply, jac, etrace_grads))
      grad = reduce(jnp.add, leaves)
      new_grads2[key] = grad
      self.assertTrue(jnp.allclose(grad, new_grads1[key], atol=1e-5), f"key: {key}")

  def test1(self):
    bc.environ.set(mode=bc.mixin.JointMode(bc.mixin.Training(), bc.mixin.Batching()))
    bc.environ.set(i=0, t=0.)

    n_batch, n_in, n_rec = 16, 4, 10
    std_model = _LIF_STDExpCu_Dense_Layer(n_in, n_rec, inp_std=False)
    stp_model = _LIF_STPExpCu_Dense_Layer(n_in, n_rec, inp_stp=False)

    bc.init_states(std_model, n_batch)
    std_model.neu.V.value = bc.random.uniform(-1., 1.5, std_model.neu.V.value.shape)
    bc.init_states(stp_model, n_batch)
    stp_model.neu.V.value = bc.random.uniform(-1., 1.5, stp_model.neu.V.value.shape)

    inputs = jnp.asarray(bc.random.rand(n_batch, n_in) < 0.2, dtype=bc.environ.dftype())

    self._whether_collective_and_independent_are_same(std_model, inputs)
    self._whether_collective_and_independent_are_same(stp_model, inputs)


