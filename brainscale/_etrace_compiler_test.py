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

from __future__ import annotations

import numbers
import unittest
from typing import Callable

import braincore as bc
import braintools as bts
import jax.numpy as jnp

import brainscale as nn
from brainscale._etrace_compiler import CompilationError, NotSupportedError


class IF_Delta_Dense_Layer(bc.Module):
  """
  The RTRL layer with IF neurons and dense connected delta synapses.
  """

  def __init__(
      self,
      linear_cls,
      n_in: int,
      n_rec: int,
      tau_mem: bc.typing.ArrayLike = 5.,
      V_th: bc.typing.ArrayLike = 1.,
      spk_fun: Callable = bc.surrogate.ReluGrad(),
      spk_reset: str = 'soft',
      init: Callable = bts.init.KaimingNormal(),
  ):
    super().__init__()
    self.neu = nn.IF(n_rec, tau=tau_mem, spk_fun=spk_fun, spk_reset=spk_reset, V_th=V_th)
    self.syn = linear_cls(n_in + n_rec, n_rec, w_init=init)

  def update(self, spk):
    spk = jnp.concat([spk, self.neu.spike], axis=-1)
    return self.neu(self.syn(spk))


class TestCompiler(unittest.TestCase):
  def test_no_weight(self):
    class Linear(bc.Module):
      def __init__(self, in_size: bc.typing.Size, out_size: bc.typing.Size,
                   w_init: Callable = bts.init.KaimingNormal()):
        super().__init__()
        self.in_size = (in_size,) if isinstance(in_size, numbers.Integral) else tuple(in_size)
        self.out_size = (out_size,) if isinstance(out_size, numbers.Integral) else tuple(out_size)
        params = dict(weight=bts.init.parameter(w_init, self.in_size + self.out_size, allow_none=False))
        self.weight = nn.ETraceParam(params)
        self.op = nn.ETraceOp(self._operation)

      def update(self, x):
        # test that no EtraceWeight is used in the etrace operator
        return self.op(x, None)

      def _operation(self, x, params):
        return x @ jnp.ones(self.in_size + self.out_size)

    model = bc.init_states(IF_Delta_Dense_Layer(Linear, 10, 20), 16)
    inp_spk = jnp.asarray(bc.random.rand(16, 10) < 0.3, dtype=bc.environ.dftype())
    graph = nn.ETraceGraph(model)

    with self.assertRaises(CompilationError):
      graph.compile_graph(inp_spk)

    bc.util.clear_buffer_memory()

  def test_multi_weight(self):
    class Linear(bc.Module):
      def __init__(self, in_size: bc.typing.Size, out_size: bc.typing.Size, w_init=bts.init.KaimingNormal()):
        super().__init__()
        self.in_size = (in_size,) if isinstance(in_size, numbers.Integral) else tuple(in_size)
        self.out_size = (out_size,) if isinstance(out_size, numbers.Integral) else tuple(out_size)
        params = dict(weight=bts.init.parameter(w_init, self.in_size + self.out_size, allow_none=False))
        self.weight1 = nn.ETraceParam(params)
        params = dict(weight=bts.init.parameter(w_init, self.in_size + self.out_size, allow_none=False))
        self.weight2 = nn.ETraceParam(params)
        self.op = nn.ETraceOp(self._operation)

      def update(self, x):
        # Test that multiple EtraceWeights are not supported
        return self.op(x, (self.weight1.value, self.weight2.value))

      def _operation(self, x, params):
        return x @ jnp.ones(self.in_size + self.out_size)

    model = bc.init_states(IF_Delta_Dense_Layer(Linear, 10, 20), 16)
    inp_spk = jnp.asarray(bc.random.rand(16, 10) < 0.3, dtype=bc.environ.dftype())
    graph = nn.ETraceGraph(model)

    with self.assertRaises(CompilationError):
      graph.compile_graph(inp_spk)

    bc.util.clear_buffer_memory()

  def test_op_multi_return(self):
    class Linear(bc.Module):
      def __init__(self, in_size: bc.typing.Size, out_size: bc.typing.Size,
                   w_init: Callable = bts.init.KaimingNormal()):
        super().__init__()
        self.in_size = (in_size,) if isinstance(in_size, numbers.Integral) else tuple(in_size)
        self.out_size = (out_size,) if isinstance(out_size, numbers.Integral) else tuple(out_size)
        params = dict(weight=bts.init.parameter(w_init, self.in_size + self.out_size, allow_none=False))
        self.weight1 = nn.ETraceParam(params)
        self.op = nn.ETraceOp(self._operation)

      def update(self, x):
        res = self.op(x, self.weight1.value)
        return res[0] + res[1]

      def _operation(self, x, params):
        y1 = x @ jnp.ones(self.in_size + self.out_size)
        y2 = x @ params['weight']
        return y1, y2

    model = IF_Delta_Dense_Layer(Linear, 10, 20)
    bc.init_states(model, 16)

    inp_spk = jnp.asarray(bc.random.rand(16, 10) < 0.3, dtype=bc.environ.dftype())
    graph = nn.ETraceGraph(model)

    with self.assertRaises(NotSupportedError):
      graph.compile_graph(inp_spk)

    bc.util.clear_buffer_memory()

  def test_etrace_op_jaxpr(self):
    class Linear1(bc.Module):
      def __init__(self, in_size: int, out_size: int, w_init: Callable = bts.init.KaimingNormal()):
        super().__init__()
        self.in_size = (in_size,)
        self.out_size = (out_size,)
        params = dict(weight=bts.init.parameter(w_init, self.in_size + self.out_size, allow_none=False))
        self.weight = nn.ETraceParam(params)
        self.mask = bc.random.random(self.in_size + self.out_size) < 0.5
        self.op = nn.ETraceOp(self._operation)

      def update(self, x):
        res = self.op(x, self.weight.value)
        return res

      def _operation(self, x, params):
        return x @ (params['weight'] * self.mask)

    model = IF_Delta_Dense_Layer(Linear1, 10, 20)
    bc.init_states(model, 16)

    inp_spk = jnp.asarray(bc.random.rand(16, 10) < 0.3, dtype=bc.environ.dftype())
    graph = nn.ETraceGraph(model)
    graph.compile_graph(inp_spk)

    bc.util.clear_buffer_memory()

  def test_etrace_and_non_etrace_weight(self):
    bc.environ.set(mode=bc.mixin.JointMode(bc.mixin.Batching(), bc.mixin.Training()))

    class Model(bc.Module):
      def __init__(self):
        super().__init__()

        self.linear1 = nn.Linear(10, 30, as_etrace_weight=True)
        self.linear2 = nn.Linear(10, 30, as_etrace_weight=False)
        self.lif = nn.LIF(30)

      def update(self, x):
        y1 = self.linear1(x)
        y2 = self.linear2(x)
        return self.lif(y1 + y2)

    # inputs
    inp_spk = jnp.asarray(bc.random.rand(16, 10) < 0.3, dtype=bc.environ.dftype())

    # model
    model = Model()
    bc.init_states(model, 16)

    # algorithms
    algorithm = nn.DiagExpSmOnAlgorithm(model, num_rank=100)
    algorithm.compile_graph(inp_spk)
    out = algorithm(inp_spk)

    weights = model.states().subset(bc.ParamState)
    grads = bc.transform.grad(lambda x: algorithm(x).sum(), grad_vars=weights)(inp_spk)

    # print(out)
    print(grads)

  def test_weight_weight_hidden_structure(self):
    pass
