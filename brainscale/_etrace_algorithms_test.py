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

import braincore as bc
import jax
import jax.numpy as jnp
from braintools import init

import brainscale as nn
from brainscale._errors import NotSupportedError


class IF_Delta_Dense_Layer(bc.Module):
  def __init__(self, n_in, n_rec, tau_mem: nn.typing.ArrayLike = 5., V_th: nn.typing.ArrayLike = 1.,
               spk_reset: str = 'soft', rec_init=init.KaimingNormal(), ff_init=init.KaimingNormal()):
    super().__init__()
    self.neu = nn.IF(n_rec, tau=tau_mem, spk_reset=spk_reset, V_th=V_th)
    w_init = jnp.concat([ff_init([n_in, n_rec]), rec_init([n_rec, n_rec])], axis=0)
    self.syn = nn.Linear(n_in + n_rec, n_rec, w_init=w_init)

  def update(self, spk):
    spk = jnp.concat([spk, self.neu.spike], axis=-1)
    return self.neu(self.syn(spk))


def try_if_delta_etrace_update_On2():
  bc.environ.set(mode=bc.mixin.Training())


  n_in, n_rec = 4, 10
  snn = IF_Delta_Dense_Layer(n_in, n_rec)
  snn = bc.init_states(snn)

  algorithm = nn.DiagOn2Algorithm(snn)
  algorithm.compile_graph(jax.ShapeDtypeStruct((n_in,), bc.environ.dftype()))

  def run_snn(i, inp_spk):
    bc.share.set(i=i, t=i * bc.environ.get_dt())
    out = algorithm.update_model_and_etrace(inp_spk)
    return out

  nt = 100
  inputs = jnp.asarray(bc.random.rand(nt, n_in) < 0.2, dtype=bc.environ.dftype())
  outs = bc.transform.for_loop(run_snn, jnp.arange(nt), inputs)
  print(outs.shape)


class TestDiagOn2(unittest.TestCase):

  def test_if_delta_etrace_update_On2(self):
    for mode in [
      bc.mixin.Batching(),
      bc.mixin.JointMode(bc.mixin.Batching(), bc.mixin.Training()),
    ]:
      bc.environ.set(mode=mode)

      with self.assertRaises(NotSupportedError):
        n_in, n_rec = 4, 10
        snn = IF_Delta_Dense_Layer(n_in, n_rec)
        snn = bc.init_states(snn)
        algorithm = nn.DiagOn2Algorithm(snn)


