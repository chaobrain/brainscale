import os
from typing import Callable

import numpy as np

os.environ['JAX_TRACEBACK_FILTERING'] = 'off'

import jax

import braincore as bc
import jax.numpy as jnp
from braintools import init

import brainscale as nn


class ALIF_ExpCu_Dense_Layer(bc.Module):
  def __init__(
      self, n_in, n_rec, n_out, tau_mem=5., tau_syn=10., V_th=1., tau_a=100., beta=0.1, tau_o=10.,
      spk_fun: Callable = bc.surrogate.ReluGrad(),
      spk_reset: str = 'soft',
      rec_init: Callable = init.KaimingNormal(),
      ff_init: Callable = init.KaimingNormal()
  ):
    super().__init__()
    self.neu = nn.ALIF(
      n_rec, tau=tau_mem, tau_a=tau_a, beta=beta, spk_fun=spk_fun, spk_reset=spk_reset, V_th=V_th
    )
    self.syn = bc.HalfProjAlignPostMg(
      comm=nn.Linear(n_in + n_rec, n_rec, jnp.concat([ff_init([n_in, n_rec]), rec_init([n_rec, n_rec])], axis=0)),
      syn=nn.Expon.delayed(size=n_rec, tau=tau_syn),
      out=nn.CUBA.delayed(),
      post=self.neu
    )
    self.out = nn.LeakyRateReadout(
      in_size=n_rec,
      out_size=n_out,
      tau=tau_o,
      w_init=init.KaimingNormal()
    )

  def update(self, spk):
    self.syn(jnp.concat([spk, self.neu.spike], axis=-1))
    self.neu()
    return self.out(self.neu.spike)


class IF_Delta_Dense_Layer(bc.Module):
  """
  The RTRL layer with IF neurons and dense connected delta synapses.
  """

  def __init__(
      self,
      n_in, n_rec,
      tau_mem: nn.typing.ArrayLike = 5.,
      V_th: nn.typing.ArrayLike = 1.,
      spk_fun: Callable = bc.surrogate.ReluGrad(),
      spk_reset: str = 'soft',
      rec_init: Callable = init.KaimingNormal(),
      ff_init: Callable = init.KaimingNormal()
  ):
    super().__init__()
    self.neu = nn.IF(n_rec, tau=tau_mem, spk_fun=spk_fun, spk_reset=spk_reset, V_th=V_th)
    w_init = jnp.concat([ff_init([n_in, n_rec]), rec_init([n_rec, n_rec])], axis=0)
    self.syn = nn.Linear(n_in + n_rec, n_rec, w_init=w_init)

  def update(self, spk):
    spk = jnp.concat([spk, self.neu.spike], axis=-1)
    return self.neu(self.syn(spk))


def try_alif_etrace_single_step():
  bc.environ.set(mode=bc.mixin.JointMode(bc.mixin.Batching(), bc.mixin.Training()))

  n_in = 40
  n_rec = 100
  n_out = 10
  n_batch = 16

  inp_spk = bc.random.rand(n_batch, n_in) < 0.8
  snn = ALIF_ExpCu_Dense_Layer(n_in, n_rec, n_out)
  snn = bc.init_states(snn, n_batch)

  out = snn(inp_spk)

  algorithm = nn.DiagExpSmOnAlgorithm(snn, decay=0.99)
  algorithm.compile_graph(inp_spk)
  out = algorithm(inp_spk, 0)

  weights = snn.states().subset(nn.ETraceParam)
  grads = bc.transform.grad(algorithm.update_model_and_etrace, grad_vars=weights)(inp_spk, 0)

  print(out)
  print(grads)


def try_if_delta_etrace_update():
  n_batch, n_in, n_rec = 16, 4, 10
  bc.environ.set(mode=bc.mixin.JointMode(bc.mixin.Batching(), bc.mixin.Training()))
  snn = IF_Delta_Dense_Layer(n_in, n_rec)
  snn = bc.init_states(snn, n_batch)

  algorithm = nn.DiagExpSmOnAlgorithm(snn, decay=0.99)
  algorithm.compile_graph(jax.ShapeDtypeStruct((n_batch, n_in), bc.environ.dftype()))

  def run_snn(i, inp_spk):
    bc.share.set(i=i, t=i * bc.environ.get_dt())
    out = algorithm.update_model_and_etrace(inp_spk)
    return out

  nt = 100
  inputs = jnp.asarray(bc.random.rand(nt, n_batch, n_in) < 0.2, dtype=bc.environ.dftype())
  outs = bc.transform.for_loop(run_snn, np.arange(nt), inputs)
  print(outs.shape)


def try_if_delta_etrace_update_On2():
  n_in, n_rec = 4, 10
  bc.environ.set(mode=bc.mixin.Training())
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
  outs = bc.transform.for_loop(run_snn, np.arange(nt), inputs)
  print(outs.shape)


def try_if_delta_etrace_update_batched_On2():
  n_batch, n_in, n_rec = 16, 4, 10
  bc.environ.set(mode=bc.mixin.JointMode(bc.mixin.Batching(), bc.mixin.Training()))
  snn = IF_Delta_Dense_Layer(n_in, n_rec)
  snn = bc.init_states(snn, n_batch)

  algorithm = nn.DiagOn2Algorithm(snn)
  algorithm.compile_graph(jax.ShapeDtypeStruct((n_batch, n_in,), bc.environ.dftype()))

  def run_snn(i, inp_spk):
    bc.share.set(i=i, t=i * bc.environ.get_dt())
    out = algorithm.update_model_and_etrace(inp_spk)
    return out

  nt = 100
  inputs = jnp.asarray(bc.random.rand(nt, n_batch, n_in) < 0.2, dtype=bc.environ.dftype())
  outs = bc.transform.for_loop(run_snn, np.arange(nt), inputs)
  print(outs.shape)


def try_if_delta_etrace_update_and_grad():
  n_batch, n_in, n_rec = 16, 4, 10
  bc.environ.set(mode=bc.mixin.JointMode(bc.mixin.Batching(), bc.mixin.Training()))
  snn = IF_Delta_Dense_Layer(n_in, n_rec)
  weights = snn.states().subset(nn.ETraceParam)
  snn = bc.init_states(snn, n_batch)

  algorithm = nn.DiagExpSmOnAlgorithm(snn, decay=0.99)
  algorithm.compile_graph(jax.ShapeDtypeStruct((n_batch, n_in), bc.environ.dftype()))

  def run_snn(last_grads, inputs):
    i, inp_spk = inputs
    bc.share.set(i=i, t=i * bc.environ.get_dt())
    out = algorithm.update_model_and_etrace(inp_spk)

    grads = bc.transform.grad(lambda x: algorithm.update_model_and_etrace(x).sum(),
                              grad_vars=weights)(inp_spk)
    grads = jax.tree.map(jnp.add, last_grads, grads)
    return grads, out

  nt = 100
  inp_spks = jnp.asarray(bc.random.rand(nt, n_batch, n_in) < 0.2, dtype=bc.environ.dftype())
  grads = jax.tree.map(lambda x: jnp.zeros_like(x), weights.to_dict_values())
  grads, outs = bc.transform.scan(run_snn, grads, (np.arange(nt), inp_spks))
  print(outs.shape)


def try_if_delta_etrace_grad_vs_bptt_grad():
  n_batch, n_in, n_rec = 16, 4, 10
  bc.environ.set(mode=bc.mixin.JointMode(bc.mixin.Batching(), bc.mixin.Training()))
  snn = IF_Delta_Dense_Layer(n_in, n_rec)
  weights = snn.states().subset(nn.ETraceParam)

  def etrace_grad(x):
    bc.init_states(snn, n_batch)
    algorithm = nn.DiagExpSmOnAlgorithm(snn, decay=0.99)
    algorithm.compile_graph(jax.ShapeDtypeStruct((n_batch, n_in), bc.environ.dftype()))

    def run_snn(last_grads, inputs):
      i, inp_spk = inputs
      bc.share.set(i=i, t=i * bc.environ.get_dt())
      out = algorithm.update_model_and_etrace(inp_spk)

      grads = bc.transform.grad(lambda x: algorithm.update_model_and_etrace(x).sum(),
                                grad_vars=weights)(inp_spk)
      grads = jax.tree.map(jnp.add, last_grads, grads)
      return grads, out

    grads = jax.tree.map(lambda x: jnp.zeros_like(x), weights.to_dict_values())
    grads, outs = bc.transform.scan(run_snn, grads, (np.arange(nt), x))
    return grads, outs

  def bptt_grad(x):
    bc.init_states(snn, n_batch)

    def run_snn(i, inp_spk):
      bc.share.set(i=i, t=i * bc.environ.get_dt())
      out = snn(inp_spk).sum()
      return out

    def simulate(inp_spks):
      outs = bc.transform.for_loop(run_snn, np.arange(nt), inp_spks)
      return outs.mean()

    grads = bc.transform.grad(simulate, grad_vars=weights)(x)
    return grads

  nt = 100
  inp_spks = jnp.asarray(bc.random.rand(nt, n_batch, n_in) < 0.2, dtype=bc.environ.dftype())
  etrace_grads, etrace_outs = etrace_grad(inp_spks)
  bptt_grads = bptt_grad(inp_spks)
  print(etrace_grads)
  print(bptt_grads)


if __name__ == '__main__':
  pass
  # try1()
  # try_if_delta_etrace_update()
  # try_if_delta_etrace_update_On2()
  try_if_delta_etrace_update_batched_On2()
  # try_if_delta_etrace_update_and_grad()
  # try_if_delta_etrace_grad_vs_bptt_grad()
  # try_traceback()
