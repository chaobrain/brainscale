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
#
# Author: Chaoming Wang <chao.brain@qq.com>
# Date: 2024-04-03
# Copyright: 2024, Chaoming Wang
# ==============================================================================

# -*- coding: utf-8 -*-

from __future__ import annotations

from functools import partial
from typing import Dict, Tuple, Any, Callable, List

import braincore as bc
import jax.core
import jax.numpy as jnp
from braincore.transform._autograd import functional_vector_grad as vector_grad

from ._errors import NotSupportedError
from ._etrace_compiler import ETraceGraph, TracedWeightOp
from ._etrace_concepts import (assign_state_values, split_states, split_states_v2,
                               stop_param_gradients, ETraceVar,
                               ETraceParamOp, ETraceGrad)
from .typing import (PyTree, Outputs, WeightID, HiddenVar, WeightXVar, WeightYVar,
                     HiddenVals, StateVals, ETraceVals,
                     dG_Inputs, dG_Weight, dG_Hidden, dG_State)

__all__ = [
  'ETraceAlgorithm',
  'DiagExpSmOnAlgorithm',
  'DiagOn2Algorithm',
  'DiagHybridAlgorithm',
]


def _format_decay_and_rank(decay, num_rank):
  # number of approximation rank and the decay factor
  if num_rank is None:
    assert 0 < decay < 1, f'The decay should be in (0, 1). While we got {decay}. '
    decay = decay  # (num_rank - 1) / (num_rank + 1)
    num_rank = round(2. / (1 - decay) - 1)
  elif decay is None:
    assert isinstance(num_rank, int), f'The num_rank should be an integer. While we got {num_rank}. '
    num_rank = num_rank
    decay = (num_rank - 1) / (num_rank + 1)  # (num_rank - 1) / (num_rank + 1)
  else:
    raise ValueError('Please provide "num_rank" (int) or "decay" (float, 0 < decay < 1). ')
  return decay, num_rank


def weight_op_gradient(op_jaxpr, dx, w, dy):
  def op(xs, ws):
    return jax.core.eval_jaxpr(op_jaxpr, (), *jax.tree.leaves([xs, ws]))[0]

  return jax.vjp(partial(op, dx), w)[1](dy)[0]


def expon_smooth(old, new, decay):
  """
  Exponential smoothing.

  :param old: the old value
  :param new: the new value
  :param decay: the decay factor
  :return: the smoothed value
  """
  return decay * old + (1 - decay) * new


def low_pass_filter(old, new, alpha):
  """
  Low-pass filter.

  :param old: the old value
  :param new: the new value
  :param alpha: the filter factor
  :return: the filtered value
  """
  return alpha * old + new


def tree_expon_smooth(olds, news, decay):
  """
  Exponential smoothing for the tree structure.

  :param olds: the old values
  :param news: the new values
  :param decay: the decay factor
  :return: the smoothed values
  """
  return jax.tree.map(partial(expon_smooth, decay=decay), olds, news)


def tree_low_pass_filter(olds, news, alpha):
  """
  Low-pass filter for the tree structure.

  :param olds: the old values
  :param news: the new values
  :param alpha: the filter factor
  :return: the filtered values
  """
  return jax.tree.map(partial(low_pass_filter, alpha=alpha), olds, news)


def update_dict(the_dict: Dict, key: Any, value: PyTree):
  """Update the dictionary.

  If the key exists, then add the value to the existing value.
  Otherwise, create a new key-value pair.

  Args:
    the_dict: The dictionary.
    key: The key.
    value: The value.
  """
  old_value = the_dict.get(key, None)
  if old_value is None:
    the_dict[key] = value
  else:
    the_dict[key] = jax.tree.map(jnp.add, old_value, value)


def dy_to_weight(mode: bc.mixin.Mode,
                 relation: TracedWeightOp,
                 weight_vals: PyTree,
                 dg_hidden: jax.Array) -> PyTree:
  # [KEY]
  # For the following operation:
  #      dL/dW = (dL/dH) \circ (dH / dW)
  #   or
  #      \partial H/\partial W = \partial H/\partial H \cdot \partial H/\partial W
  #
  # we can compute the gradient of the weight using the following two merging operations:

  # [KEY]
  # The weights and operators must be defined with ``ETraceParamOp``.
  # Moreover, it must be compatible with the batching and non-batching mode.
  if not isinstance(relation.weight, ETraceParamOp):
    raise NotSupportedError(f'The weight should be an {ETraceParamOp.__name__}. '
                            f'While we got {type(relation.weight)}. ')

  # input
  if mode.has(bc.mixin.Batching):
    # TODO: Assuming the batch size is the first dimension
    x_data = jnp.ones(relation.x.aval.shape[1:], relation.x.aval.dtype)
  else:
    x_data = jnp.ones(relation.x.aval.shape, relation.x.aval.dtype)

  # transform
  fun = lambda dh: jax.vjp(partial(relation.weight.op, x_data), weight_vals)[1](dh)[0]
  if mode.has(bc.mixin.Batching):
    dG_hidden_like_weight = jax.vmap(fun)(dg_hidden)
  else:
    dG_hidden_like_weight = fun(dg_hidden)

  return dG_hidden_like_weight


def dx_dy_to_weight(mode: bc.mixin.Mode,
                    relation: TracedWeightOp,
                    weight_vals: PyTree,
                    dg_x: jax.Array,
                    dg_y: jax.Array) -> PyTree:
  # [KEY]
  # For the following operation:
  #      dW = dy \otimes dx
  #
  # we can compute the gradient of the weight using the following two merging operations:

  if not isinstance(relation.weight, ETraceParamOp):
    raise NotSupportedError(f'The weight should be an {ETraceParamOp.__name__}. '
                            f'While we got {type(relation.weight)}. ')
  fun = lambda dx, dy: jax.vjp(partial(relation.weight.op, dx), weight_vals)[1](dy)[0]
  if mode.has(bc.mixin.Batching):
    dG_weight = jax.vmap(fun)(dg_x, dg_y)
  else:
    dG_weight = fun(dg_x, dg_y)
  return dG_weight


def _diag_hidden_update(self: 'ETraceAlgorithm', hiddens, others, *params):
  # [ KEY: assuming the weight are not changed ]
  assign_state_values(self.hidden_states, hiddens)
  assign_state_values(self.other_states, others)
  with stop_param_gradients():
    self.graph._call_org_model(*params)
  hiddens = [st.value for st in self.hidden_states]
  return hiddens


def batched_zeros_like(batch_size: int | None, x: jax.Array):
  """
  Create a batched zeros array like the input array.

  Args:
    batch_size: int, the batch size.
    x: jax.Array, the input array.

  Returns:
    jax.Array, the batched zeros array.
  """
  if batch_size is None:
    return jnp.zeros_like(x)
  else:
    return jnp.zeros((batch_size,) + x.shape, x.dtype)


class ETraceAlgorithm(bc.Module):
  """
  The base class for the eligibility trace algorithm.

  Parameters:
  -----------
  model_or_graph: Union[Callable, ETraceGraph]
      The model or the etrace graph. The model is the function that we want to
      compute the recurrent states in one time step. The etrace graph is the
      graph that we want to compute the eligibility trace.
  decay: float
      The decay factor for the eligibility trace. If the decay is not provided,
      the number of approximation rank ``num_rank`` should be provided.
  num_rank: int
      The number of approximation rank for the RTRL algorithm. If the number of
      approximation rank is not provided, the decay factor ``decay`` should be provided.
  """
  __module__ = 'brainscale'

  graph: ETraceGraph  # the etrace graph
  weight_states: List[bc.ParamState]  # the weight states
  hidden_states: List[ETraceVar]  # the hidden states
  other_states: List[bc.State]  # the other states
  is_compiled: bool  # whether the etrace algorithm has been compiled

  def __init__(self,
               model_or_graph: Callable | ETraceGraph,
               name: str | None = None,
               mode: bc.mixin.Mode | None = None):
    super().__init__(name=name, mode=mode)

    # the model and the graph
    if isinstance(model_or_graph, ETraceGraph):
      self.graph = model_or_graph
      self.is_compiled = model_or_graph.revised_jaxpr is not None
    else:
      if not callable(model_or_graph):
        raise ValueError('The model should be a callable function. ')
      self.graph = ETraceGraph(model_or_graph)
      self.is_compiled = False

  def compile_graph(self, *args, **kwargs) -> None:
    """
    Compile the eligibility trace graph of the relationship between etrace weight, variable and operators.

    The compilation process includes:
    - building the etrace graph
    - separating the states
    - initializing the states

    :param args: the input arguments
    :param kwargs: the keyword arguments
    """
    if not self.is_compiled:
      # --- the model etrace graph -- #
      self.graph.compile_graph(*args, **kwargs)

      # --- the state separation --- #
      # NOTE: the `ETraceGraph` and the following states suggests that
      # `ETraceAlgorithm` depends on the states we created in the `ETraceGraph`,
      # including:
      #   - the weight states, which is invariant during the training process
      #   - the hidden states, the recurrent states, which may be changed between different training epochs
      #   - the other states, which may be changed between different training epochs
      self.weight_states, self.hidden_states, self.other_states = split_states(self.graph.states)

      # --- the initialization of the states --- #
      self.init_etrace_state(*args, **kwargs)

      # mark the graph is compiled
      self.is_compiled = True

  def init_etrace_state(self, *args, **kwargs) -> None:
    """
    Initialize the eligibility trace states of the etrace algorithm.

    This method is needed after compiling the etrace graph. See `.compile_graph()` for the details.
    """
    raise NotImplementedError


class _DiagETraceAlgorithmForVJP(ETraceAlgorithm):
  """
  The base class for the eligibility trace algorithm which supporting the VJP gradient
  computation (reverse-mode differentiation).

  This module is designed to be compatible with the JAX's VJP mechanism.
  The true update function is defined as a custom VJP function ``._true_update_fun()``,
  which receives the inputs, the hidden states, other states, and etrace variables at
  the last time step, and returns the outputs, the hidden states, other states, and etrace
  variables at the current time step.

  For each subclass (or the instance of an etrace algorithm), we should define the
  following methods:

  - ``._update()``: update the eligibility trace states and return the outputs,
                    hidden states, other states, and etrace data.
  - ``._update_fwd()``: the forward pass of the custom VJP rule.
  - ``._update_bwd()``: the backward pass of the custom VJP rule.

  However, this class has provided a default implementation for the ``._update()``, ``._update_fwd()``,
  and ``._update_bwd()`` methods. To implement a new etrace algorithm, users just need to override the
  following methods:

  - ``._solve_temporal_gradients()``: solve the temporal gradients of the hidden states.
  - ``._update_etrace_data()``: update the eligibility trace data.
  - ``._assign_etrace_data()``: assign the eligibility trace data to the states.
  - ``._get_etrace_data()``: get the eligibility trace data.

  """

  __module__ = 'brainscale'

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

    # the update rule
    self._true_update_fun = jax.custom_vjp(self._update)
    self._true_update_fun.defvjp(fwd=self._update_fwd, bwd=self._update_bwd)

  def __call__(self, *args) -> Any:
    return self.update_model_and_etrace(*args)

  def update_model_and_etrace(self, *args) -> Any:
    if not self.is_compiled:
      raise ValueError('The etrace algorithm has not been compiled. Please call `compile_graph` first. ')

    # state values
    weight_vals = [st.value for st in self.weight_states]
    hidden_vals = [st.value for st in self.hidden_states]
    other_vals = [st.value for st in self.other_states]
    # etrace data
    last_etrace_vals = self._get_etrace_data()

    # update all states
    # [KEY] The key here is that we change the object-oriented attributes as the function arguments.
    #       Therefore, the function arguments are the states of the current time step, and the function
    #       returns the states of the next time step.
    out, hidden_vals, other_vals, new_etrace_vals = self._true_update_fun(args,
                                                                          weight_vals,
                                                                          hidden_vals,
                                                                          other_vals,
                                                                          last_etrace_vals)
    # assign the weight values back, [KEY] assuming the weight values are not changed
    assign_state_values(self.weight_states, weight_vals)
    # assign the new hidden and state values
    assign_state_values(self.hidden_states, hidden_vals)
    assign_state_values(self.other_states, other_vals)
    # assign the new etrace values
    self._assign_etrace_data(new_etrace_vals)
    return out

  def _update(self, inputs, weight_vals, hidden_vals, othstate_vals, etrace_vals,
              ) -> Tuple[Outputs, HiddenVals, StateVals, ETraceVals]:
    # state value assignment
    assign_state_values(self.weight_states, weight_vals)
    assign_state_values(self.hidden_states, hidden_vals)
    assign_state_values(self.other_states, othstate_vals)

    weight_id_to_its_val = {id(st): val for st, val in zip(self.weight_states, weight_vals)}

    # temporal gradients of the recurrent layer
    temporal_grads = self._solve_jacobian_gradients(*inputs)

    # spatial gradients of the weights
    out, hidden_vals, othstate_vals, current_etrace_data = self.graph.solve_spatial_gradients(*inputs)
    current_etrace_data = jax.lax.stop_gradient(current_etrace_data)

    # eligibility trace update
    etrace_vals = self._update_etrace_data(temporal_grads,
                                           etrace_vals,
                                           current_etrace_data,
                                           weight_id_to_its_val)

    # returns
    return out, hidden_vals, othstate_vals, etrace_vals

  def _update_fwd(self, args, weight_vals, hidden_vals, othstate_vals, etrace_vals,
                  ) -> Tuple[Tuple[Outputs, HiddenVals, StateVals, ETraceVals], Any]:
    # state value assignment
    assign_state_values(self.weight_states, weight_vals)
    assign_state_values(self.hidden_states, hidden_vals)
    assign_state_values(self.other_states, othstate_vals)

    weight_id_to_its_val = {id(st): val for st, val in zip(self.weight_states, weight_vals)}

    # temporal gradients of the recurrent layer
    temporal_grads = self._solve_jacobian_gradients(*args)

    # spatial gradients of the weights
    out, hiddens, oth_states, hid2weight_jac, residuals = self.graph.solve_spatial_gradients_and_vjp_jaxpr(*args)
    hid2weight_jac = jax.lax.stop_gradient(hid2weight_jac)

    # eligibility trace update
    etrace_vals = self._update_etrace_data(temporal_grads,
                                           etrace_vals,
                                           hid2weight_jac,
                                           weight_id_to_its_val)

    # returns
    return (out, hiddens, oth_states, etrace_vals), (residuals, etrace_vals, weight_id_to_its_val)

  def _update_bwd(self, fwd_res, grads) -> Tuple[dG_Inputs, dG_Weight, dG_Hidden, dG_State, None]:
    # interpret the fwd results
    residuals, etrace_vals, id2weight_val = fwd_res
    jaxpr, in_tree, out_tree, consts = residuals

    # interpret the downstream gradients
    # Since
    #     dg_out, dg_hiddens, dg_others, dg_etrace = grads
    # we need to remove the "dg_etrace" iterm from the gradients for matching the
    # jaxpr vjp gradients.
    grad_flat, grad_tree = jax.tree.flatten((grads[:-1],))

    # compute the original gradients
    if out_tree != grad_tree:
      raise TypeError(f'Gradient tree should be the same as the function output tree. '
                      f'While we got: \n'
                      f'out_tree  = {out_tree}\n!=\n'
                      f'grad_tree = {grad_tree}')
    cts_out = jax.core.eval_jaxpr(jaxpr, consts, *grad_flat)
    # The gradients of inputs, hidden states, and other states are computed through the
    # normal back-propagation algorithm.
    dg_args, dg_hiddens, dG_non_etrace_params, dg_othstates, dg_perturbs = jax.tree.unflatten(in_tree, cts_out)
    # However, the gradients of the weights are computed through the RTRL algorithm.
    dg_perturbs = {hid_var: dg for hid_var, dg in zip(self.graph.out_hidden_jaxvars, dg_perturbs)}
    dg_weights = self._solve_weight_gradients(etrace_vals,
                                              dg_perturbs,
                                              id2weight_val,
                                              dG_non_etrace_params)

    # Note that there are no gradients flowing through the etrace data.
    dg_etrace = None
    return dg_args, dg_weights, dg_hiddens, dg_othstates, dg_etrace

  def _solve_jacobian_gradients(self, *args) -> Dict[HiddenVar, jax.Array]:
    """
    The common method to solve the temporal Jacobian gradients of the hidden states.
    
    Note here the temporal gradients are the gradients of the hidden states with respect to the hidden states, 
    and only consider the diagonal structure of such hidden Jacobian matrix.
    """
    # [compute D^t]
    # approximate the hidden to hidden Jacobian diagonal using the VJP
    hidden_values = [st.value for st in self.hidden_states]
    other_values = [st.value for st in self.other_states]
    diagonal = vector_grad(partial(_diag_hidden_update, self),
                           argnums=0)(jax.lax.stop_gradient(hidden_values),
                                      jax.lax.stop_gradient(other_values),
                                      *jax.lax.stop_gradient(args), )
    diagonal = {self.graph.hidden_id_to_outvar[id(st)]: val
                for st, val in zip(self.hidden_states, diagonal)}

    # recovery the state values
    assign_state_values(self.hidden_states, hidden_values)
    assign_state_values(self.other_states, other_values)

    # returns
    diag_no_grad = jax.lax.stop_gradient(diagonal)
    return diag_no_grad

  def _solve_weight_gradients(self,
                              etrace_data: Any,
                              dG_hiddens: Dict[HiddenVar, jax.Array],
                              id2weight_val: Dict[WeightID, PyTree],
                              dG_non_etrace_params: List[PyTree]):
    """
    The method to solve the weight gradients.

    Args:
      etrace_data: Any, the eligibility trace data.
      dG_hiddens: Dict[HiddenVar, jax.Array], the gradients of the hidden states.
      id2weight_val: Dict[WeightID, PyTree], the weight values.
      dG_non_etrace_params: List[PyTree], the gradients of the non-etrace parameters
    """
    raise NotImplementedError

  def _update_etrace_data(self,
                          temporal_jacobian: Dict[HiddenVar, jax.Array],
                          etrace_vals: ETraceVals,
                          hid2weight_jac: ETraceVals,
                          weight_id_to_its_val: Dict[WeightID, PyTree]) -> ETraceVals:
    """
    The method to update the eligibility trace data.

    Args:
      temporal_jacobian: Dict[HiddenVar, jax.Array], the temporal gradients of the hidden states.
      etrace_vals: ETraceVals, the history eligibility trace data.
      hid2weight_jac: ETraceVals, the current eligibility trace data.

    Returns:
      ETraceVals, the updated eligibility trace data.
    """
    raise NotImplementedError

  def _get_etrace_data(self) -> ETraceVals:
    """
    Get the eligibility trace data.

    Returns:
      ETraceVals, the eligibility trace data.
    """
    raise NotImplementedError

  def _assign_etrace_data(self, etrace_vals: ETraceVals) -> None:
    """
    Assign the eligibility trace data to the states.

    Args:
      etrace_vals: ETraceVals, the eligibility trace data.
    """
    raise NotImplementedError


class DiagExpSmOnAlgorithm(_DiagETraceAlgorithmForVJP):
  """
  The online gradient computation algorithm with the exponential smoothing and diagonal approximation.

  This algorithm has the O(n) memory complexity and O(n^2) computational complexity, where n is the
  number of hidden states.

  """

  __module__ = 'brainscale'

  etrace_xs: Dict[WeightXVar, bc.State]  # the spatial gradients of the weights
  etrace_dfs: Dict[Tuple[WeightYVar, HiddenVar], bc.State]  # the spatial gradients of the hidden states

  decay: float  # the decay factor
  num_rank: int  # the number of approximation rank

  def __init__(self,
               model_or_graph: Callable | ETraceGraph,
               decay: float = None,
               num_rank: int = None,
               name: str | None = None,
               mode: bc.mixin.Mode | None = None):
    super().__init__(model_or_graph, name=name, mode=mode)

    # the learning parameters
    self.decay, self.num_rank = _format_decay_and_rank(decay, num_rank)

  def init_etrace_state(self, *args, **kwargs):
    # The states of weight spatial gradients:
    #   1. x
    #   2. df
    self.etrace_xs = bc.visible_state_dict()
    self.etrace_dfs = bc.visible_state_dict()
    for relation in self.graph.weight_hidden_relations:
      if relation.x not in self.etrace_xs:
        self.etrace_xs[relation.x] = bc.State(jnp.zeros(relation.x.aval.shape, relation.x.aval.dtype))
      for hidden_var in relation.hidden_vars:
        key = (relation.y, hidden_var)
        if key in self.etrace_dfs:
          raise ValueError(f'The relation {key} has been added. ')
        self.etrace_dfs[key] = bc.State(jnp.zeros(relation.y.aval.shape, relation.y.aval.dtype))

  def _get_etrace_data(self):
    etrace_xs = {k: v.value for k, v in self.etrace_xs.items()}
    etrace_dfs = {k: v.value for k, v in self.etrace_dfs.items()}
    return etrace_xs, etrace_dfs

  def _assign_etrace_data(self, hist_etrace_vals):
    etrace_xs, etrace_dfs = hist_etrace_vals
    # the weight x
    for x, val in etrace_xs.items():
      self.etrace_xs[x].value = val
    # the weight df
    for dfkey, val in etrace_dfs.items():
      self.etrace_dfs[dfkey].value = val

  def _update_etrace_data(
      self,
      temporal_jacobian: Dict[jax.core.Var, jax.Array],
      hist_etrace_vals: PyTree,
      hid2weight_jac: Tuple[Dict[WeightXVar, jax.Array], Dict[Tuple[WeightYVar, HiddenVar], jax.Array]],
      weight_id_to_its_val: Dict[WeightID, PyTree]
  ) -> ETraceVals:

    # the etrace data at the current time step (t) of the O(n) algorithm
    # is a tuple, including the weight x and df values.
    xs, dfs = hid2weight_jac

    # the history etrace values
    hist_xs, hist_dfs = hist_etrace_vals

    # the new etrace values
    new_etrace_xs, new_etrace_dfs = dict(), dict()

    # update the weight x
    for x in hist_xs.keys():
      new_etrace_xs[x] = low_pass_filter(hist_xs[x], xs[x], self.decay)

    # update the weight df * diagonal
    for dfkey in hist_dfs.keys():
      df_var, hidden_var = dfkey
      new_etrace_dfs[dfkey] = hist_dfs[dfkey] * temporal_jacobian[hidden_var]

    # update the weight df
    for dfkey in hist_dfs.keys():
      new_etrace_dfs[dfkey] = expon_smooth(new_etrace_dfs[dfkey], dfs[dfkey], self.decay)
    return new_etrace_xs, new_etrace_dfs

  def _solve_weight_gradients(
      self,
      hist_etrace_data: Tuple[Dict[WeightXVar, jax.Array], Dict[Tuple[WeightYVar, HiddenVar], jax.Array]],
      dG_hiddens: Dict[HiddenVar, jax.Array],
      weight_id_to_its_val: Dict[WeightID, PyTree],
      dG_non_etrace_params: List[PyTree]
  ):
    """
    Solve the weight gradients according to the eligibility trace data.
    
    Particularly, for each weight, we compute its gradients according to the ``x`` and ``df``.
    """
    xs, dfs = hist_etrace_data
    dg_weights = {id(st): None for st in self.weight_states}
    for relation in self.graph.weight_hidden_relations:
      x = xs[relation.x]
      for i, hid_var in enumerate(relation.hidden_vars):
        df = dfs[(relation.y, hid_var)]
        df_hid = df * relation.hidden2df[i](dG_hiddens[hid_var])
        weight_id = id(relation.weight)
        dg_weight = weight_op_gradient(relation.op_jaxpr, x, weight_id_to_its_val[weight_id], df_hid)
        update_dict(dg_weights, weight_id, dg_weight)

    # update the non-etrace parameters
    _, _, non_etrace_params, _ = split_states_v2(self.graph.states)
    for st, dg in zip(non_etrace_params, dG_non_etrace_params):
      update_dict(dg_weights, id(st), dg)
    return list(dg_weights.values())


class DiagOn2Algorithm(_DiagETraceAlgorithmForVJP):
  """
  The online gradient computation algorithm with the diagonal approximation.

  This algorithm has the O(n^2) memory complexity and O(n^3) computational complexity, where n is the
  number of hidden states.

  """

  etrace_bwg: Dict[Tuple[WeightID, WeightXVar, HiddenVar], bc.State]  # batch of weight gradients

  def init_etrace_state(self, *args, **kwargs):
    # The states of batched weight gradients
    self.etrace_bwg = bc.visible_state_dict()
    for relation in self.graph.weight_hidden_relations:
      # TODO: assume the batch size is the first dimension
      batch_size = relation.x.aval.shape[0] if self.mode.has(bc.mixin.Batching) else None
      for hidden_var in relation.hidden_vars:
        key = (id(relation.weight), relation.x, hidden_var)
        if key in self.etrace_bwg:
          raise ValueError(f'The relation {key} has been added. ')
        self.etrace_bwg[key] = bc.State(jax.tree.map(partial(batched_zeros_like, batch_size),
                                                     relation.weight.value))

  def _get_etrace_data(self) -> Dict:
    return {k: v.value for k, v in self.etrace_bwg.items()}

  def _assign_etrace_data(self, etrace_vals: Dict) -> None:
    for x, val in etrace_vals.items():
      self.etrace_bwg[x].value = val

  def _update_etrace_data(
      self,
      temporal_jacobian: Dict[HiddenVar, jax.Array],
      hist_etrace_vals: Dict[Tuple[WeightID, WeightXVar, HiddenVar], PyTree],
      hid2weight_jac: Tuple[Dict[WeightXVar, jax.Array], Dict[Tuple[WeightYVar, HiddenVar], jax.Array]],
      weight_id_to_its_val: Dict[WeightID, PyTree]
  ) -> Dict[Tuple[WeightID, WeightXVar, HiddenVar], PyTree]:
    # 1. "hist_etrace_vals" has the following structure:
    #    - key: the weight id, the weight-x jax var, the hidden state var
    #    - value: the batched weight gradients
    #
    # 2. "hid2weight_jac" has the following structure:
    #    - a dict of weight x gradients
    #       * key: the weight x jax var
    #       * value: the weight x gradients
    #    - a dict of weight y gradients
    #       * key: the tuple of the weight y jax var and the hidden state jax var
    #       * value: the weight y gradients
    #
    # 3. "temporal_jacobian" has the following structure:
    #    - key: the hidden state jax var
    #    - value: the hidden state jacobian gradients

    cur_etrace_xs, cur_etrace_ys = hid2weight_jac

    new_etrace_bwg = dict()
    for relation in self.graph.weight_hidden_relations:
      weight_id = id(relation.weight)
      weight_vals = weight_id_to_its_val[weight_id]
      for i, hid_var in enumerate(relation.hidden_vars):
        w_key = (weight_id, relation.x, hid_var)
        y_key = (relation.y, hid_var)
        dg_hidden = relation.hidden2df[i](temporal_jacobian[hid_var])
        dg_weight = dy_to_weight(self.mode, relation, weight_vals, dg_hidden)
        current_etrace = dx_dy_to_weight(self.mode, relation, weight_vals,
                                         cur_etrace_xs[relation.x],
                                         cur_etrace_ys[y_key])
        new_etrace_bwg[w_key] = jax.tree.map(lambda old, jac, new: old * jac + new,
                                             hist_etrace_vals[w_key],
                                             dg_weight,
                                             current_etrace)
    return new_etrace_bwg

  def _solve_weight_gradients(self,
                              etrace_data: Dict[Any, PyTree],
                              dG_hiddens: Dict[HiddenVar, jax.Array],
                              weight_id_to_its_val: Dict[WeightID, PyTree],
                              dG_non_etrace_params: List[PyTree]):
    """
    Solve the weight gradients according to the eligibility trace data.

    Particularly, for each weight, we compute its gradients according to the batched weight gradients.
    """

    # update the etrace weight gradients
    temp_data = dict()
    for relation in self.graph.weight_hidden_relations:
      weight_id = id(relation.weight)
      weight_vals = weight_id_to_its_val[weight_id]
      for i, hid_var in enumerate(relation.hidden_vars):
        key = (weight_id, relation.x, hid_var)
        # dE/dH, hidden to weight gradients
        dg_hidden = relation.hidden2df[i](dG_hiddens[hid_var])
        hid2w = dy_to_weight(self.mode, relation, weight_vals, dg_hidden)
        # dE/dW = dE/dH * dH/dW
        dg_weight = jax.tree.map(lambda x, y: x * y, etrace_data[key], hid2w)
        update_dict(temp_data, weight_id, dg_weight)
    if self.mode.has(bc.mixin.Batching):
      # sum up the batched weight gradients
      for key, val in temp_data.items():
        temp_data[key] = jax.tree_map(lambda x: jnp.sum(x, axis=0), val)

    # update the weight gradients
    dG_weights = {id(st): None for st in self.weight_states}
    for key, val in temp_data.items():
      update_dict(dG_weights, key, val)

    # update the non-etrace weight gradients
    _, _, non_etrace_params, _ = split_states_v2(self.graph.states)
    for st, dg in zip(non_etrace_params, dG_non_etrace_params):
      update_dict(dG_weights, id(st), dg)

    return list(dG_weights.values())


class DiagHybridAlgorithm(_DiagETraceAlgorithmForVJP):
  etrace_xs: Dict[WeightXVar, bc.State]  # the spatial gradients of the weights
  etrace_dfs: Dict[Tuple[WeightYVar, HiddenVar], bc.State]  # the spatial gradients of the hidden states
  etrace_bwg: Dict[Tuple[WeightID, WeightXVar, HiddenVar], bc.State]  # batch of weight gradients

  decay: float  # the decay factor
  num_rank: int  # the number of approximation rank

  def __init__(self,
               model_or_graph: Callable | ETraceGraph,
               decay: float = None,
               num_rank: int = None,
               name: str | None = None,
               mode: bc.mixin.Mode | None = None):
    super().__init__(model_or_graph, name=name, mode=mode)

    # the learning parameters
    self.decay, self.num_rank = _format_decay_and_rank(decay, num_rank)

  def init_etrace_state(self, *args, **kwargs):
    # The states of weight spatial gradients:
    #   1. x
    #   2. df
    #   3. batched weight gradients
    self.etrace_xs = bc.visible_state_dict()
    self.etrace_dfs = bc.visible_state_dict()
    self.etrace_bwg = bc.visible_state_dict()
    for relation in self.graph.weight_hidden_relations:
      if isinstance(relation.weight, ETraceParamOp) and relation.weight.gradient == ETraceGrad.full:
        # TODO: assume the batch size is the first dimension
        batch_size = relation.x.aval.shape[0] if self.mode.has(bc.mixin.Batching) else None
        for hidden_var in relation.hidden_vars:
          key = (id(relation.weight), relation.x, hidden_var)
          if key in self.etrace_bwg:
            raise ValueError(f'The relation {key} has been added. ')
          self.etrace_bwg[key] = bc.State(jax.tree.map(partial(batched_zeros_like, batch_size),
                                                       relation.weight.value))
      else:
        if relation.x not in self.etrace_xs:
          self.etrace_xs[relation.x] = bc.State(jnp.zeros(relation.x.aval.shape, relation.x.aval.dtype))
        for hidden_var in relation.hidden_vars:
          key = (relation.y, hidden_var)
          if key in self.etrace_dfs:
            raise ValueError(f'The relation {key} has been added. ')
          self.etrace_dfs[key] = bc.State(jnp.zeros(relation.y.aval.shape, relation.y.aval.dtype))

  def _get_etrace_data(self) -> Tuple[Dict, Dict, Dict]:
    etrace_xs = {x: val.value for x, val in self.etrace_xs.items()}
    etrace_dfs = {x: val.value for x, val in self.etrace_dfs.items()}
    etrace_wgrads = {x: val.value for x, val in self.etrace_bwg.items()}
    return etrace_xs, etrace_dfs, etrace_wgrads

  def _assign_etrace_data(self, etrace_vals: Tuple[Dict, Dict, Dict]) -> None:
    etrace_xs, etrace_dfs, etrace_wgrads = etrace_vals
    for x, val in etrace_xs.items():
      self.etrace_xs[x].value = val
    for x, val in etrace_dfs.items():
      self.etrace_dfs[x].value = val
    for x, val in etrace_wgrads.items():
      self.etrace_bwg[x].value = val

  def _update_etrace_data(
      self,
      temporal_jacobian: Dict[HiddenVar, jax.Array],
      hist_etrace_vals: Tuple[Dict, Dict, Dict],
      hid2weight_jac: Tuple[Dict[WeightXVar, jax.Array], Dict[Tuple[WeightYVar, HiddenVar], jax.Array]],
      weight_id_to_its_val: Dict[WeightID, PyTree]
  ) -> Tuple[Dict, Dict, Dict]:
    # the history etrace values
    hist_xs, hist_dfs, hist_wgrads = hist_etrace_vals

    # the current spatial gradients
    cur_etrace_xs, cur_etrace_ys = hid2weight_jac

    # the new etrace values
    new_etrace_xs, new_etrace_dfs, new_etrace_bwg = dict(), dict(), dict()

    # update the etrace weight gradients
    for relation in self.graph.weight_hidden_relations:
      if isinstance(relation.weight, ETraceParamOp) and relation.weight.gradient == ETraceGrad.full:
        weight_id = id(relation.weight)
        weight_vals = weight_id_to_its_val[weight_id]
        for i, hid_var in enumerate(relation.hidden_vars):
          key = (weight_id, relation.x, hid_var)
          y_key = (relation.y, hid_var)
          dg_hidden = relation.hidden2df[i](temporal_jacobian[hid_var])
          dg_weight = dy_to_weight(self.mode, relation, weight_vals, dg_hidden)
          current_etrace = dx_dy_to_weight(self.mode, relation, weight_vals,
                                           cur_etrace_xs[relation.x],
                                           cur_etrace_ys[y_key])
          new_etrace_bwg[key] = jax.tree.map(lambda old, jac, new: old * jac + new,
                                             hist_wgrads[key],
                                             dg_weight,
                                             current_etrace)

    # update the weight x
    for x in hist_xs.keys():
      new_etrace_xs[x] = low_pass_filter(hist_xs[x],
                                         cur_etrace_xs[x],
                                         self.decay)

    # update the weight df * diagonal
    for dfkey in hist_dfs.keys():
      df_var, hidden_var = dfkey
      new_etrace_dfs[dfkey] = hist_dfs[dfkey] * temporal_jacobian[hidden_var]

    # update the weight df
    for dfkey in hist_dfs.keys():
      new_etrace_dfs[dfkey] = expon_smooth(new_etrace_dfs[dfkey],
                                           cur_etrace_ys[dfkey],
                                           self.decay)
    return new_etrace_xs, new_etrace_dfs, new_etrace_bwg

  def _solve_weight_gradients(self,
                              etrace_data: Dict[Any, PyTree],
                              dG_hiddens: Dict[HiddenVar, jax.Array],
                              weight_id_to_its_val: Dict[WeightID, PyTree],
                              dG_non_etrace_params: List[PyTree]):
    """
    Solve the weight gradients according to the eligibility trace data.

    Particularly, for each weight, we compute its gradients according to the batched weight gradients.
    """

    xs, dfs, wgrads = etrace_data
    dG_weights = {id(st): None for st in self.weight_states}

    # update the etrace weight gradients
    temp_data = dict()
    for relation in self.graph.weight_hidden_relations:
      if isinstance(relation.weight, ETraceParamOp) and relation.weight.gradient == ETraceGrad.full:
        weight_id = id(relation.weight)
        weight_vals = weight_id_to_its_val[weight_id]
        for i, hid_var in enumerate(relation.hidden_vars):
          key = (weight_id, relation.x, hid_var)
          # dE/dH, hidden to weight gradients
          dg_hidden = relation.hidden2df[i](dG_hiddens[hid_var])
          hid2w = dy_to_weight(self.mode, relation, weight_vals, dg_hidden)
          # dE/dW = dE/dH * dH/dW
          dg_weight = jax.tree.map(lambda x, y: x * y, wgrads[key], hid2w)
          update_dict(temp_data, weight_id, dg_weight)
      else:
        x = xs[relation.x]
        for i, hid_var in enumerate(relation.hidden_vars):
          df = dfs[(relation.y, hid_var)]
          df_hid = df * relation.hidden2df[i](dG_hiddens[hid_var])
          weight_id = id(relation.weight)
          dg_weight = weight_op_gradient(relation.op_jaxpr, x, weight_id_to_its_val[weight_id], df_hid)
          update_dict(dG_weights, weight_id, dg_weight)

    # update the weight gradients
    if self.mode.has(bc.mixin.Batching):
      # average the batched weight gradients
      for key, val in temp_data.items():
        temp_data[key] = jax.tree_map(lambda x: jnp.mean(x, axis=0), val)
    for key, val in temp_data.items():
      update_dict(dG_weights, key, val)

    # update the non-etrace weight gradients
    _, _, non_etrace_params, _ = split_states_v2(self.graph.states)
    for st, dg in zip(non_etrace_params, dG_non_etrace_params):
      update_dict(dG_weights, id(st), dg)

    return list(dG_weights.values())
