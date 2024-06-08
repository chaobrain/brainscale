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
from typing import Dict, Tuple, Any, Callable, List, Protocol, Optional

import brainstate as bst
import jax.core
import jax.numpy as jnp
from brainstate.transform._autograd import functional_vector_grad as vgrad

from ._etrace_compiler import (ETraceGraphForVJP,
                               HiddenWeightOpRelation,
                               HiddenGroupRelation,
                               _DiagJacobian,
                               _VJPTime)
from ._etrace_concepts import (assign_state_values,
                               split_states,
                               split_states_v2,
                               stop_param_gradients,
                               ETraceVar,
                               ETraceParamOp,
                               _ETraceGrad)
from ._etrace_operators import (StandardETraceOp,
                                GeneralETraceOp)
from ._typing import (PyTree,
                      Outputs,
                      WeightID,
                      HiddenOutVar,
                      WeightXVar,
                      WeightYVar,
                      HiddenVals,
                      StateVals,
                      ETraceVals,
                      dG_Inputs,
                      dG_Weight,
                      dG_Hidden,
                      dG_State)

__all__ = [
  'ETraceAlgorithm',
  'FakedETraceAlgorithm',
  'DiagETraceAlgorithmForVJP',
  'DiagIODimAlgorithm',  # the diagonally approximated algorithm with the input-output dimension complexity
  'DiagParamDimAlgorithm',  # the diagonally approximated algorithm with the parameter dimension complexity
  'DiagHybridDimAlgorithm',  # the diagonally approximated algorithm with hybrid complexity (either I/O or parameter)
]

_common_doc = '''
  diag_jacobian: str
      The method to compute the hidden Jacobian diagonal matrix. It should be one of
      the following values:

      - 'exact': the exact Jacobian diagonal matrix
      - 'vjp': the vector-Jacobian product computed Jacobian diagonal matrix
      - 'jvp': the Jacobian-vector product computed Jacobian diagonal matrix
  diag_normalize: bool
      Whether to normalize the hidden Jacobian diagonal matrix to the range of ``[-1, 1]``.
      Supported only when the ``diag_jacobian`` is ``'vjp'`` or ``'jvp'``. Default is ``None``.
  vjp_time: str
      The time to compute the loss-to-hidden Jacobian. It should be one of the
      following values:

      - 't': compute the loss-to-hidden Jacobian at the current time step
      - 't-1': compute the loss-to-hidden Jacobian at the last time step
  name: str, optional
      The name of the etrace algorithm.
  mode: brainstate.mixin.Mode, optional
      The mode of the etrace algorithm. Note that the etrace algorithm is particularly sensitive to 
      ``brainstate.mixin.Batching``, since it is used to compute and initialize the eligibility trace 
      states with or without batch size.

'''

_io_dim_doc = '''
  decay_or_rank: float, int
      The exponential smoothing factor for the eligibility trace. If it is a float,
      it is the decay factor, should be in the range of (0, 1). If it is an integer,
      it is the number of approximation rank for the algorithm, should be greater than 0.
  num_snap: int
      The number of chunks for the online learning. If it is None, it will not use
      the chunked eligibility trace.
  snap_freq: int
      The frequency of the chunked eligibility trace. If it is None, it will use the
      number of approximation rank.
'''


def _format_decay_and_rank(decay_or_rank) -> Tuple[float, int]:
  """
  Format the decay or the rank of the approximation.

  Args:
    decay_or_rank: float, int, the decay factor or the number of approximation rank.

  Returns:
    Tuple[float, int], the decay factor and the number of approximation rank.
  """
  # number of approximation rank and the decay factor
  if isinstance(decay_or_rank, float):
    assert 0 < decay_or_rank < 1, f'The decay should be in (0, 1). While we got {decay_or_rank}. '
    decay = decay_or_rank  # (num_rank - 1) / (num_rank + 1)
    num_rank = round(2. / (1 - decay) - 1)
  elif isinstance(decay_or_rank, int):
    assert decay_or_rank > 0, f'The num_rank should be greater than 0. While we got {decay_or_rank}. '
    num_rank = decay_or_rank
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


def update_dict(the_dict: Dict,
                key: Any,
                value: PyTree,
                error_when_no_key: Optional[bool] = False):
  """Update the dictionary.

  If the key exists, then add the value to the existing value.
  Otherwise, create a new key-value pair.

  Args:
    the_dict: The dictionary.
    key: The key.
    value: The value.
    error_when_no_key: bool, whether to raise an error when the key does not exist.

  """
  old_value = the_dict.get(key, None)
  if old_value is None:
    if error_when_no_key:
      raise ValueError(f'The key {key} does not exist in the dictionary. ')
    the_dict[key] = value
  else:
    the_dict[key] = jax.tree.map(jnp.add, old_value, value)


def batched_zeros_like(batch_size: Optional[int],
                       x: jax.Array):
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


def _normalize(x):
  """
  Normalize the input array to ``[-1, 1]``.

  Args:
    x: jax.Array, the input array.

  Returns:
    jax.Array, the normalized array.
  """
  max_ = jnp.max(jnp.abs(x))
  return jnp.where(jnp.allclose(max_, 0), x, x / max_)


class ETraceAlgorithm(bst.Module):
  r"""
  The base class for the eligibility trace algorithm.

  Note than the :py:class:`ETraceAlgorithm` is a subclass of :py:class:`brainstate.Module`,
  meaning that it is sensitive to the context/mode of the computation.Particularly,
  the :py:class:`ETraceAlgorithm` is sensitive to ``brainstate.mixin.Batching``.

  Parameters:
  -----------
  {common}

  """
  __module__ = 'brainscale'

  graph: ETraceGraphForVJP  # the etrace graph
  param_states: List[bst.ParamState]  # the weight states
  hidden_states: List[ETraceVar]  # the hidden states
  other_states: List[bst.State]  # the other states
  is_compiled: bool  # whether the etrace algorithm has been compiled
  diag_normalize: bool  # whether to normalize the hidden Jacobian diagonal matrix

  def __init__(
      self,
      model: Callable,
      diag_normalize: Optional[bool] = None,
      diag_jacobian: str = 'exact',
      vjp_time: str = 't',
      name: Optional[str] = None,
      mode: Optional[bst.mixin.Mode] = None
  ):
    super().__init__(name=name, mode=mode)

    # The method to compute the hidden Jacobian diagonal matrix,
    # and whether to normalize the hidden Jacobian diagonal matrix
    self.diag_jacobian = _DiagJacobian.get(diag_jacobian)
    if self.diag_jacobian == _DiagJacobian.exact:
      if diag_normalize is not None:
        raise ValueError('The normalization is not supported for the exact Jacobian diagonal matrix. ')
    self.diag_normalize = False if diag_normalize is None else diag_normalize

    # The time to compute the loss-to-hidden Jacobian
    self.vjp_time = _VJPTime.get(vjp_time)

    # The model and graph
    if not callable(model):
      raise ValueError(f'The model should be a callable function. But we got {model}.')
    self.graph = ETraceGraphForVJP(model, diag_jacobian=self.diag_jacobian, vjp_time=vjp_time)

    # The flag to indicate whether the etrace algorithm has been compiled
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
      # [NOTE]
      # The `ETraceGraph` and the following states suggests that
      # `ETraceAlgorithm` depends on the states we created in the
      # `ETraceGraph`, including:
      #   - the weight states, which is invariant during the training process
      #   - the hidden states, the recurrent states, which may be changed between different training epochs
      #   - the other states, which may be changed between different training epochs
      self.param_states, self.hidden_states, self.other_states = split_states(self.graph.states)

      # --- the initialization of the states --- #
      self.init_etrace_state(*args, **kwargs)

      # mark the graph is compiled
      self.is_compiled = True

  def show_graph(self) -> None:
    """
    Show the etrace graph.
    """
    return self.graph.show_graph()

  def __call__(self, *args, running_index: int = None) -> Any:
    """
    Update the model and the eligibility trace states.
    """
    return self.update_model_and_etrace(*args, running_index=running_index)

  def update_model_and_etrace(self, *args, running_index: int = None) -> Any:
    """
    Update the model and the eligibility trace states.

    :param args: the input arguments.
    :param running_index: int, the running index at the current time step.
    :return: the output of the model.
    """
    raise NotImplementedError

  def init_etrace_state(self, *args, **kwargs) -> None:
    """
    Initialize the eligibility trace states of the etrace algorithm.

    This method is needed after compiling the etrace graph. See `.compile_graph()` for the details.
    """
    raise NotImplementedError


class FakedETraceAlgorithm(ETraceAlgorithm):
  """
  Faked eligibility trace algorithm which is used to be compatible with the
  standard etrace algorithm.

  Particularly, for the given model which receives::

      out = model(*args)

  The etrace algorithm should be defined and called as::

      etrace = ETraceAlgorithm(model)
      out = etrace(*args, running_index=i)

  The :py:class:`FakedETraceAlgorithm` is a faked etrace algorithm which does not update the etrace data,
  but receives the same arguments as the standard etrace algorithm.
  """

  def update_model_and_etrace(self, *args, running_index: int = None) -> Any:
    return self.graph.model(*args)


def _diag_hidden_update(self: 'ETraceAlgorithm', hiddens, others, *params):
  # [ KEY ]  assuming the weight are not changed
  assign_state_values(self.hidden_states, hiddens)
  assign_state_values(self.other_states, others)
  with stop_param_gradients():
    self.graph.model(*params)
  hiddens = [st.value for st in self.hidden_states]
  return hiddens


def _solve_diagonal_jacobian_by_vjp_or_jvp(self: ETraceAlgorithm, *args) -> Dict[HiddenOutVar, jax.Array]:
  """
  The common method to solve the temporal Jacobian gradients of the hidden states.

  Note here the temporal gradients are the gradients of the hidden states with respect to the hidden states,
  and only consider the diagonal structure of such hidden Jacobian matrix.
  """
  # [compute D^t]
  # approximate the hidden to hidden Jacobian diagonal using the VJP
  hidden_values = [st.value for st in self.hidden_states]
  other_values = [st.value for st in self.other_states]
  hidden_values = jax.lax.stop_gradient(hidden_values)
  other_values = jax.lax.stop_gradient(other_values)
  args = jax.lax.stop_gradient(args)

  # compute the diagonal Jacobian matrix through VJP
  if self.diag_jacobian == _DiagJacobian.vjp:
    diagonal = vgrad(partial(_diag_hidden_update, self), argnums=0)(hidden_values, other_values, *args)
    diagonal = {self.graph.hidden_id_to_outvar[id(st)]: val
                for st, val in zip(self.hidden_states, diagonal)}

  # compute the diagonal Jacobian matrix through JVP
  elif self.diag_jacobian == _DiagJacobian.jvp:
    hidden_ones = jax.tree.map(jnp.zeros_like, hidden_values)
    fun = lambda hiddens: _diag_hidden_update(self, hiddens, other_values, *args)
    _, diagonal = jax.jvp(fun, (hidden_values,), (hidden_ones,))
    diagonal = {self.graph.hidden_id_to_outvar[id(st)]: val
                for st, val in zip(self.hidden_states, diagonal)}

  else:
    raise ValueError(f'The diagonal Jacobian method {self.diag_jacobian} is not supported. ')

  # recovery the state values
  assign_state_values(self.hidden_states, hidden_values)
  assign_state_values(self.other_states, other_values)

  return diagonal


class DiagETraceAlgorithmForVJP(FakedETraceAlgorithm):
  """
  The base class for the eligibility trace algorithm which supporting the VJP gradient
  computation (reverse-mode differentiation).

  This module is designed to be compatible with the JAX's VJP mechanism.
  The true update function is defined as a custom VJP function ``._true_update_fun()``,
  which receives the inputs, the hidden states, other states, and etrace variables at
  the last time step, and returns the outputs, the hidden states, other states, and
  etrace variables at the current time step.

  For each subclass (or the instance of an etrace algorithm), we should define the
  following methods:

  - ``._update()``: update the eligibility trace states and return the outputs,
                    hidden states, other states, and etrace data.
  - ``._update_fwd()``: the forward pass of the custom VJP rule.
  - ``._update_bwd()``: the backward pass of the custom VJP rule.

  However, this class has provided a default implementation for the ``._update()``,
  ``._update_fwd()``, and ``._update_bwd()`` methods.

  To implement a new etrace algorithm, users just need to override the following methods:

  - ``._solve_weight_gradients()``: solve the gradients of the learnable weights / parameters.
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

  def update_model_and_etrace(self, *args, running_index: int = None) -> Any:
    # ----------------------------------------------------------------------------------------------
    #
    # This method is the main method to update the model and the eligibility trace states.
    #
    # The key here is that we change the object-oriented attributes as the function arguments.
    # Therefore, the function arguments are the states of the current time step, and the function
    # returns the states of the next time step.
    #
    # Particularly, the model calls the "_true_update_fun()" function to update the states.
    #
    # ----------------------------------------------------------------------------------------------

    if not self.is_compiled:
      raise ValueError('The etrace algorithm has not been compiled. Please call `compile_graph` first. ')

    # state values
    weight_vals = [st.value for st in self.param_states]
    hidden_vals = [st.value for st in self.hidden_states]
    other_vals = [st.value for st in self.other_states]
    # etrace data
    last_etrace_vals = self._get_etrace_data()

    # update all states
    # [KEY] The key here is that we change the object-oriented attributes as the function arguments.
    #       Therefore, the function arguments are the states of the current time step, and the function
    #       returns the states of the next time step.
    out, hidden_vals, other_vals, new_etrace_vals = self._true_update_fun(
      args,
      weight_vals,
      hidden_vals,
      other_vals,
      last_etrace_vals,
      running_index
    )
    # assign the weight values back,
    # [KEY] assuming the weight values are not changed
    assign_state_values(self.param_states, weight_vals)
    # assign the new hidden and state values
    assign_state_values(self.hidden_states, hidden_vals)
    assign_state_values(self.other_states, other_vals)
    # assign the new etrace values
    self._assign_etrace_data(new_etrace_vals)
    return out

  def _update(
      self,
      inputs,
      weight_vals,
      hidden_vals,
      oth_state_vals,
      etrace_vals,
      running_index,
  ) -> Tuple[Outputs, HiddenVals, StateVals, ETraceVals]:
    # ----------------------------------------------------------------------------------------------
    #
    # The main function to update the [model] and the [eligibility trace] states.
    #
    # Particularly, ``self.graph.solve_h2w_h2h_jacobian()`` is called to:
    #   - compute the model output, the hidden states, and the other states
    #   - compute the hidden-to-weight Jacobian and the hidden-to-hidden Jacobian
    #
    # Then, ``self._update_etrace_data`` is called to:
    #   - update the eligibility trace data
    #
    # Moreover, this function returns:
    #   - the model output
    #   - the updated hidden states
    #   - the updated other general states
    #   - the updated eligibility trace data
    #
    # ----------------------------------------------------------------------------------------------

    # state value assignment
    assign_state_values(self.param_states, weight_vals)
    assign_state_values(self.hidden_states, hidden_vals)
    assign_state_values(self.other_states, oth_state_vals)

    weight_id_to_its_val = {id(st): val for st, val in zip(self.param_states, weight_vals)}

    # necessary gradients of the weights
    out, hidden_vals, oth_state_vals, hid2weight_jac, data_for_hid2hid_jac = (
      self.graph.solve_h2w_h2h_jacobian(*inputs)
    )
    hid2weight_jac = jax.lax.stop_gradient(hid2weight_jac)

    # gradients for the diagonal hidden Jacobian matrix
    if self.diag_jacobian != _DiagJacobian.exact:
      data_for_hid2hid_jac = _solve_diagonal_jacobian_by_vjp_or_jvp(self, *inputs)

    # eligibility trace update
    etrace_vals = self._update_etrace_data(
      running_index,
      etrace_vals,
      hid2weight_jac,
      data_for_hid2hid_jac,
      weight_id_to_its_val,
    )

    # returns
    return out, hidden_vals, oth_state_vals, etrace_vals

  def _update_fwd(
      self,
      args,
      weight_vals,
      hidden_vals,
      othstate_vals,
      etrace_vals,
      running_index,
  ) -> Tuple[Tuple[Outputs, HiddenVals, StateVals, ETraceVals], Any]:
    # ----------------------------------------------------------------------------------------------
    #
    # The forward function to update the [model] and the [eligibility trace] states when computing
    # the VJP gradients.
    #
    # Particularly, ``self.graph.solve_h2w_h2h_jacobian_and_l2h_vjp()`` is called to:
    #   - compute the model output, the hidden states, and the other states
    #   - compute the hidden-to-weight Jacobian and the hidden-to-hidden Jacobian
    #   - compute the loss-to-hidden or loss-to-weight Jacobian
    #
    # Then, ``self._update_etrace_data`` is called to:
    #   - update the eligibility trace data
    #
    # The forward function returns two parts of data:
    #   - The first part is the functional returns (same as "self._update()" function):
    #       * the model output
    #       * the updated hidden states
    #       * the updated other general states
    #       * the updated eligibility trace data
    #   - The second part is the data used for backward gradient computation:
    #       * the residuals of the model
    #       * the eligibility trace data at the current/last time step
    #       * the weight id to its value mapping
    #       * the running index
    #
    # ----------------------------------------------------------------------------------------------

    # state value assignment
    assign_state_values(self.param_states, weight_vals)
    assign_state_values(self.hidden_states, hidden_vals)
    assign_state_values(self.other_states, othstate_vals)

    weight_id_to_its_val = {id(st): val for st, val in zip(self.param_states, weight_vals)}

    # necessary gradients of the weights
    out, hiddens, oth_states, hid2weight_jac, data_for_hid2hid_jac, residuals = (
      self.graph.solve_h2w_h2h_jacobian_and_l2h_vjp(*args)
    )
    hid2weight_jac = jax.lax.stop_gradient(hid2weight_jac)

    # gradients for the diagonal hidden Jacobian matrix
    if self.diag_jacobian != _DiagJacobian.exact:
      data_for_hid2hid_jac = _solve_diagonal_jacobian_by_vjp_or_jvp(self, *args)

    # eligibility trace update
    new_etrace_vals = self._update_etrace_data(
      running_index,
      etrace_vals,
      hid2weight_jac,
      data_for_hid2hid_jac,
      weight_id_to_its_val,
    )

    # returns
    fwd_out = (out, hiddens, oth_states, new_etrace_vals)
    fwd_res = (residuals,
               new_etrace_vals if self.vjp_time == _VJPTime.t else etrace_vals,
               weight_id_to_its_val,
               running_index)
    return fwd_out, fwd_res

  def _update_bwd(
      self,
      fwd_res,
      grads,
  ) -> Tuple[dG_Inputs, dG_Weight, dG_Hidden, dG_State, None, None]:
    # ----------------------------------------------------------------------------------------------
    #
    # The backward function to compute the VJP gradients
    #
    # There are three steps:
    #
    # 1. Interpret the forward results and top-down gradients
    # 2. Compute the original gradients
    # 3. Compute the gradients of the weights
    # ----------------------------------------------------------------------------------------------

    # [1] Interpret the fwd results
    #
    residuals, etrace_vals_at_t, id2weight_val, running_index = fwd_res
    jaxpr, in_tree, out_tree, consts = residuals
    if running_index is None:
      raise ValueError('The running index should be provided. Please call the etrace model using: \n\n'
                       '>>> etrace(*args, running_index=i)\n')

    # [1] Interpret the top-down gradient signals
    #
    # Since
    #
    #     dg_out, dg_hiddens, dg_others, dg_etrace = grads
    #
    # we need to remove the "dg_etrace" iterm from the gradients for matching
    # the jaxpr vjp gradients.
    grad_flat, grad_tree = jax.tree.flatten((grads[:-1],))

    # [2] Compute the original gradients
    #
    # The original gradients are computed through the normal back-propagation algorithm.
    if out_tree != grad_tree:
      raise TypeError(
        f'Gradient tree should be the same as the function output tree. '
        f'While we got: \n'
        f'out_tree  = {out_tree}\n!=\n'
        f'grad_tree = {grad_tree}'
      )
    cts_out = jax.core.eval_jaxpr(jaxpr, consts, *grad_flat)

    if self.vjp_time == _VJPTime.t:
      # We compute:
      #   - the gradients of input arguments
      #   - the gradients of the hidden states at the last time step
      #   - the gradients of the non-etrace parameters
      #   - the gradients of the other states
      #   - the gradients of the loss-to-hidden at the current time step
      dg_args, dg_last_hiddens, dg_non_etrace_params, dg_oth_states, dl_to_dh_at_t = (
        jax.tree.unflatten(in_tree, cts_out)
      )
      dl_to_dh_at_t = {hid_var: dg for hid_var, dg in zip(self.graph.out_hidden_jaxvars, dl_to_dh_at_t)}
      dg_etrace_params = None

    elif self.vjp_time == _VJPTime.t_minus_1:
      # We compute:
      #   - the gradients of input arguments
      #   - the gradients of the hidden states at the last time step
      #   - the gradients of the non-etrace parameters
      #   - the gradients of the etrace parameters
      #   - the gradients of the other states
      dg_args, dg_last_hiddens, dg_non_etrace_params, dg_etrace_params, dg_oth_states = (
        jax.tree.unflatten(in_tree, cts_out)
      )
      # TODO: checking whether the correspondence is correct
      dl_to_dh_at_t = {hid_var: dg for hid_var, dg in zip(self.graph.out_hidden_jaxvars, dg_last_hiddens)}

    else:
      raise ValueError(f'The VJP time {self.vjp_time} is not supported. ')

    # [3] Compute the gradients of the weights
    #
    # the gradients of the weights are computed through the RTRL algorithm.
    dg_weights = self._solve_weight_gradients(
      running_index,
      etrace_vals_at_t,
      dl_to_dh_at_t,
      id2weight_val,
      dg_non_etrace_params,
      dg_etrace_params,
    )

    # Note that there are no gradients flowing through the etrace data and the running index.
    dg_etrace = None
    dg_running_index = None

    return (dg_args, dg_weights,
            dg_last_hiddens, dg_oth_states,
            dg_etrace, dg_running_index)

  def _solve_weight_gradients(
      self,
      running_index: Optional[int],
      etrace_h2w_at_t: Any,
      dl_to_dh_at_t: Dict[HiddenOutVar, jax.Array],
      id2weight_val: Dict[WeightID, PyTree],
      dl_to_nonetws_at_t: List[PyTree],
      dl_to_etws_at_t: Optional[List[PyTree]],
  ):
    r"""
    The method to solve the weight gradients, i.e., :math:`\partial L / \partial W`.

    Particularly, the weight gradients are computed through::

    .. math::

        \frac{\partial L^t}{\partial W} = \frac{\partial L^t}{\partial h^t} \frac{\partial h^t}{\partial W}

    Or,

    .. math::

        \frac{\partial L^t}{\partial W} = \frac{\partial L^{t-1}}{\partial h^{t-1}}
                                          \frac{\partial h^{t-1}}{\partial W}
                                          + \frac{\partial L^t}{\partial W^t}


    Args:
      running_index: Optional[int], the running index.
      etrace_h2w_at_t: Any, the eligibility trace data (which track the hidden-to-weight Jacobian)
          that have accumulated util the time ``t``.
      dl_to_dh_at_t: Dict[HiddenOutVar, jax.Array], the gradients of the loss-to-hidden
          at the time ``t``.
      id2weight_val: Dict[WeightID, PyTree], the weight values.
      dl_to_nonetws_at_t: List[PyTree], the gradients of the loss-to-non-etrace parameters
          at the time ``t``, i.e., :math:``\partial L^t / \partial W^t``.
      dl_to_etws_at_t: List[PyTree], the gradients of the loss-to-etrace parameters
          at the time ``t``, i.e., :math:``\partial L^t / \partial W^t``.
    """
    raise NotImplementedError

  def _update_etrace_data(
      self,
      running_index: Optional[int],
      etrace_vals_util_t_1: ETraceVals,
      hid2weight_jac_at_t: ETraceVals,
      data_for_hid2hid_jac_at_t,
      weight_id_to_its_val: Dict[WeightID, PyTree],
  ) -> ETraceVals:
    """
    The method to update the eligibility trace data.

    Args:
      running_index: Optional[int], the running index.
      etrace_vals_util_t_1: ETraceVals, the history eligibility trace data that have accumulated util :math:`t-1`.
      hid2weight_jac_at_t: ETraceVals, the current eligibility trace data at the time :math:`t`.
      data_for_hid2hid_jac_at_t: The data for computing the hidden-to-hidden Jacobian at the time :math:`t`.
      weight_id_to_its_val: Dict[WeightID, PyTree], the weight values.

    Returns:
      ETraceVals, the updated eligibility trace data that have accumulated util :math:`t`.
    """
    raise NotImplementedError

  def _get_etrace_data(self) -> ETraceVals:
    """
    Get the eligibility trace data at the last time-step.

    Returns:
      ETraceVals, the eligibility trace data.
    """
    raise NotImplementedError

  def _assign_etrace_data(self, etrace_vals: ETraceVals) -> None:
    """
    Assign the eligibility trace data to the states at the current time-step.

    Args:
      etrace_vals: ETraceVals, the eligibility trace data.
    """
    raise NotImplementedError


class _OnAlgorithm(Protocol):
  num_snap: int
  etrace_xs: Dict[WeightXVar, bst.State]
  etrace_chunked_xs: Dict[WeightXVar, bst.State]
  etrace_dfs: Dict[Tuple[WeightYVar, HiddenOutVar], bst.State]
  etrace_chunked_dfs: Dict[Tuple[WeightYVar, HiddenOutVar], bst.State]


def _init_on_state(
    self: _OnAlgorithm,
    relation: HiddenWeightOpRelation,
):
  if relation.x not in self.etrace_xs:
    shape = relation.x.aval.shape
    dtype = relation.x.aval.dtype
    self.etrace_xs[relation.x] = bst.ShortTermState(jnp.zeros(shape, dtype))
    if self.num_snap > 0:
      self.etrace_chunked_xs[relation.x] = bst.ShortTermState(jnp.zeros((self.num_snap + 1,) + shape, dtype))
  for hidden_var in relation.hidden_vars:
    key = (relation.y, hidden_var)
    if key in self.etrace_dfs:
      raise ValueError(f'The relation {key} has been added. ')
    shape = relation.y.aval.shape
    dtype = relation.y.aval.dtype
    self.etrace_dfs[key] = bst.ShortTermState(jnp.zeros(shape, dtype))
    if self.num_snap > 0:
      self.etrace_chunked_dfs[key] = bst.ShortTermState(jnp.zeros((self.num_snap + 1,) + shape, dtype))


def _update_on_etrace_with_exact_jac(
    hist_etrace_vals: Tuple,
    hid2weight_jac: Tuple[Dict, Dict],
    data_for_hid2hid_jac: Dict,
    hid_weight_op_relations: List,
    hid_group_relations: Dict,
    hidden_outvar_to_invar: Dict,
    running_index: int,
    decay: float,
    num_snap: int,
    snap_freq: int
):
  # --- the data --- #

  #
  # the etrace data at the current time step (t) of the O(n) algorithm
  # is a tuple, including the weight x and df values.
  #
  # For the weight x, it is a dictionary,
  #    {WeightXVar: brainstate.State}
  # For the weight df, it is a dictionary,
  #    {(WeightYVar, HiddenOutVar): brainstate.State}
  #
  xs, dfs = hid2weight_jac

  #
  # the history etrace values
  #
  # - hist_xs: {WeightXVar: brainstate.State}
  # - hist_dfs: {(WeightYVar, HiddenOutVar): brainstate.State}
  # - hist_chunk_xs: {WeightXVar: brainstate.State}
  # - hist_chunk_dfs: {(WeightYVar, HiddenOutVar): brainstate.State}
  #
  hist_xs, hist_dfs, hist_chunk_xs, hist_chunk_dfs = hist_etrace_vals

  # the new etrace values
  new_etrace_xs, new_etrace_dfs, new_etrace_chunk_xs, new_etrace_chunk_dfs = dict(), dict(), dict(), dict()

  # --- the update --- #

  # update the weight x
  for xkey in hist_xs.keys():
    new_etrace_xs[xkey] = low_pass_filter(hist_xs[xkey], xs[xkey], decay)

  # update the weight diagonal * df
  for hwo_relation in hid_weight_op_relations:
    hh_relation: HiddenGroupRelation = hid_group_relations[frozenset(hwo_relation.hidden_vars)]
    input_vals = [data_for_hid2hid_jac[var] for var in hh_relation.input_vars]
    primals = [data_for_hid2hid_jac[hidden_outvar_to_invar[outvar]]
               for outvar in hwo_relation.hidden_vars]
    tangents = [hist_dfs[(hwo_relation.y, outvar)] for outvar in hwo_relation.hidden_vars]
    #
    # JVP equation for the following Jacobian computation:
    #
    # [∂V^t/∂V^t-1, ∂V^t/∂a^t-1,  [∂V^t-1/∂θ1,
    #  ∂a^t/∂V^t-1, ∂a^t/∂a^t-1]   ∂a^t-1/∂θ1,]
    #
    # [∂V^t/∂V^t-1, ∂V^t/∂a^t-1,  [∂V^t-1/∂θ2,
    #  ∂a^t/∂V^t-1, ∂a^t/∂a^t-1]   ∂a^t-1/∂θ2]
    #
    fun = partial(hh_relation.state_transition, input_vals=input_vals)
    _, new_grads = jax.jvp(fun, (primals,), (tangents,))
    for i, outvar in enumerate(hwo_relation.hidden_vars):
      dfkey = (hwo_relation.y, outvar)
      new_etrace_dfs[dfkey] = new_grads[i]

  # update the weight df + new hidden df
  for dfkey in hist_dfs.keys():
    new_etrace_dfs[dfkey] = expon_smooth(new_etrace_dfs[dfkey], dfs[dfkey], decay)

  # -- the chunked etrace values -- #

  if num_snap > 0:

    for hwo_relation in hid_weight_op_relations:
      hh_relation: HiddenGroupRelation = hid_group_relations[frozenset(hwo_relation.hidden_vars)]
      input_vals = [data_for_hid2hid_jac[var] for var in hh_relation.input_vars]
      primals = [data_for_hid2hid_jac[hidden_outvar_to_invar[outvar]]
                 for outvar in hwo_relation.hidden_vars]
      tangents = [hist_chunk_dfs[(hwo_relation.y, outvar)] for outvar in hwo_relation.hidden_vars]
      #
      # JVP equation for the following Jacobian computation:
      #
      # [∂V^t/∂V^t-1, ∂V^t/∂a^t-1,  [∂V^t-1/∂θ1,
      #  ∂a^t/∂V^t-1, ∂a^t/∂a^t-1]   ∂a^t-1/∂θ1,]
      #
      # [∂V^t/∂V^t-1, ∂a^t/∂a^t-1,  [∂V^t-1/∂θ2,
      #  ∂a^t/∂V^t-1, ∂a^t/∂a^t-1]   ∂a^t-1/∂θ2]
      #
      fun = partial(hh_relation.state_transition, input_vals=input_vals)
      new_grads = jax.vmap(lambda tang: jax.jvp(fun, (primals,), (tang,))[1])(tangents)
      for i, outvar in enumerate(hwo_relation.hidden_vars):
        dfkey = (hwo_relation.y, outvar)
        hist_chunk_dfs[dfkey] = new_grads[i]

    def at_chunked_index(old_chunk_xs, old_chunk_dfs, new_xs, new_dfs):
      new_chunk_xs = dict()
      for x in new_xs:
        new_chunk_xs[x] = jnp.concatenate(
          [old_chunk_xs[x][1:], jnp.expand_dims(new_xs[x], axis=0)],
          axis=0
        )
      new_chunk_dfs = dict()
      for dfkey in new_dfs:
        new_chunk_dfs[dfkey] = jnp.concatenate(
          [old_chunk_dfs[dfkey][1:], jnp.expand_dims(new_dfs[dfkey], axis=0)],
          axis=0
        )
      return new_chunk_xs, new_chunk_dfs

    def not_chunked_index(old_chunk_xs, old_chunk_dfs, new_xs, new_dfs):
      return old_chunk_xs, old_chunk_dfs

    new_etrace_chunk_xs, new_etrace_chunk_dfs = jax.lax.cond(
      running_index % snap_freq == 0,
      at_chunked_index,
      not_chunked_index,
      hist_chunk_xs, hist_chunk_dfs, new_etrace_xs, new_etrace_dfs
    )

  return new_etrace_xs, new_etrace_dfs, new_etrace_chunk_xs, new_etrace_chunk_dfs


def _update_on_etrace_with_jvp_or_vjp_jac(
    hist_etrace_vals: Tuple,
    hid2weight_jac: Tuple[Dict, Dict],
    temporal_jacobian,
    diag_normalize: bool,
    running_index: int,
    decay: float,
    num_snap: int,
    snap_freq: int
):
  # Normalize the temporal Jacobian so that the temporal gradients are
  # within [-1., 1.], preventing the gradient vanishing and exploding.
  if diag_normalize:
    temporal_jacobian = jax.tree.map(_normalize, temporal_jacobian)

  # --- the data --- #

  # the etrace data at the current time step (t) of the O(n) algorithm
  # is a tuple, including the weight x and df values.
  xs, dfs = hid2weight_jac

  # the history etrace values
  hist_xs, hist_dfs, hist_chunk_xs, hist_chunk_dfs = hist_etrace_vals

  # the new etrace values
  new_etrace_xs, new_etrace_dfs, new_etrace_chunk_xs, new_etrace_chunk_dfs = dict(), dict(), dict(), dict()

  # --- the update --- #

  # update the weight x
  for xkey in hist_xs.keys():
    new_etrace_xs[xkey] = low_pass_filter(hist_xs[xkey], xs[xkey], decay)
  # update the weight df * diagonal
  for dfkey in hist_dfs.keys():
    df_var, hidden_var = dfkey
    new_etrace_dfs[dfkey] = hist_dfs[dfkey] * temporal_jacobian[hidden_var]
  # update the weight df
  for dfkey in hist_dfs.keys():
    new_etrace_dfs[dfkey] = expon_smooth(new_etrace_dfs[dfkey], dfs[dfkey], decay)

  # -- the chunked etrace values -- #

  if num_snap > 0:

    for dfkey in hist_chunk_dfs.keys():
      df_var, hidden_var = dfkey
      hist_chunk_dfs[dfkey] = hist_chunk_dfs[dfkey] * jnp.expand_dims(temporal_jacobian[hidden_var], axis=0)

    def at_chunked_index(old_chunk_xs, old_chunk_dfs, new_xs, new_dfs):
      new_chunk_xs = dict()
      for x in new_xs:
        new_chunk_xs[x] = jnp.concatenate([old_chunk_xs[x][1:], jnp.expand_dims(new_xs[x], axis=0)],
                                          axis=0)
      new_chunk_dfs = dict()
      for dfkey in new_dfs:
        new_chunk_dfs[dfkey] = jnp.concatenate([old_chunk_dfs[dfkey][1:], jnp.expand_dims(new_dfs[dfkey], axis=0)],
                                               axis=0)
      return new_chunk_xs, new_chunk_dfs

    def not_chunked_index(old_chunk_xs, old_chunk_dfs, new_xs, new_dfs):
      return old_chunk_xs, old_chunk_dfs

    new_etrace_chunk_xs, new_etrace_chunk_dfs = jax.lax.cond(
      running_index % snap_freq == 0,
      at_chunked_index,
      not_chunked_index,
      hist_chunk_xs, hist_chunk_dfs, new_etrace_xs, new_etrace_dfs
    )

  return new_etrace_xs, new_etrace_dfs, new_etrace_chunk_xs, new_etrace_chunk_dfs


def _solve_on_weight_gradients(
    hist_etrace_data,
    dG_weights: Dict[WeightID, dG_Weight],
    dG_hiddens: Dict[HiddenOutVar, jax.Array],
    weight_hidden_relations,
    weight_id_to_its_val,
    running_index: int,
    decay: float,
    num_snap: int
):
  # avoid the exponential smoothing bias at the beginning
  correction_factor = 1 - jnp.power(1 - decay, running_index + 1)

  xs, dfs, chunk_xs, chunk_dfs = hist_etrace_data

  for relation in weight_hidden_relations:
    x = xs[relation.x]
    weight_id = id(relation.weight)
    #
    # Solve the weight gradients by using the etrace data
    #
    #   dw = (dL/dH \circ df) \otimes x
    #
    for i, hid_var in enumerate(relation.hidden_vars):
      df = dfs[(relation.y, hid_var)] / correction_factor  # the hidden gradients
      df_hid = df * relation.hidden2df[i](dG_hiddens[hid_var])  # the hidden gradients
      dg_weight = weight_op_gradient(relation.op_jaxpr, x, weight_id_to_its_val[weight_id], df_hid)
      update_dict(dG_weights, weight_id, dg_weight)  # update the weight gradients

    if num_snap > 0:
      chunk_x = chunk_xs[relation.x][:-1]  # batched x, ignore the last one
      fun = lambda x, df: weight_op_gradient(relation.op_jaxpr, x, weight_id_to_its_val[weight_id], df)
      for i, hid_var in enumerate(relation.hidden_vars):
        #
        # Solve the weight gradients by the batched / chunked etrace data
        #
        #  dw = \sum_i (dL/dH \circ df_i) \otimes x_i
        #
        chunk_df_hid = (chunk_dfs[(relation.y, hid_var)][:-1]  # batched df, ignore the last one
                        * jnp.expand_dims(relation.hidden2df[i](dG_hiddens[hid_var]), axis=0))
        chunk_df_hid = chunk_df_hid / correction_factor
        chunk_dg_weight = jax.vmap(fun)(chunk_x, chunk_df_hid)  # batched weight gradients
        dg_weight = jax.tree.map(lambda a: jnp.sum(a, axis=0), chunk_dg_weight)  # sum over the batch dimension
        update_dict(dG_weights, weight_id, dg_weight)  # update the weight gradients


class DiagIODimAlgorithm(DiagETraceAlgorithmForVJP):
  r"""
  The online gradient computation algorithm with the diagonal approximation and the input-output dimensional complexity.

  This algorithm has the :math:`O(BI+BO)` memory complexity and :math:`O(BIO)` computational
  complexity, where :math:`I` and :math:`O` are the number of input and output dimensions, and
  :math:`B` the batch size.

  Particularly, for a Linear transformation layer, the algorithm computes the weight gradients
  with the :math:`O(Bn)` memory complexity and :math:`O(Bn^2)` computational complexity, where
  :math:`n` is the number of hidden dimensions.

  Parameters:
  -----------
  {common}
  {io_dim_doc}
  """

  __module__ = 'brainscale'

  etrace_xs: Dict[WeightXVar, bst.State]  # the spatial gradients of the weights
  etrace_dfs: Dict[Tuple[WeightYVar, HiddenOutVar], bst.State]  # the spatial gradients of the hidden states
  etrace_chunked_xs: Dict[WeightXVar, bst.State]  # the chunked spatial gradients of the weights
  etrace_chunked_dfs: Dict[Tuple[WeightYVar, HiddenOutVar], bst.State]  # the chunked hidden spatial gradients
  decay: float  # the exponential smoothing decay factor
  num_snap: int  # the number of snap shoot
  snap_freq: int  # the frequency of the snap shoot

  def __init__(
      self,
      model: Callable,
      decay_or_rank: float | int,
      num_snap: int = 0,
      snap_freq: Optional[int] = None,
      diag_normalize: Optional[bool] = None,
      diag_jacobian: str = 'exact',
      vjp_time: str = 't',
      name: Optional[str] = None,
      mode: Optional[bst.mixin.Mode] = None
  ):
    super().__init__(
      model,
      diag_jacobian=diag_jacobian,
      diag_normalize=diag_normalize,
      vjp_time=vjp_time,
      name=name,
      mode=mode
    )

    # the learning parameters
    self.decay, num_rank = _format_decay_and_rank(decay_or_rank)
    assert isinstance(num_snap, int) and num_snap >= 0, (f'The number of snap shoot should be '
                                                         f'greater than 0. While we got {num_snap}. ')
    self.num_snap = num_snap
    self.snap_freq = snap_freq or num_rank

  def init_etrace_state(self, *args, **kwargs):
    # The states of weight spatial gradients:
    #   1. x
    #   2. df
    self.etrace_xs = bst.visible_state_dict()
    self.etrace_dfs = bst.visible_state_dict()
    self.etrace_chunked_xs = bst.visible_state_dict()
    self.etrace_chunked_dfs = bst.visible_state_dict()
    for relation in self.graph.hidden_param_op_relations:
      _init_on_state(self, relation)

  def _get_etrace_data(self) -> Tuple:
    etrace_xs = {k: v.value for k, v in self.etrace_xs.items()}
    etrace_dfs = {k: v.value for k, v in self.etrace_dfs.items()}
    etrace_chunked_xs = {k: v.value for k, v in self.etrace_chunked_xs.items()}
    etrace_chunked_dfs = {k: v.value for k, v in self.etrace_chunked_dfs.items()}
    return etrace_xs, etrace_dfs, etrace_chunked_xs, etrace_chunked_dfs

  def _assign_etrace_data(self, hist_etrace_vals):
    etrace_xs, etrace_dfs, etrace_chunked_xs, etrace_chunked_dfs = hist_etrace_vals

    # the weight x and df
    for x, val in etrace_xs.items():
      self.etrace_xs[x].value = val
    for dfkey, val in etrace_dfs.items():
      self.etrace_dfs[dfkey].value = val

    # the chunked weight x and df
    if self.num_snap > 0:
      for x, val in etrace_chunked_xs.items():
        self.etrace_chunked_xs[x].value = val
      for dfkey, val in etrace_chunked_dfs.items():
        self.etrace_chunked_dfs[dfkey].value = val

  def _update_etrace_data(
      self,
      hist_etrace_vals: PyTree,
      hid2weight_jac_at_t: Tuple[Dict[WeightXVar, jax.Array], Dict[Tuple[WeightYVar, HiddenOutVar], jax.Array]],
      data_for_hid2hid_jac_at_t: Dict,
      weight_id_to_its_val: Dict[WeightID, PyTree],
      running_index: Optional[int]
  ) -> ETraceVals:

    if self.diag_jacobian == _DiagJacobian.exact:
      return _update_on_etrace_with_exact_jac(
        hist_etrace_vals,
        hid2weight_jac_at_t,
        data_for_hid2hid_jac_at_t,
        self.graph.hidden_param_op_relations,
        self.graph.hidden_group_relations,
        self.graph.hidden_outvar_to_invar,
        running_index,
        decay=self.decay,
        num_snap=self.num_snap,
        snap_freq=self.snap_freq
      )

    else:
      return _update_on_etrace_with_jvp_or_vjp_jac(
        hist_etrace_vals,
        hid2weight_jac_at_t,
        data_for_hid2hid_jac_at_t,
        self.diag_normalize,
        running_index,
        decay=self.decay,
        num_snap=self.num_snap,
        snap_freq=self.snap_freq
      )

  def _solve_weight_gradients(
      self,
      running_index: int,
      etrace_h2w_at_t: Tuple,
      dl_to_dh_at_t: Dict[HiddenOutVar, jax.Array],
      weight_id_to_its_val: Dict[WeightID, PyTree],
      dl_to_nonetws_at_t: List[PyTree],
      dl_to_etws_at_t: Optional[List[PyTree]],
  ):
    """
    See the documentation in the super class for the details.
    """
    dG_weights = {id(st): None for st in self.param_states}

    # update the etrace parameters
    _solve_on_weight_gradients(
      etrace_h2w_at_t,
      dG_weights,
      dl_to_dh_at_t,
      self.graph.hidden_param_op_relations,
      weight_id_to_its_val,
      running_index,
      self.decay,
      self.num_snap,
    )

    # update the non-etrace parameters
    etrace_params, _, non_etrace_params, _ = split_states_v2(self.graph.states)
    for st, dg in zip(non_etrace_params, dl_to_nonetws_at_t):
      update_dict(dG_weights, id(st), dg)

    # update the etrace parameters when "dl_to_etws_at_t" is not None
    if dl_to_etws_at_t is not None:
      for st, dg in zip(etrace_params, dl_to_etws_at_t):
        update_dict(dG_weights, id(st), dg, error_when_no_key=True)
    return list(dG_weights.values())


class _On2Algorithm(Protocol):
  etrace_bwg: Dict[Tuple[WeightID, WeightXVar, HiddenOutVar], bst.State]
  mode: bst.mixin.Mode


def _init_on2_state(
    self: _On2Algorithm,
    relation: HiddenWeightOpRelation
):
  # TODO: assume the batch size is the first dimension
  batch_size = relation.x.aval.shape[0] if self.mode.has(bst.mixin.Batching) else None
  for hidden_var in relation.hidden_vars:
    key = (id(relation.weight), relation.x, hidden_var)
    if key in self.etrace_bwg:
      raise ValueError(f'The relation {key} has been added. ')
    self.etrace_bwg[key] = bst.ShortTermState(
      jax.tree.map(partial(batched_zeros_like, batch_size), relation.weight.value)
    )


def _update_on2_etrace_with_exact_jac(
    hist_etrace_vals: Dict,
    hid2weight_jac: Tuple,
    data_for_hid2hid_jac: Dict,
    weight_id_to_its_val: Dict,
    hidden_param_op_relations,
    hid_group_relations,
    hidden_outvar_to_invar: Dict,
    diag_normalize: bool,
    mode: bst.mixin.Mode
):
  #
  # + "hist_etrace_vals" has the following structure:
  #    - key: the weight id, the weight-x jax var, the hidden state var
  #    - value: the batched weight gradients
  #
  # + "hid2weight_jac_at_t" has the following structure:
  #    - a dict of weight x gradients
  #       * key: the weight x jax var
  #       * value: the weight x gradients
  #    - a dict of weight y gradients
  #       * key: the tuple of the weight y jax var and the hidden state jax var
  #       * value: the weight y gradients
  #

  cur_etrace_xs, cur_etrace_ys = hid2weight_jac

  global_diag_jacobian = dict()

  new_etrace_bwg = dict()
  for relation in hidden_param_op_relations:
    if not isinstance(relation.weight, ETraceParamOp):
      raise NotImplementedError(
        f'When using {DiagParamDimAlgorithm.__name__} or {DiagHybridDimAlgorithm.__name__} algorithms, '
        f'the weight should be an {ETraceParamOp.__name__}. '
      )

    group = frozenset(relation.hidden_vars)
    if group not in global_diag_jacobian:
      global_diag_jacobian[group] = dict()
      hh_relation: HiddenGroupRelation = hid_group_relations[group]
      input_vals = [data_for_hid2hid_jac[var] for var in hh_relation.input_vars]
      primals = [data_for_hid2hid_jac[hidden_outvar_to_invar[outvar]] for outvar in relation.hidden_vars]

      for i, hid_var in enumerate(relation.hidden_vars):
        #
        # VJP equation for the following Jacobian computation:
        #
        # [∂V^t/∂V^t-1, ∂V^t/∂a^t-1, ∂V^t/∂g^t-1,]
        #
        jac = vgrad(hh_relation.state_transition, argnums=0)(
          primals, input_vals=input_vals, return_index=i
        )
        global_diag_jacobian[group][hid_var] = jac

    # get the stored diagonal Jacobian
    diag_jac = global_diag_jacobian[group]

    # weight information
    weight_id = id(relation.weight)
    weight_vals = weight_id_to_its_val[weight_id]

    # previous gradients
    old_grads = [hist_etrace_vals[(weight_id, relation.x, hid_var)]
                 for hid_var in relation.hidden_vars]

    # the etrace operation for computing etrace updates
    if isinstance(relation.weight.op, StandardETraceOp):
      etrace_op = relation.weight.op
    else:
      etrace_op = GeneralETraceOp(relation.weight.op,
                                  xinfo=jax.ShapeDtypeStruct(relation.x.aval.shape, relation.x.aval.dtype), )

    # update the etrace weight gradients
    for i, hid_var in enumerate(relation.hidden_vars):
      new_bwg = etrace_op.etrace_update(
        mode,
        weight_vals,
        old_grads,
        diag_jac[hid_var],
        cur_etrace_xs[relation.x],
        cur_etrace_ys[(relation.y, hid_var)]
      )

      w_key = (weight_id, relation.x, hid_var)
      new_etrace_bwg[w_key] = new_bwg
  return new_etrace_bwg


def _update_on2_etrace_with_jvp_or_vjp_jac(hist_etrace_vals: Dict,
                                           hid2weight_jac: Tuple,
                                           temporal_jacobian: Dict,
                                           weight_id_to_its_val: Dict,
                                           weight_hidden_relations,
                                           diag_normalize: bool,
                                           mode: bst.mixin.Mode):
  #
  # 1. "temporal_jacobian" has the following structure:
  #    - key: the hidden state jax var
  #    - value: the hidden state jacobian gradients
  #
  # 2. "hist_etrace_vals" has the following structure:
  #    - key: the weight id, the weight-x jax var, the hidden state var
  #    - value: the batched weight gradients
  #
  # 3. "hid2weight_jac_at_t" has the following structure:
  #    - a dict of weight x gradients
  #       * key: the weight x jax var
  #       * value: the weight x gradients
  #    - a dict of weight y gradients
  #       * key: the tuple of the weight y jax var and the hidden state jax var
  #       * value: the weight y gradients

  cur_etrace_xs, cur_etrace_ys = hid2weight_jac

  new_etrace_bwg = dict()
  for relation in weight_hidden_relations:
    if not isinstance(relation.weight, ETraceParamOp):
      raise NotImplementedError('The weight should be an ETraceParamOp. ')

    weight_id = id(relation.weight)
    weight_vals = weight_id_to_its_val[weight_id]
    for i, hid_var in enumerate(relation.hidden_vars):
      w_key = (weight_id, relation.x, hid_var)
      y_key = (relation.y, hid_var)
      dg_hidden = relation.hidden2df[i](temporal_jacobian[hid_var])
      if isinstance(relation.weight.op, StandardETraceOp):
        if diag_normalize:
          # normalize the temporal Jacobian so that the maximum value is 1.,
          # preventing the gradient vanishing and exploding
          dg_hidden = _normalize(dg_hidden)
        new_bwg = relation.weight.op.etrace_update(mode,
                                                   weight_vals,
                                                   [hist_etrace_vals[w_key]],
                                                   [dg_hidden],
                                                   cur_etrace_xs[relation.x],
                                                   cur_etrace_ys[y_key])

      else:
        x = relation.x
        dg_weight = GeneralETraceOp._dy_to_weight(
          mode,
          weight_vals,
          relation.weight.op,
          jax.ShapeDtypeStruct(x.aval.shape, x.aval.dtype),
          dg_hidden
        )
        if diag_normalize:
          # normalize the temporal Jacobian so that the maximum value is 1.,
          # preventing the gradient vanishing and exploding
          dg_weight = _normalize(dg_weight)
        current_etrace = GeneralETraceOp._dx_dy_to_weight(
          mode,
          weight_vals,
          relation.weight.op,
          cur_etrace_xs[relation.x],
          cur_etrace_ys[y_key]
        )
        new_bwg = jax.tree.map(lambda old, jac, new: old * jac + new,
                               hist_etrace_vals[w_key],
                               dg_weight,
                               current_etrace)
      new_etrace_bwg[w_key] = new_bwg
  return new_etrace_bwg


def _solve_on2_weight_gradients(
    hist_etrace_data: Dict,
    dG_weights: Dict[WeightID, dG_Weight],
    dG_hiddens: Dict[HiddenOutVar, jax.Array],
    weight_hidden_relations,
    weight_id_to_its_val,
    mode: bst.mixin.Mode
):
  # update the etrace weight gradients
  temp_data = dict()
  for relation in weight_hidden_relations:
    #
    # the weight information
    #
    weight_id = id(relation.weight)
    weight_vals = weight_id_to_its_val[weight_id]

    #
    # the etrace operation for computing etrace updates
    #
    if isinstance(relation.weight.op, StandardETraceOp):
      etrace_op = relation.weight.op
    else:
      etrace_op = GeneralETraceOp(
        relation.weight.op,
        xinfo=jax.ShapeDtypeStruct(relation.x.aval.shape, relation.x.aval.dtype),
      )

    for i, hid_var in enumerate(relation.hidden_vars):
      key = (weight_id, relation.x, hid_var)
      #
      # dE/dH, computing the hidden to weight gradients
      #
      dg_hidden = relation.hidden2df[i](dG_hiddens[hid_var])
      #
      # dE/dW = dE/dH * dH/dW, computing the final weight gradients
      #
      dg_weight = etrace_op.hidden_to_etrace(
        mode,
        weight_vals,
        dg_hidden,
        hist_etrace_data[key],
      )

      # update the weight gradients
      update_dict(temp_data, weight_id, dg_weight)

  # sum up the batched weight gradients
  if mode.has(bst.mixin.Batching):
    for key, val in temp_data.items():
      temp_data[key] = jax.tree_map(lambda x: jnp.sum(x, axis=0), val)

  # update the weight gradients
  for key, val in temp_data.items():
    update_dict(dG_weights, key, val)


class DiagParamDimAlgorithm(DiagETraceAlgorithmForVJP):
  """
  The online gradient computation algorithm with the diagonal approximation and the parameter dimension complexity.

  This algorithm has the :math:`O(B\theta)` memory complexity, where :math:`\theta` is the number of parameters,
  and :math:`B` the batch size.

  For a convolutional layer, the algorithm computes the weight gradients with the :math:`O(B\theta)`
  memory complexity, where :math:`\theta` is the dimension of the convolutional kernel.

  For a Linear transformation layer, the algorithm computes the weight gradients with the :math:`O(BIO)``
  computational complexity, where :math:`I` and :math:`O` are the number of input and output dimensions.

  Parameters:
  -----------
  {common}
  """

  etrace_bwg: Dict[Tuple[WeightID, WeightXVar, HiddenOutVar], bst.State]  # batch of weight gradients

  def __init__(
      self,
      model: Callable,
      diag_jacobian: str = 'exact',
      diag_normalize: Optional[bool] = None,
      vjp_time: str = 't',
      name: Optional[str] = None,
      mode: Optional[bst.mixin.Mode] = None
  ):
    super().__init__(
      model,
      diag_jacobian=diag_jacobian,
      diag_normalize=diag_normalize,
      vjp_time=vjp_time,
      name=name,
      mode=mode
    )

  def init_etrace_state(self, *args, **kwargs):
    # The states of batched weight gradients
    self.etrace_bwg = bst.visible_state_dict()
    for relation in self.graph.hidden_param_op_relations:
      _init_on2_state(self, relation)

  def _get_etrace_data(self) -> Dict:
    return {k: v.value for k, v in self.etrace_bwg.items()}

  def _assign_etrace_data(self, etrace_vals: Dict) -> None:
    for x, val in etrace_vals.items():
      self.etrace_bwg[x].value = val

  def _update_etrace_data(
      self,
      hist_etrace_vals: Dict[Tuple[WeightID, WeightXVar, HiddenOutVar], PyTree],
      hid2weight_jac_at_t: Tuple[Dict[WeightXVar, jax.Array], Dict[Tuple[WeightYVar, HiddenOutVar], jax.Array]],
      data_for_hid2hid_jac_at_t,
      weight_id_to_its_val: Dict[WeightID, PyTree],
      running_index: Optional[int]
  ) -> Dict[Tuple[WeightID, WeightXVar, HiddenOutVar], PyTree]:
    if self.diag_jacobian == _DiagJacobian.exact:
      return _update_on2_etrace_with_exact_jac(
        hist_etrace_vals,
        hid2weight_jac_at_t,
        data_for_hid2hid_jac_at_t,
        weight_id_to_its_val,
        self.graph.hidden_param_op_relations,
        self.graph.hidden_group_relations,
        self.graph.hidden_outvar_to_invar,
        self.diag_normalize,
        self.mode
      )
    else:
      return _update_on2_etrace_with_jvp_or_vjp_jac(
        hist_etrace_vals,
        hid2weight_jac_at_t,
        data_for_hid2hid_jac_at_t,
        weight_id_to_its_val,
        self.graph.hidden_param_op_relations,
        self.diag_normalize,
        self.mode
      )

  def _solve_weight_gradients(
      self,
      running_index: int,
      etrace_h2w_at_t: Dict,
      dl_to_dh_at_t: Dict[HiddenOutVar, jax.Array],
      weight_id_to_its_val: Dict[WeightID, PyTree],
      dl_to_nonetws_at_t: List[PyTree],
      dl_to_etws_at_t: Optional[List[PyTree]],
  ):
    """
    Solve the weight gradients according to the eligibility trace data.

    Particularly, for each weight, we compute its gradients according to the batched weight gradients.
    """
    dG_weights = {id(st): None for st in self.param_states}

    # update the etrace weight gradients
    _solve_on2_weight_gradients(
      etrace_h2w_at_t,
      dG_weights,
      dl_to_dh_at_t,
      self.graph.hidden_param_op_relations,
      weight_id_to_its_val,
      self.mode
    )

    # update the non-etrace weight gradients
    etrace_params, _, non_etrace_params, _ = split_states_v2(self.graph.states)
    for st, dg in zip(non_etrace_params, dl_to_nonetws_at_t):
      update_dict(dG_weights, id(st), dg)

    # update the etrace parameters when "dl_to_etws_at_t" is not None
    if dl_to_etws_at_t is not None:
      for st, dg in zip(etrace_params, dl_to_etws_at_t):
        update_dict(dG_weights, id(st), dg, error_when_no_key=True)
    return list(dG_weights.values())


def _numel(pytree: PyTree):
  return sum(jnp.size(x) for x in jax.tree_leaves(pytree))


def _is_weight_need_full_grad(
    relation: HiddenWeightOpRelation,
    mode: bst.mixin.Mode
):
  if isinstance(relation.weight, ETraceParamOp):
    if relation.weight.gradient == _ETraceGrad.full:
      return True

  batch_size = relation.x.shape[0] if mode.has(bst.mixin.Batching) else 1
  if _numel(relation.x) + _numel(relation.y) > batch_size * _numel(relation.weight.value):
    return True

  return False


class DiagHybridDimAlgorithm(DiagETraceAlgorithmForVJP):
  r"""
  The hybrid online gradient computation algorithm with the diagonal approximation and hybrid complexity.

  For a function :math:`O = f(I, \theta)`, where :math:`I` is the input, :math:`\theta` is the parameters,
  and :math:`O` is the output, the algorithm computes the weight gradients with the ``O(BI + BO)`` memory complexity
  when :math:`I + O < \theta`, or the ``O(B\theta)`` memory complexity when :math:`I + O \geq \theta`.

  This means that the algorithm combine the memory efficiency of the :py:class:`DiagParamDimAlgorithm` and the
  computational efficiency of the :py:class:`DiagIODimAlgorithm` together.

  Parameters:
  -----------
  {common}
  {io_dim_doc}
  """
  etrace_xs: Dict[WeightXVar, bst.State]  # the spatial gradients of the weights
  etrace_dfs: Dict[Tuple[WeightYVar, HiddenOutVar], bst.State]  # the spatial gradients of the hidden states
  etrace_chunked_xs: Dict[WeightXVar, bst.State]  # the chunked spatial gradients of the weights
  # the chunked spatial gradients of the hidden states
  etrace_chunked_dfs: Dict[Tuple[WeightYVar, HiddenOutVar], bst.State]
  # batch of weight gradients
  etrace_bwg: Dict[Tuple[WeightID, WeightXVar, HiddenOutVar], bst.State]
  decay: float  # the exponential smoothing decay factor

  def __init__(
      self,
      model: Callable,
      decay_or_rank: float | int,
      num_snap: int = 0,
      snap_freq: Optional[int] = None,
      diag_jacobian: str = 'exact',
      vjp_time: str = 't',
      diag_normalize: Optional[bool] = None,
      name: Optional[str] = None,
      mode: Optional[bst.mixin.Mode] = None
  ):
    super().__init__(
      model,
      diag_jacobian=diag_jacobian,
      diag_normalize=diag_normalize,
      vjp_time=vjp_time,
      name=name,
      mode=mode
    )

    # the learning parameters
    self.decay, num_rank = _format_decay_and_rank(decay_or_rank)
    assert isinstance(num_snap, int) and num_snap >= 0, (f'The number of snap shoot should be '
                                                         f'greater than 0. While we got {num_snap}. ')
    self.num_snap = num_snap
    self.snap_freq = snap_freq or num_rank

  def init_etrace_state(self, *args, **kwargs):
    #
    # The states of weight spatial gradients:
    #   1. x
    #   2. df
    #   3. batched weight gradients
    #
    self.etrace_xs = bst.visible_state_dict()
    self.etrace_dfs = bst.visible_state_dict()
    self.etrace_bwg = bst.visible_state_dict()
    self.etrace_chunked_xs = bst.visible_state_dict()
    self.etrace_chunked_dfs = bst.visible_state_dict()

    for relation in self.graph.hidden_param_op_relations:
      if isinstance(relation.weight, ETraceParamOp):
        if relation.weight.gradient == _ETraceGrad.full:
          #
          # When
          #     weight.gradient == _ETraceGrad.full
          #
          # the weights will be forced to use O(n^2) algorithm
          # to compute the eligibility trace.
          #
          _init_on2_state(self, relation)
          continue
        elif relation.weight.gradient == _ETraceGrad.approx:
          #
          # When
          #     weight.gradient == _ETraceGrad.approx
          #
          # the weights will be forced to use O(n) algorithm
          # to compute the eligibility trace.
          #
          _init_on_state(self, relation)
          continue

      batch_size = relation.x.shape[0] if self.mode.has(bst.mixin.Batching) else 1
      if _numel(relation.x) + _numel(relation.y) > batch_size * _numel(relation.weight.value):
        #
        # When the number of elements in the inputs and outputs are bigger than the weight number,
        # we will use the O(n^2) algorithm to compute the eligibility trace, since
        # storing the batched weight gradients will be less expensive.
        #
        _init_on2_state(self, relation)
      else:
        #
        # For most cases, we will use the O(n) algorithm to compute the eligibility trace.
        # Since the number of elements in input and output (I + O) is greatly less than the number
        # of elements in the weight (W = I * O).
        #
        _init_on_state(self, relation)

  def _get_etrace_data(self) -> Tuple[Dict, ...]:
    etrace_xs = {x: val.value for x, val in self.etrace_xs.items()}
    etrace_dfs = {x: val.value for x, val in self.etrace_dfs.items()}
    etrace_chunked_xs = {k: v.value for k, v in self.etrace_chunked_xs.items()}
    etrace_chunked_dfs = {k: v.value for k, v in self.etrace_chunked_dfs.items()}
    etrace_wgrads = {x: val.value for x, val in self.etrace_bwg.items()}
    return etrace_xs, etrace_dfs, etrace_chunked_xs, etrace_chunked_dfs, etrace_wgrads

  def _assign_etrace_data(self, etrace_vals: Tuple[Dict, ...]) -> None:
    etrace_xs, etrace_dfs, etrace_chunked_xs, etrace_chunked_dfs, etrace_wgrads = etrace_vals
    for x, val in etrace_xs.items():
      self.etrace_xs[x].value = val
    for x, val in etrace_dfs.items():
      self.etrace_dfs[x].value = val
    for x, val in etrace_wgrads.items():
      self.etrace_bwg[x].value = val

    if self.num_snap > 0:
      for x, val in etrace_chunked_xs.items():
        self.etrace_chunked_xs[x].value = val
      for x, val in etrace_chunked_dfs.items():
        self.etrace_chunked_dfs[x].value = val

  def _update_etrace_data(
      self,
      hist_etrace_vals: Tuple[Dict, ...],
      hid2weight_jac_at_t: Tuple[Dict[WeightXVar, jax.Array], Dict[Tuple[WeightYVar, HiddenOutVar], jax.Array]],
      data_for_hid2hid_jac_at_t,
      weight_id_to_its_val: Dict[WeightID, PyTree],
      running_index: Optional[int]
  ) -> Tuple[Dict, ...]:

    # the history etrace values
    hist_xs, hist_dfs, hist_chunk_xs, hist_chunk_dfs, hist_bwg = hist_etrace_vals

    # ---- O(n^2) etrace gradients update ---- #
    on_weight_hidden_relations = []
    on2_weight_hidden_relations = []
    for relation in self.graph.hidden_param_op_relations:
      if _is_weight_need_full_grad(relation, self.mode):
        on2_weight_hidden_relations.append(relation)
      else:
        on_weight_hidden_relations.append(relation)

    if self.diag_jacobian == _DiagJacobian.exact:
      new_bwg = _update_on2_etrace_with_exact_jac(
        hist_bwg,
        hid2weight_jac_at_t,
        data_for_hid2hid_jac_at_t,
        weight_id_to_its_val,
        on2_weight_hidden_relations,
        self.graph.hidden_group_relations,
        self.graph.hidden_outvar_to_invar,
        self.diag_normalize,
        self.mode
      )
    else:
      new_bwg = _update_on2_etrace_with_jvp_or_vjp_jac(
        hist_bwg,
        hid2weight_jac_at_t,
        data_for_hid2hid_jac_at_t,
        weight_id_to_its_val,
        on2_weight_hidden_relations,
        self.diag_normalize,
        self.mode
      )

    # ---- O(n) etrace gradients update ---- #
    if self.diag_jacobian == _DiagJacobian.exact:
      new_xs, new_dfs, new_chunked_xs, new_chunked_dfs = (
        _update_on_etrace_with_exact_jac(
          hist_etrace_vals,
          hid2weight_jac_at_t,
          data_for_hid2hid_jac_at_t,
          on_weight_hidden_relations,
          self.graph.hidden_group_relations,
          self.graph.hidden_outvar_to_invar,
          running_index,
          decay=self.decay,
          num_snap=self.num_snap,
          snap_freq=self.snap_freq
        )
      )
    else:
      new_xs, new_dfs, new_chunked_xs, new_chunked_dfs = (
        _update_on_etrace_with_jvp_or_vjp_jac(
          hist_etrace_vals,
          hid2weight_jac_at_t,
          data_for_hid2hid_jac_at_t,
          self.diag_normalize,
          running_index,
          decay=self.decay,
          num_snap=self.num_snap,
          snap_freq=self.snap_freq
        )
      )

    return new_xs, new_dfs, new_chunked_xs, new_chunked_dfs, new_bwg

  def _solve_weight_gradients(
      self,
      running_index: int,
      etrace_h2w_at_t: Tuple,
      dl_to_dh_at_t: Dict[HiddenOutVar, jax.Array],
      weight_id_to_its_val: Dict[WeightID, PyTree],
      dl_to_nonetws_at_t: List[PyTree],
      dl_to_etws_at_t: Optional[List[PyTree]],
  ):
    """
    Solve the weight gradients according to the eligibility trace data.

    Particularly, for each weight, we compute its gradients according to the batched weight gradients.
    """

    xs, dfs, chunked_xs, chunked_dfs, wgrads = etrace_h2w_at_t
    dG_weights = {id(st): None for st in self.param_states}

    # weight-hidden relations
    on_weight_hidden_relations = []
    on2_weight_hidden_relations = []
    for relation in self.graph.hidden_param_op_relations:
      if _is_weight_need_full_grad(relation, self.mode):
        on2_weight_hidden_relations.append(relation)
      else:
        on_weight_hidden_relations.append(relation)

    # update the etrace weight gradients by the O(n) algorithm
    _solve_on_weight_gradients(
      [xs, dfs, chunked_xs, chunked_dfs],
      dG_weights,
      dl_to_dh_at_t,
      on_weight_hidden_relations,
      weight_id_to_its_val,
      running_index,
      self.decay,
      self.num_snap
    )

    # update the etrace weight gradients by the O(n^2) algorithm
    _solve_on2_weight_gradients(
      wgrads,
      dG_weights,
      dl_to_dh_at_t,
      on2_weight_hidden_relations,
      weight_id_to_its_val,
      self.mode
    )

    # update the non-etrace weight gradients
    etrace_params, _, non_etrace_params, _ = split_states_v2(self.graph.states)
    for st, dg in zip(non_etrace_params, dl_to_nonetws_at_t):
      update_dict(dG_weights, id(st), dg)

    # update the etrace parameters when "dl_to_etws_at_t" is not None
    if dl_to_etws_at_t is not None:
      for st, dg in zip(etrace_params, dl_to_etws_at_t):
        update_dict(dG_weights, id(st), dg, error_when_no_key=True)
    return list(dG_weights.values())


# -------------------------------------
#
# Update the model documentation
#

ETraceAlgorithm.__doc__ = ETraceAlgorithm.__doc__.format(
  common=_common_doc.strip()
)

DiagIODimAlgorithm.__doc__ = DiagIODimAlgorithm.__doc__.format(
  common=_common_doc.strip(),
  io_dim_doc=_io_dim_doc.strip(),
)

DiagParamDimAlgorithm.__doc__ = DiagParamDimAlgorithm.__doc__.format(
  common=_common_doc.strip()
)

DiagHybridDimAlgorithm.__doc__ = DiagHybridDimAlgorithm.__doc__.format(
  common=_common_doc.strip(),
  io_dim_doc=_io_dim_doc.strip(),
)
