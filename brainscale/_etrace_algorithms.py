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
from typing import Dict, Tuple, Any, Callable, List, Protocol, Optional, Sequence

import brainstate as bst
import brainunit as u
import jax.core

from ._etrace_compiler import (ETraceGraph,
                               HiddenWeightOpRelation,
                               HiddenGroup,
                               _VJPTime,
                               _compiler_docstr)
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
  {doc}
  name: str, optional
      The name of the etrace algorithm.
  mode: brainstate.mixin.Mode, optional
      The mode of the etrace algorithm. Note that the etrace algorithm is particularly sensitive to 
      ``brainstate.mixin.Batching``, since it is used to compute and initialize the eligibility trace 
      states with or without batch size.

'''.format(doc=_compiler_docstr)

_io_dim_doc = '''
  decay_or_rank: float, int
      The exponential smoothing factor for the eligibility trace. If it is a float,
      it is the decay factor, should be in the range of (0, 1). If it is an integer,
      it is the number of approximation rank for the algorithm, should be greater than 0.
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


def _weight_op_gradient(op_jaxpr, dx, w, dy):
  def op(xs, ws):
    return jax.core.eval_jaxpr(op_jaxpr, (), *jax.tree.leaves([xs, ws]))[0]

  return jax.vjp(partial(op, dx), w)[1](dy)[0]


def _expon_smooth(old, new, decay):
  """
  Exponential smoothing.

  :param old: the old value
  :param new: the new value
  :param decay: the decay factor
  :return: the smoothed value
  """
  if new is None:
    return decay * old
  return decay * old + (1 - decay) * new


def _low_pass_filter(old, new, alpha):
  """
  Low-pass filter.

  :param old: the old value
  :param new: the new value
  :param alpha: the filter factor
  :return: the filtered value
  """
  if new is None:
    return alpha * old
  return alpha * old + new


def _update_dict(
    the_dict: Dict,
    key: Any,
    value: PyTree,
    error_when_no_key: Optional[bool] = False
):
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
    the_dict[key] = jax.tree.map(u.math.add, old_value, value, is_leaf=lambda x: isinstance(x, u.Quantity))


def _batched_zeros_like(batch_size: Optional[int],
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
    return u.math.zeros_like(x)
  else:
    return u.math.zeros((batch_size,) + x.shape, x.dtype)


def _normalize_group(vals):
  """
  Normalize the input array to ``[-1, 1]``.

  Args:
    vals: jax.Array, the input array.

  Returns:
    jax.Array, the normalized array.
  """

  def _get_max(x):
    return u.math.max(u.math.abs(x))

  max_val = _get_max(u.math.asarray(jax.tree.leaves(jax.tree.map(_get_max, vals))))
  return jax.tree.map(lambda x: u.math.where(max_val <= 1., x, x / max_val), vals)


def _normalize_individual(vals):
  """
  Normalize the input array to ``[-1, 1]``.

  Args:
    vals: jax.Array, the input array.

  Returns:
    jax.Array, the normalized array.
  """

  def _normalize(x):
    max_val = u.math.max(u.math.abs(x))
    return u.math.where(max_val <= 1., x, x / max_val)

  return jax.tree.map(_normalize, vals)


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

  graph: ETraceGraph  # the etrace graph
  param_states: List[bst.ParamState]  # the weight states
  hidden_states: List[ETraceVar]  # the hidden states
  other_states: List[bst.State]  # the other states
  is_compiled: bool  # whether the etrace algorithm has been compiled
  diag_normalize: bool  # whether to normalize the hidden Jacobian diagonal matrix

  def __init__(
      self,
      model: Callable,
      diag_normalize: Optional[bool] = None,
      vjp_time: str = 't',
      name: Optional[str] = None,
      mode: Optional[bst.mixin.Mode] = None
  ):
    super().__init__(name=name, mode=mode)

    # The method to compute the hidden Jacobian diagonal matrix,
    # and whether to normalize the hidden Jacobian diagonal matrix
    self.diag_normalize = False if diag_normalize is None else diag_normalize

    # The time to compute the loss-to-hidden Jacobian
    self.vjp_time = _VJPTime.get(vjp_time)

    # The model and graph
    if not callable(model):
      raise ValueError(f'The model should be a callable function. But we got {model}.')
    self.graph = ETraceGraph(model, vjp_time=vjp_time)

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

  def get_etrace_of(self, weight: bst.ParamState | WeightID) -> Any:
    """
    Get the eligibility trace of the given weight.

    Parameters
    ----------
    weight: brainstate.ParamState, int
      The parameter weight.

    Returns
    -------
    out: Any
      The eligibility trace.
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


def _diag_hidden_update(
    self: 'ETraceAlgorithm',
    hiddens,
    others,
    *params
):
  # [ KEY ]  assuming the weight are not changed
  assign_state_values(self.hidden_states, hiddens)
  assign_state_values(self.other_states, others)
  with stop_param_gradients():
    self.graph.model(*params)
  hiddens = [st.value for st in self.hidden_states]
  return hiddens


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

  def _assert_compiled(self):
    if not self.is_compiled:
      raise ValueError('The etrace algorithm has not been compiled. Please call `compile_graph()` first. ')

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

    self._assert_compiled()

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
    out, hidden_vals, oth_state_vals, hid2weight_jac, hid2hid_jac = (
      self.graph.solve_h2w_h2h_jacobian(*inputs)
    )
    hid2weight_jac = jax.lax.stop_gradient(hid2weight_jac)

    # eligibility trace update
    etrace_vals = self._update_etrace_data(
      running_index,
      etrace_vals,
      hid2weight_jac,
      hid2hid_jac,
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

    weight_id_to_its_val = {
      id(st): val for st, val in zip(self.param_states, weight_vals)
    }

    # necessary gradients of the weights
    out, hiddens, oth_states, hid2weight_jac, hid2hid_jac, residuals = (
      self.graph.solve_h2w_h2h_jacobian_and_l2h_vjp(*args)
    )
    hid2weight_jac = jax.lax.stop_gradient(hid2weight_jac)

    # eligibility trace update
    new_etrace_vals = self._update_etrace_data(
      running_index,
      etrace_vals,
      hid2weight_jac,
      hid2hid_jac,
      weight_id_to_its_val,
    )

    # returns
    fwd_out = (out, hiddens, oth_states, new_etrace_vals)
    fwd_res = (
      residuals,
      new_etrace_vals if self.vjp_time == _VJPTime.t else etrace_vals,
      weight_id_to_its_val,
      running_index
    )
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
      assert len(self.graph.out_hidden_jaxvars) == len(dl_to_dh_at_t)
      dl_to_dh_at_t = {
        hid_var: dg for hid_var, dg in
        zip(self.graph.out_hidden_jaxvars, dl_to_dh_at_t)
      }
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
      assert len(self.graph.out_hidden_jaxvars) == len(dg_last_hiddens)
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

    return (dg_args,
            dg_weights,
            dg_last_hiddens,
            dg_oth_states,
            dg_etrace,
            dg_running_index)

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
      hid2hid_jac_at_t,
      weight_id_to_its_val: Dict[WeightID, PyTree],
  ) -> ETraceVals:
    """
    The method to update the eligibility trace data.

    Args:
      running_index: Optional[int], the running index.
      etrace_vals_util_t_1: ETraceVals, the history eligibility trace data that have accumulated util :math:`t-1`.
      hid2weight_jac_at_t: ETraceVals, the current eligibility trace data at the time :math:`t`.
      hid2hid_jac_at_t: The data for computing the hidden-to-hidden Jacobian at the time :math:`t`.
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


class _IODimAlgorithm(Protocol):
  etrace_xs: Dict[WeightXVar, bst.State]
  etrace_dfs: Dict[Tuple[WeightYVar, HiddenOutVar], bst.State]


def _init_IO_dim_state(
    self: _IODimAlgorithm,
    relation: HiddenWeightOpRelation,
):
  # For the relation
  #
  #   h1, h2, ... = f(x, w)
  #
  # we need to initialize the eligibility trace states for the weight x and the df.

  if relation.x not in self.etrace_xs:
    shape = relation.x.aval.shape
    dtype = relation.x.aval.dtype
    # wx may be repeatedly used in the graph
    self.etrace_xs[relation.x] = bst.ShortTermState(u.math.zeros(shape, dtype))

  for group in relation.hidden_groups:
    group: HiddenGroup
    #
    # Group 1:
    #
    #   [∂a^t-1/∂θ1, ∂b^t-1/∂θ1, ...]
    #
    # Group 2:
    #
    #   [∂A^t-1/∂θ1, ∂B^t-1/∂θ1, ...]
    #
    for hidden_outvar in group.hidden_outvars:
      key = (relation.y, hidden_outvar)
      if key in self.etrace_dfs:  # relation.y is an unique output of the weight operation
        raise ValueError(f'The relation {key} has been added. ')
      shape = relation.y.aval.shape
      dtype = relation.y.aval.dtype
      self.etrace_dfs[key] = bst.ShortTermState(u.math.zeros(shape, dtype))


def _update_IO_dim_etrace_with_exact_jac(
    hist_etrace_vals: Tuple,
    hid2weight_jac: Tuple[Dict, Dict],
    hid2hid_jac_at_t: Dict,
    hid_weight_op_relations: Sequence[HiddenWeightOpRelation],
    running_index: int,
    decay: float,
):
  # --- the data --- #

  #
  # the etrace data at the current time step (t) of the O(n) algorithm
  # is a tuple, including the weight x and df values.
  #
  # For the weight x, it is a dictionary,
  #    {WeightXVar: jax.Array}
  #
  # For the weight df, it is a dictionary,
  #    {(WeightYVar, HiddenOutVar): jax.Array}
  #
  xs, dfs = hid2weight_jac

  #
  # the history etrace values
  #
  # - hist_xs: {WeightXVar: brainstate.State}
  # - hist_dfs: {(WeightYVar, HiddenOutVar): brainstate.State}
  #
  hist_xs, hist_dfs = hist_etrace_vals

  #
  # the new etrace values
  new_etrace_xs, new_etrace_dfs = dict(), dict()

  # --- the update --- #

  #
  # Step 1:
  #
  #   update the weight x using the equation:
  #           x^t = α * x^t-1 + x^t, where α is the decay factor.
  #
  for xkey in hist_xs.keys():
    new_etrace_xs[xkey] = _low_pass_filter(hist_xs[xkey], xs[xkey], decay)

  for hwo_relation in hid_weight_op_relations:
    hwo_relation: HiddenWeightOpRelation

    for group in hwo_relation.hidden_groups:
      group: HiddenGroup
      #
      # Step 2:
      #
      # update the eligibility trace * hidden diagonal Jacobian
      #         dϵ^t = D_h ⊙ dϵ^t-1, where D_h is the hidden-to-hidden Jacobian diagonal matrix.
      #
      #
      # JVP equation for the following Jacobian computation:
      #
      # [∂V^t/∂V^t-1, ∂V^t/∂a^t-1,  [∂V^t-1/∂θ1,
      #  ∂a^t/∂V^t-1, ∂a^t/∂a^t-1]   ∂a^t-1/∂θ1,]
      #
      # [∂V^t/∂V^t-1, ∂V^t/∂a^t-1,  [∂V^t-1/∂θ2,
      #  ∂a^t/∂V^t-1, ∂a^t/∂a^t-1]   ∂a^t-1/∂θ2]
      #

      for hidden_outvar_at_t in group.hidden_outvars:
        #
        # computing the following vector-Jacobian product:
        #  df^t = D_h ⊙ df^t-1
        #
        # i.e., the hidden-to-hidden Jacobian diagonal matrix * the hidden df at the previous time step
        #
        #  ∂V^t/∂θ1 = ∂V^t/∂V^t-1 * ∂V^t-1/∂θ1 + ∂V^t/∂a^t-1 * ∂a^t-1/∂θ1 + ...
        #
        new_etrace = None
        for hidden_outvar_at_t_minus_1 in group.hidden_outvars:
          diag_jac_key = (hidden_outvar_at_t_minus_1, hidden_outvar_at_t)
          if diag_jac_key in hid2hid_jac_at_t:
            # diagonal Jacobian * hidden df
            data = hist_dfs[(hwo_relation.y, hidden_outvar_at_t_minus_1)] * hid2hid_jac_at_t[diag_jac_key]
            new_etrace = data if new_etrace is None else new_etrace + data
        assert new_etrace is not None, f'The new etrace should not be None. '

        #
        # Step 3:
        #
        # update: eligibility trace * hidden diagonal Jacobian + new hidden df
        #        dϵ^t = D_h ⊙ dϵ^t-1 + df^t, where D_h is the hidden-to-hidden Jacobian diagonal matrix.
        #
        new_df_key = (hwo_relation.y, hidden_outvar_at_t)
        new_etrace_dfs[new_df_key] = _expon_smooth(new_etrace, dfs.get(new_df_key, None), decay)

  return new_etrace_xs, new_etrace_dfs


def _solve_IO_dim_weight_gradients(
    hist_etrace_data,
    dG_weights: Dict[WeightID, dG_Weight],
    dG_hiddens: Dict[HiddenOutVar, jax.Array],
    weight_hidden_relations: Sequence[HiddenWeightOpRelation],
    weight_id_to_its_val: Dict[WeightID, PyTree],
    running_index: int,
    decay: float,
):
  #
  # Avoid the exponential smoothing bias at the beginning.
  # This is the correction factor for the exponential smoothing.
  correction_factor = 1. - u.math.power(1. - decay, running_index + 1)

  xs, dfs = hist_etrace_data

  for relation in weight_hidden_relations:
    x = xs[relation.x]
    weight_id = id(relation.weight)

    #
    # Function to compute the weight gradients
    # according to the inputs and df gradients
    #
    fun_dxy2dw = lambda dx, dy: _weight_op_gradient(relation.op_jaxpr, dx, weight_id_to_its_val[weight_id], dy)

    #
    # Solve the weight gradients by using the etrace data
    #
    #   dw = (dL/dH \circ df) \otimes x
    #
    for i, hid_var in enumerate(relation.hidden_vars):
      df = dfs[(relation.y, hid_var)] / correction_factor  # the hidden gradients
      df_hid = df * dG_hiddens[hid_var]  # the hidden gradients
      #
      # Compute the weight gradients according to the x and y
      #
      #    dw = df(dx, dy)
      #
      dg_weight = fun_dxy2dw(x, df_hid)
      _update_dict(dG_weights, weight_id, dg_weight)  # update the weight gradients


def _zeros_like_batch_or_not(
    batch_size: Optional[int],
    mode: bst.mixin.Mode,
    x: jax.Array
):
  """
  Create a batched zeros array like the input array.

  Args:
    batch_size: int, the batch size.
    mode: The computing mode.
    x: jax.Array, the input array.

  Returns:
    jax.Array, the batched zeros array.
  """
  if mode.has(bst.mixin.Batching):
    # TODO:
    #      Assuming the first axis is the batch dimension.
    assert batch_size is not None, 'The batch size should be provided. '
    return u.math.zeros((batch_size,) + x.shape[1:], x.dtype)
  else:
    return u.math.zeros_like(x)


def _reset_state_in_a_dict(
    state_dict: Dict[Any, bst.State],
    batch_size: Optional[int],
    mode: bst.mixin.Mode
):
  for k, v in state_dict.items():
    state_dict[k].value = jax.tree.map(partial(_zeros_like_batch_or_not, batch_size, mode), v)


class DiagTruncatedAlgorithm(DiagETraceAlgorithmForVJP):
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
  """

  __module__ = 'brainscale'

  etrace_xs: Dict[WeightXVar, bst.State]  # the spatial gradients of the weights
  etrace_dfs: Dict[Tuple[WeightYVar, HiddenOutVar], bst.State]  # the spatial gradients of the hidden states
  n_truncation: int  # the truncation length

  def __init__(
      self,
      model: Callable,
      n_truncation: int,
      diag_normalize: Optional[bool] = None,
      vjp_time: str = 't',
      name: Optional[str] = None,
      mode: Optional[bst.mixin.Mode] = None
  ):
    super().__init__(
      model,
      diag_normalize=diag_normalize,
      vjp_time=vjp_time,
      name=name,
      mode=mode
    )

    # the learning parameters
    assert isinstance(n_truncation, int), 'The truncation length should be an integer. '
    assert n_truncation > 0, 'The truncation length should be greater than 0. '
    self.n_truncation = n_truncation

  def init_etrace_state(self, *args, **kwargs):
    """
    Initialize the eligibility trace states of the etrace algorithm.

    This method is needed after compiling the etrace graph. See
    `.compile_graph()` for the details.
    """
    # The states of weight spatial gradients:
    #   1. x
    #   2. df
    self.etrace_xs = bst.visible_state_dict()
    self.etrace_dfs = bst.visible_state_dict()
    for relation in self.graph.hidden_param_op_relations:
      # For the relation
      #
      #   h1, h2, ... = f(x, w)
      #
      # we need to initialize the eligibility trace states for the weight x and the df.

      if relation.x not in self.etrace_xs:
        shape = (self.n_truncation,) + relation.x.aval.shape
        dtype = relation.x.aval.dtype
        # wx may be repeatedly used in the graph
        self.etrace_xs[relation.x] = bst.ShortTermState(u.math.zeros(shape, dtype))

      for group in relation.hidden_groups:
        group: HiddenGroup
        #
        # Group 1:
        #
        #   [∂a^t-1/∂θ1, ∂b^t-1/∂θ1, ...]
        #
        # Group 2:
        #
        #   [∂A^t-1/∂θ1, ∂B^t-1/∂θ1, ...]
        #
        for hidden_outvar in group.hidden_outvars:
          key = (relation.y, hidden_outvar)
          if key in self.etrace_dfs:  # relation.y is an unique output of the weight operation
            raise ValueError(f'The relation {key} has been added. ')
          shape = (self.n_truncation,) + relation.y.aval.shape
          dtype = relation.y.aval.dtype
          self.etrace_dfs[key] = bst.ShortTermState(u.math.zeros(shape, dtype))

  def reset_state(self, batch_size: int = None, **kwargs):
    """
    Reset the eligibility trace states.
    """

    def _zeros_like_batch_(x: jax.Array):
      if self.mode.has(bst.mixin.Batching):
        assert batch_size is not None, 'The batch size should be provided. '
        shape = (self.n_truncation, batch_size) + x.shape[1:]
      else:
        shape = (self.n_truncation,) + x.shape
      return u.math.zeros(shape, x.dtype)

    for k, v in self.etrace_xs.items():
      self.etrace_xs[k].value = jax.tree.map(_zeros_like_batch_, v)
    for k, v in self.etrace_dfs.items():
      self.etrace_dfs[k].value = jax.tree.map(_zeros_like_batch_, v)

  def _get_etrace_data(self) -> Tuple:
    etrace_xs = {k: v.value for k, v in self.etrace_xs.items()}
    etrace_dfs = {k: v.value for k, v in self.etrace_dfs.items()}
    return etrace_xs, etrace_dfs

  def _assign_etrace_data(self, hist_etrace_vals):
    #
    # For any operation:
    #
    #           h^t = f(x^t \theta)
    #
    # etrace_xs:
    #           x^t
    #
    # etrace_dfs:
    #           df^t = ∂h^t / ∂y^t, where y^t = x^t \theta
    #
    (etrace_xs, etrace_dfs) = hist_etrace_vals

    # the weight x and df
    for x, val in etrace_xs.items():
      self.etrace_xs[x].value = val
    for dfkey, val in etrace_dfs.items():
      self.etrace_dfs[dfkey].value = val

  def _update_etrace_data(
      self,
      running_index: Optional[int],
      hist_etrace_vals: PyTree,
      hid2weight_jac_at_t: Tuple[Dict[WeightXVar, jax.Array], Dict[Tuple[WeightYVar, HiddenOutVar], jax.Array]],
      hid2hid_jac_at_t: Dict[(HiddenOutVar, HiddenOutVar), jax.Array],
      weight_id_to_its_val: Dict[WeightID, PyTree],
  ) -> ETraceVals:
    #
    # "running_index":
    #            the running index
    #
    # "hist_etrace_vals":
    #            the history etrace values,
    #            including the x and df values, see "etrace_xs" and "etrace_dfs".
    #
    # "hid2weight_jac_at_t":
    #           the current etrace values at the time "t", \epsilon^t, if vjp_time == "t".
    #           Otherwise, the etrace values at the time "t-1", \epsilon^{t-1}.
    #
    # "hid2hid_jac_at_t":
    #           the data for computing the hidden-to-hidden Jacobian at the time "t".
    #
    # "weight_id_to_its_val":
    #           the weight values.
    #

    # --- the data --- #
    #
    # the etrace data at the current time step (t) of the O(n) algorithm
    # is a tuple, including the weight x and df values.
    #
    # For the weight x, it is a dictionary,
    #    {WeightXVar: jax.Array}
    #
    # For the weight df, it is a dictionary,
    #    {(WeightYVar, HiddenOutVar): jax.Array}
    #
    xs, dfs = hid2weight_jac_at_t

    #
    # the history etrace values
    #
    # - hist_xs: {WeightXVar: brainstate.State}
    # - hist_dfs: {(WeightYVar, HiddenOutVar): brainstate.State}
    #
    hist_xs, hist_dfs = hist_etrace_vals

    #
    # the new etrace values
    new_etrace_xs, new_etrace_dfs = dict(), dict()

    # --- the update --- #

    #
    # Step 1:
    #
    #   update the weight x using the equation:
    #           x^t = α * x^t-1 + x^t, where α is the decay factor.
    #
    for xkey in hist_xs.keys():
      new_etrace_xs[xkey] = u.math.concatenate((hist_xs[xkey][1:], u.math.expand_dims(xs[xkey], 0)), axis=0)

    for hwo_relation in self.graph.hidden_param_op_relations:
      hwo_relation: HiddenWeightOpRelation

      for group in hwo_relation.hidden_groups:
        group: HiddenGroup
        #
        # Step 2:
        #
        # update the eligibility trace * hidden diagonal Jacobian
        #         dϵ^t = D_h ⊙ dϵ^t-1, where D_h is the hidden-to-hidden Jacobian diagonal matrix.
        #
        #
        # JVP equation for the following Jacobian computation:
        #
        # [∂V^t/∂V^t-1, ∂V^t/∂a^t-1,  [∂V^t-1/∂θ1,
        #  ∂a^t/∂V^t-1, ∂a^t/∂a^t-1]   ∂a^t-1/∂θ1,]
        #
        # [∂V^t/∂V^t-1, ∂V^t/∂a^t-1,  [∂V^t-1/∂θ2,
        #  ∂a^t/∂V^t-1, ∂a^t/∂a^t-1]   ∂a^t-1/∂θ2]
        #

        for hidden_outvar_at_t in group.hidden_outvars:
          #
          # computing the following vector-Jacobian product:
          #  df^t = D_h ⊙ df^t-1
          #
          # i.e., the hidden-to-hidden Jacobian diagonal matrix * the hidden df at the previous time step
          #
          #  ∂V^t/∂θ1 = ∂V^t/∂V^t-1 * ∂V^t-1/∂θ1 + ∂V^t/∂a^t-1 * ∂a^t-1/∂θ1 + ...
          #
          new_etrace = None
          for hidden_outvar_at_t_minus_1 in group.hidden_outvars:
            diag_jac_key = (hidden_outvar_at_t_minus_1, hidden_outvar_at_t)
            if diag_jac_key in hid2hid_jac_at_t:
              # diagonal Jacobian * hidden df
              data = (hist_dfs[(hwo_relation.y, hidden_outvar_at_t_minus_1)] *
                      u.math.expand_dims(hid2hid_jac_at_t[diag_jac_key], axis=0))
              new_etrace = data if new_etrace is None else new_etrace + data
          assert new_etrace is not None, f'The new etrace should not be None. '

          #
          # Step 3:
          #
          # update: eligibility trace * hidden diagonal Jacobian + new hidden df
          #        dϵ^t = D_h ⊙ dϵ^t-1 + df^t, where D_h is the hidden-to-hidden Jacobian diagonal matrix.
          #
          new_df_key = (hwo_relation.y, hidden_outvar_at_t)
          new_etrace_dfs[new_df_key] = u.math.concatenate(
            [new_etrace[1:], u.math.expand_dims(dfs.get(new_df_key, u.math.zeros_like(new_etrace[0])), axis=0)],
            axis=0
          )
    return new_etrace_xs, new_etrace_dfs

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
    xs, dfs = etrace_h2w_at_t
    for relation in self.graph.hidden_param_op_relations:
      x = xs[relation.x]
      weight_id = id(relation.weight)

      #
      # Function to compute the weight gradients
      # according to the inputs and df gradients
      #
      fun_dxy2dw = lambda dx, dy: _weight_op_gradient(relation.op_jaxpr, dx, weight_id_to_its_val[weight_id], dy)

      #
      # Solve the weight gradients by using the etrace data
      #
      #   dw = (dL/dH \circ df) \otimes x
      #
      for i, hid_var in enumerate(relation.hidden_vars):
        df = dfs[(relation.y, hid_var)]  # the hidden gradients
        df_hid = df * u.math.expand_dims(dl_to_dh_at_t[hid_var], axis=0)  # the hidden gradients
        #
        # Compute the weight gradients according to the x and y
        #
        #    dw = df(dx, dy)
        #
        dg_weight = jax.tree.map(lambda a: a.sum(0), jax.vmap(fun_dxy2dw)(x, df_hid))
        _update_dict(dG_weights, weight_id, dg_weight)  # update the weight gradients

    # update the non-etrace parameters
    etrace_params, _, non_etrace_params, _ = split_states_v2(self.graph.states)
    for st, dg in zip(non_etrace_params, dl_to_nonetws_at_t):
      _update_dict(dG_weights, id(st), dg)

    # update the etrace parameters when "dl_to_etws_at_t" is not None
    if dl_to_etws_at_t is not None:
      for st, dg in zip(etrace_params, dl_to_etws_at_t):
        _update_dict(dG_weights, id(st), dg, error_when_no_key=True)
    return list(dG_weights.values())


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
  decay: float  # the exponential smoothing decay factor

  def __init__(
      self,
      model: Callable,
      decay_or_rank: float | int,
      diag_normalize: Optional[bool] = None,
      vjp_time: str = 't',
      name: Optional[str] = None,
      mode: Optional[bst.mixin.Mode] = None
  ):
    super().__init__(
      model,
      diag_normalize=diag_normalize,
      vjp_time=vjp_time,
      name=name,
      mode=mode
    )

    # the learning parameters
    self.decay, num_rank = _format_decay_and_rank(decay_or_rank)

  def init_etrace_state(self, *args, **kwargs):
    """
    Initialize the eligibility trace states of the etrace algorithm.

    This method is needed after compiling the etrace graph. See
    `.compile_graph()` for the details.
    """
    # The states of weight spatial gradients:
    #   1. x
    #   2. df
    self.etrace_xs = bst.visible_state_dict()
    self.etrace_dfs = bst.visible_state_dict()
    for relation in self.graph.hidden_param_op_relations:
      _init_IO_dim_state(self, relation)

  def reset_state(self, batch_size: int = None, **kwargs):
    """
    Reset the eligibility trace states.
    """
    _reset_state_in_a_dict(self.etrace_xs, batch_size, self.mode)
    _reset_state_in_a_dict(self.etrace_dfs, batch_size, self.mode)

  def get_etrace_of(self, weight: bst.ParamState | WeightID) -> Tuple[Dict, Dict]:
    """
    Get the eligibility trace of the given weight.

    The eligibility trace contains the following structures:

    """

    self._assert_compiled()

    # the weight ID
    weight_id = id(weight) if isinstance(weight, bst.ParamState) else weight
    if not isinstance(weight_id, int):
      raise TypeError

    etrace_xs = dict()
    etrace_dfs = dict()
    find_this_weight = False
    for relation in self.graph.hidden_param_op_relations:
      relation: HiddenWeightOpRelation
      if id(relation.weight) != weight_id:
        continue
      find_this_weight = True

      # get the weight_op input
      wx_var = relation.x
      etrace_xs[wx_var] = self.etrace_xs[wx_var].value

      # get the weight_op df
      wy_var = relation.y
      for group in relation.hidden_groups:
        group: HiddenGroup
        for hidden_var in group.hidden_outvars:
          etrace_dfs[(wy_var, hidden_var)] = self.etrace_dfs[(wy_var, hidden_var)].value
    if not find_this_weight:
      raise ValueError(f'Do not the etrace of the given weight: {weight}.')
    return etrace_xs, etrace_dfs

  def _get_etrace_data(self) -> Tuple:
    etrace_xs = {k: v.value for k, v in self.etrace_xs.items()}
    etrace_dfs = {k: v.value for k, v in self.etrace_dfs.items()}
    return etrace_xs, etrace_dfs

  def _assign_etrace_data(self, hist_etrace_vals):
    #
    # For any operation:
    #
    #           h^t = f(x^t \theta)
    #
    # etrace_xs:
    #           x^t
    #
    # etrace_dfs:
    #           df^t = ∂h^t / ∂y^t, where y^t = x^t \theta
    #
    (etrace_xs, etrace_dfs) = hist_etrace_vals

    # the weight x and df
    for x, val in etrace_xs.items():
      self.etrace_xs[x].value = val
    for dfkey, val in etrace_dfs.items():
      self.etrace_dfs[dfkey].value = val

  def _update_etrace_data(
      self,
      running_index: Optional[int],
      hist_etrace_vals: PyTree,
      hid2weight_jac_at_t: Tuple[Dict[WeightXVar, jax.Array], Dict[Tuple[WeightYVar, HiddenOutVar], jax.Array]],
      hid2hid_jac_at_t: Dict[(HiddenOutVar, HiddenOutVar), jax.Array],
      weight_id_to_its_val: Dict[WeightID, PyTree],
  ) -> ETraceVals:
    #
    # "running_index":
    #            the running index
    #
    # "hist_etrace_vals":
    #            the history etrace values,
    #            including the x and df values, see "etrace_xs" and "etrace_dfs".
    #
    # "hid2weight_jac_at_t":
    #           the current etrace values at the time "t", \epsilon^t, if vjp_time == "t".
    #           Otherwise, the etrace values at the time "t-1", \epsilon^{t-1}.
    #
    # "hid2hid_jac_at_t":
    #           the data for computing the hidden-to-hidden Jacobian at the time "t".
    #
    # "weight_id_to_its_val":
    #           the weight values.
    #
    return _update_IO_dim_etrace_with_exact_jac(
      hist_etrace_vals,
      hid2weight_jac_at_t,
      hid2hid_jac_at_t,
      self.graph.hidden_param_op_relations,
      running_index,
      decay=self.decay,
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
    _solve_IO_dim_weight_gradients(
      etrace_h2w_at_t,
      dG_weights,
      dl_to_dh_at_t,
      self.graph.hidden_param_op_relations,
      weight_id_to_its_val,
      running_index,
      self.decay,
    )

    # update the non-etrace parameters
    etrace_params, _, non_etrace_params, _ = split_states_v2(self.graph.states)
    for st, dg in zip(non_etrace_params, dl_to_nonetws_at_t):
      _update_dict(dG_weights, id(st), dg)

    # update the etrace parameters when "dl_to_etws_at_t" is not None
    if dl_to_etws_at_t is not None:
      for st, dg in zip(etrace_params, dl_to_etws_at_t):
        _update_dict(dG_weights, id(st), dg, error_when_no_key=True)
    return list(dG_weights.values())


class _ParamDimAlgorithm(Protocol):
  etrace_bwg: Dict[Tuple[WeightID, WeightXVar, HiddenOutVar], bst.State]
  mode: bst.mixin.Mode


def _init_param_dim_state(
    self: _ParamDimAlgorithm,
    relation: HiddenWeightOpRelation
):
  # For the relation
  #
  #   h1, h2, ... = f(x, w)
  #
  # we need to initialize the eligibility trace states for the weight x and the df.

  # TODO: assume the batch size is the first dimension
  batch_size = relation.x.aval.shape[0] if self.mode.has(bst.mixin.Batching) else None
  for group in relation.hidden_groups:
    group: HiddenGroup

    for hidden_outvar in group.hidden_outvars:
      key = (id(relation.weight), relation.x, hidden_outvar)
      if key in self.etrace_bwg:  # The key should be unique
        raise ValueError(f'The relation {key} has been added. ')
      self.etrace_bwg[key] = bst.ShortTermState(
        jax.tree.map(partial(_batched_zeros_like, batch_size), relation.weight.value)
      )


def _update_param_dim_etrace_with_exact_jac(
    hist_etrace_vals: Dict,
    hid2weight_jac: Tuple,
    hid2hid_jac_at_t: Dict[(HiddenOutVar, HiddenOutVar), jax.Array],
    weight_id_to_its_val: Dict,
    hidden_param_op_relations,
    diag_normalize: bool,
    mode: bst.mixin.Mode
):
  #
  # + "hist_etrace_vals" has the following structure:
  #    - key: the weight id, the weight-x jax var, the hidden state var
  #    - value: the batched weight gradients
  #

  # + "hid2weight_jac" has the following structure:
  #    - a dict of weight x gradients
  #       * key: the weight x jax var
  #       * value: the weight x gradients
  #    - a dict of weight y gradients
  #       * key: the tuple of the weight y jax var and the hidden state jax var
  #       * value: the weight y gradients
  #
  etrace_xs_at_t, etrace_ys_at_t = hid2weight_jac

  #
  # The etrace weight gradients at the current time step.
  # i.e., The "hist_etrace_vals" at the next time step
  #
  new_etrace_bwg = dict()

  for relation in hidden_param_op_relations:
    relation: HiddenWeightOpRelation

    #
    # ParamDim algorithm relies on the "ETraceOp" to compute the etrace updates
    # Therefore, the weight should be defined as an "ETraceParamOp", so that
    # the "ETraceOp" can be obtained directly through "ETraceParamOp.op".
    #
    if not isinstance(relation.weight, ETraceParamOp):
      raise NotImplementedError(
        f'When using {DiagParamDimAlgorithm.__name__} or '
        f'{DiagHybridDimAlgorithm.__name__} algorithms, '
        f'the weight should be an {ETraceParamOp.__name__}. '
      )

    #
    # Step 1:
    #
    # Define the etrace operation for computing etrace updates
    #

    # the etrace operation for computing etrace updates
    if isinstance(relation.weight.op, StandardETraceOp):
      etrace_op = relation.weight.op
    else:
      etrace_op = GeneralETraceOp(
        relation.weight.op,
        xinfo=jax.ShapeDtypeStruct(relation.x.aval.shape, relation.x.aval.dtype),
        is_diagonal=relation.weight.op.is_diagonal
      )

    # weight information
    weight_id = id(relation.weight)
    weight_val = weight_id_to_its_val[weight_id]

    for group in relation.hidden_groups:
      for hidden_outvar_at_t in group.hidden_outvars:
        group: HiddenGroup
        #
        # Step 2:
        #
        # update: eligibility trace * hidden diagonal Jacobian + new hidden df
        #        dϵ^t = D_h ⊙ dϵ^t-1 + df^t, where D_h is the hidden-to-hidden Jacobian diagonal matrix.
        #
        #
        # computing the following vector-Jacobian product:
        #  df^t = D_h ⊙ df^t-1
        #
        # i.e., the hidden-to-hidden Jacobian diagonal matrix * the hidden df at the previous time step
        #
        #  ∂V^t/∂θ1 = ∂V^t/∂V^t-1 * ∂V^t-1/∂θ1 + ∂V^t/∂a^t-1 * ∂a^t-1/∂θ1 + ...
        #
        diag_jac, dh_to_dw = [], []
        for hidden_outvar_at_t_minus_1 in group.hidden_outvars:
          diag_jac_key = (hidden_outvar_at_t_minus_1, hidden_outvar_at_t)
          if diag_jac_key in hid2hid_jac_at_t:
            diag_jac.append(hid2hid_jac_at_t[diag_jac_key])
            dh_to_dw.append(hist_etrace_vals[(weight_id, relation.x, hidden_outvar_at_t_minus_1)])

        assert len(diag_jac) > 0, f'The diagonal Jacobian should not be empty. '
        new_bwg = etrace_op.etrace_update(
          mode,
          weight_val,  # pytree
          dh_to_dw,  # list of pytree as weight_val, with batch size
          diag_jac,  # List of jax.Array
          etrace_xs_at_t[relation.x],  # jax.Array
          etrace_ys_at_t.get((relation.y, hidden_outvar_at_t), None)  # jax.Array | None
        )

        # assignment
        w_key = (weight_id, relation.x, hidden_outvar_at_t)
        new_etrace_bwg[w_key] = new_bwg
  return new_etrace_bwg


def _solve_param_dim_weight_gradients(
    hist_etrace_data: Dict,
    dG_weights: Dict[WeightID, dG_Weight],
    dG_hiddens: Dict[HiddenOutVar, jax.Array],
    weight_hidden_relations: Sequence[HiddenWeightOpRelation],
    weight_id_to_its_val: Dict[WeightID, PyTree],
    mode: bst.mixin.Mode
):
  # update the etrace weight gradients
  temp_data = dict()
  for relation in weight_hidden_relations:
    if not isinstance(relation.weight, ETraceParamOp):
      raise NotImplementedError(
        f'When using {DiagParamDimAlgorithm.__name__} '
        f'or {DiagHybridDimAlgorithm.__name__} algorithms, '
        f'the weight should be an {ETraceParamOp.__name__}. '
      )

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
        is_diagonal=relation.weight.op.is_diagonal,
      )

    for i, hid_var in enumerate(relation.hidden_vars):
      key = (weight_id, relation.x, hid_var)
      #
      # dE/dH, computing the hidden to weight gradients
      #
      dg_hidden = dG_hiddens[hid_var]
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
      _update_dict(temp_data, weight_id, dg_weight)

  # sum up the batched weight gradients
  if mode.has(bst.mixin.Batching):
    for key, val in temp_data.items():
      temp_data[key] = jax.tree_map(lambda x: u.math.sum(x, axis=0), val)

  # update the weight gradients
  for key, val in temp_data.items():
    _update_dict(dG_weights, key, val)


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
      diag_normalize: Optional[bool] = None,
      vjp_time: str = 't',
      name: Optional[str] = None,
      mode: Optional[bst.mixin.Mode] = None
  ):
    super().__init__(
      model,
      diag_normalize=diag_normalize,
      vjp_time=vjp_time,
      name=name,
      mode=mode
    )

  def init_etrace_state(self, *args, **kwargs):
    """
    Initialize the eligibility trace states of the etrace algorithm.

    This method is needed after compiling the etrace graph. See
    `.compile_graph()` for the details.
    """
    # The states of batched weight gradients
    self.etrace_bwg = bst.visible_state_dict()
    for relation in self.graph.hidden_param_op_relations:
      _init_param_dim_state(self, relation)

  def reset_state(self, batch_size: int = None, **kwargs):
    """
    Reset the eligibility trace states.
    """
    _reset_state_in_a_dict(self.etrace_bwg, batch_size, self.mode)

  def get_etrace_of(self, weight: bst.ParamState | WeightID) -> Dict:
    """
    Get the eligibility trace of the given weight.

    The eligibility trace contains the following structures:

    """

    self._assert_compiled()

    # get the wight id
    weight_id = id(weight) if isinstance(weight, bst.ParamState) else weight
    if not isinstance(weight_id, int):
      raise TypeError

    find_this_weight = False
    etraces = dict()
    for relation in self.graph.hidden_param_op_relations:
      relation: HiddenWeightOpRelation
      if id(relation.weight) != weight_id:
        continue
      find_this_weight = True

      # retrieve the etrace data
      wx_var = relation.x
      for group in relation.hidden_groups:
        group: HiddenGroup
        for hidden_outvar in group.hidden_outvars:
          key = (weight_id, wx_var, hidden_outvar)
          etraces[key] = self.etrace_bwg[key].value

    if not find_this_weight:
      raise ValueError(f'Do not the etrace of the given weight: {weight}.')
    return etraces

  def _get_etrace_data(self) -> Dict:
    return {k: v.value for k, v in self.etrace_bwg.items()}

  def _assign_etrace_data(self, etrace_vals: Dict) -> None:
    for x, val in etrace_vals.items():
      self.etrace_bwg[x].value = val

  def _update_etrace_data(
      self,
      running_index: Optional[int],
      hist_etrace_vals: Dict[Tuple[WeightID, WeightXVar, HiddenOutVar], PyTree],
      hid2weight_jac_at_t: Tuple[Dict[WeightXVar, jax.Array], Dict[Tuple[WeightYVar, HiddenOutVar], jax.Array]],
      hid2hid_jac_at_t,
      weight_id_to_its_val: Dict[WeightID, PyTree],
  ) -> Dict[Tuple[WeightID, WeightXVar, HiddenOutVar], PyTree]:
    return _update_param_dim_etrace_with_exact_jac(
      hist_etrace_vals,
      hid2weight_jac_at_t,
      hid2hid_jac_at_t,
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
    _solve_param_dim_weight_gradients(
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
      _update_dict(dG_weights, id(st), dg)

    # update the etrace parameters when "dl_to_etws_at_t" is not None
    if dl_to_etws_at_t is not None:
      for st, dg in zip(etrace_params, dl_to_etws_at_t):
        _update_dict(dG_weights, id(st), dg, error_when_no_key=True)
    return list(dG_weights.values())


def _numel(pytree: PyTree):
  return sum(u.math.size(x) for x in jax.tree_leaves(pytree))


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
  # batch of weight gradients
  etrace_bwg: Dict[Tuple[WeightID, WeightXVar, HiddenOutVar], bst.State]
  decay: float  # the exponential smoothing decay factor

  def __init__(
      self,
      model: Callable,
      decay_or_rank: float | int,
      vjp_time: str = 't',
      diag_normalize: Optional[bool] = None,
      name: Optional[str] = None,
      mode: Optional[bst.mixin.Mode] = None
  ):
    super().__init__(
      model,
      diag_normalize=diag_normalize,
      vjp_time=vjp_time,
      name=name,
      mode=mode
    )

    # the learning parameters
    self.decay, num_rank = _format_decay_and_rank(decay_or_rank)

  def init_etrace_state(self, *args, **kwargs):
    """
    Initialize the eligibility trace states of the etrace algorithm.

    This method is needed after compiling the etrace graph. See
    `.compile_graph()` for the details.
    """
    #
    # The states of weight spatial gradients:
    #   1. x
    #   2. df
    #   3. batched weight gradients
    #
    self.etrace_xs = bst.visible_state_dict()
    self.etrace_dfs = bst.visible_state_dict()
    self.etrace_bwg = bst.visible_state_dict()

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
          _init_param_dim_state(self, relation)
          continue
        elif relation.weight.gradient == _ETraceGrad.approx:
          #
          # When
          #     weight.gradient == _ETraceGrad.approx
          #
          # the weights will be forced to use O(n) algorithm
          # to compute the eligibility trace.
          #
          _init_IO_dim_state(self, relation)
          continue

      batch_size = relation.x.shape[0] if self.mode.has(bst.mixin.Batching) else 1
      if _numel(relation.x) + _numel(relation.y) > batch_size * _numel(relation.weight.value):
        #
        # When the number of elements in the inputs and outputs are bigger than the weight number,
        # we will use the O(n^2) algorithm to compute the eligibility trace, since
        # storing the batched weight gradients will be less expensive.
        #
        _init_param_dim_state(self, relation)
      else:
        #
        # For most cases, we will use the O(n) algorithm to compute the eligibility trace.
        # Since the number of elements in input and output (I + O) is greatly less than the number
        # of elements in the weight (W = I * O).
        #
        _init_IO_dim_state(self, relation)

  def reset_state(self, batch_size: int = None, **kwargs):
    """
    Reset the eligibility trace states.
    """
    _reset_state_in_a_dict(self.etrace_xs, batch_size, self.mode)
    _reset_state_in_a_dict(self.etrace_dfs, batch_size, self.mode)
    _reset_state_in_a_dict(self.etrace_bwg, batch_size, self.mode)

  def get_etrace_of(self, weight: bst.ParamState | WeightID) -> Tuple[Dict, Dict, Dict]:
    """
    Get the eligibility trace of the given weight.

    The eligibility trace contains the following structures:

    """

    self._assert_compiled()

    # the weight ID
    weight_id = id(weight) if isinstance(weight, bst.ParamState) else weight
    if not isinstance(weight_id, int):
      raise TypeError

    etrace_xs = dict()
    etrace_dfs = dict()
    etrace_bws = dict()
    find_this_weight = False
    for relation in self.graph.hidden_param_op_relations:
      relation: HiddenWeightOpRelation
      if id(relation.weight) != weight_id:
        continue
      find_this_weight = True

      wx_var = relation.x
      if wx_var in self.etrace_xs:
        # get the weight_op input
        etrace_xs[wx_var] = self.etrace_xs[wx_var].value

        # get the weight_op df
        wy_var = relation.y
        for group in relation.hidden_groups:
          group: HiddenGroup
          for hidden_var in group.hidden_outvars:
            etrace_dfs[(wy_var, hidden_var)] = self.etrace_dfs[(wy_var, hidden_var)].value

      # get the batched weight gradients
      for group in relation.hidden_groups:
        group: HiddenGroup
        for hidden_outvar in group.hidden_outvars:
          key = (weight_id, wx_var, hidden_outvar)
          etrace_bws[key] = self.etrace_bwg[key].value

    if not find_this_weight:
      raise ValueError(f'Do not the etrace of the given weight: {weight}.')

    return etrace_xs, etrace_dfs, etrace_bws

  def _get_etrace_data(self) -> Tuple[Dict, ...]:
    etrace_xs = {x: val.value for x, val in self.etrace_xs.items()}
    etrace_dfs = {x: val.value for x, val in self.etrace_dfs.items()}
    etrace_wgrads = {x: val.value for x, val in self.etrace_bwg.items()}
    return etrace_xs, etrace_dfs, etrace_wgrads

  def _assign_etrace_data(self, etrace_vals: Tuple[Dict, ...]) -> None:
    etrace_xs, etrace_dfs, etrace_wgrads = etrace_vals
    for x, val in etrace_xs.items():
      self.etrace_xs[x].value = val
    for x, val in etrace_dfs.items():
      self.etrace_dfs[x].value = val
    for x, val in etrace_wgrads.items():
      self.etrace_bwg[x].value = val

  def _update_etrace_data(
      self,
      running_index: Optional[int],
      hist_etrace_vals: Tuple[Dict, ...],
      hid2weight_jac_at_t: Tuple[Dict[WeightXVar, jax.Array], Dict[Tuple[WeightYVar, HiddenOutVar], jax.Array]],
      hid2hid_jac_at_t,
      weight_id_to_its_val: Dict[WeightID, PyTree],
  ) -> Tuple[Dict, ...]:

    # the history etrace values
    hist_xs, hist_dfs, hist_bwg = hist_etrace_vals

    # ---- O(n^2) etrace gradients update ---- #
    on_weight_hidden_relations = []
    on2_weight_hidden_relations = []
    for relation in self.graph.hidden_param_op_relations:
      if _is_weight_need_full_grad(relation, self.mode):
        on2_weight_hidden_relations.append(relation)
      else:
        on_weight_hidden_relations.append(relation)

    new_bwg = _update_param_dim_etrace_with_exact_jac(
      hist_bwg,
      hid2weight_jac_at_t,
      hid2hid_jac_at_t,
      weight_id_to_its_val,
      on2_weight_hidden_relations,
      self.diag_normalize,
      self.mode
    )

    # ---- O(n) etrace gradients update ---- #
    new_xs, new_dfs = (
      _update_IO_dim_etrace_with_exact_jac(
        hist_etrace_vals,
        hid2weight_jac_at_t,
        hid2hid_jac_at_t,
        on_weight_hidden_relations,
        running_index,
        decay=self.decay,
      )
    )

    return new_xs, new_dfs, new_bwg

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

    xs, dfs, wgrads = etrace_h2w_at_t
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
    _solve_IO_dim_weight_gradients(
      [xs, dfs],
      dG_weights,
      dl_to_dh_at_t,
      on_weight_hidden_relations,
      weight_id_to_its_val,
      running_index,
      self.decay,
    )

    # update the etrace weight gradients by the O(n^2) algorithm
    _solve_param_dim_weight_gradients(
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
      _update_dict(dG_weights, id(st), dg)

    # update the etrace parameters when "dl_to_etws_at_t" is not None
    if dl_to_etws_at_t is not None:
      for st, dg in zip(etrace_params, dl_to_etws_at_t):
        _update_dict(dG_weights, id(st), dg, error_when_no_key=True)
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
