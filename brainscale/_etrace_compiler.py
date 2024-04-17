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
from typing import Callable, NamedTuple, List, Dict, Sequence, Tuple, Set, Any

import braincore as bc
import jax.core
import jax.numpy as jnp
from jax.extend import linear_util as lu
from jax.extend import source_info_util
from jax.interpreters import partial_eval as pe

from ._errors import NotSupportedError, CompilationError
from ._etrace_concepts import _etrace_op_name, ETraceParam, ETraceVar
from ._etrace_concepts import assign_state_values, split_states_v2
from ._misc import git_issue_addr, state_traceback, set_module_as
from .typing import (PyTree, StateID, WeightID, WeightXVar, WeightYVar, HiddenVar,
                     ETraceVals, TempData, Outputs, HiddenVals, StateVals, WeightVals)

# TODO
# - [ ] visualization of the etrace graph


__all__ = [
  'ETraceGraph', 'build_etrace_graph',
]

shape_inverse_rule = {
  jax.lax.concatenate_p: jax.lax.slice_p,
  jax.lax.slice_p: jax.lax.dynamic_slice_p,
  jax.lax.reshape_p: jax.lax.reshape_p,
  jax.lax.dynamic_slice_p: jax.lax.slice_p,
  jax.lax.dynamic_update_slice_p: jax.lax.slice_p,
  jax.lax.transpose_p: jax.lax.transpose_p,
  jax.lax.broadcast_in_dim_p: jax.lax.squeeze_p,
  jax.lax.squeeze_p: jax.lax.broadcast_in_dim_p,
  jax.lax.rev_p: jax.lax.rev_p,
  jax.lax.scatter_p: jax.lax.gather_p,
  jax.lax.scatter_add_p: jax.lax.gather_p,
  jax.lax.scatter_max_p: jax.lax.gather_p,
  jax.lax.scatter_min_p: jax.lax.gather_p,
  jax.lax.scatter_mul_p: jax.lax.gather_p,
}

shape_changing_rule = {
  jax.lax.slice_p: jax.lax.slice,
  jax.lax.dynamic_slice_p: jax.lax.dynamic_slice,
}


def identity(x):
  """
  Identity function. Return x
  """
  return x


def fun_compose(*funs: Callable) -> Callable:
  """
  Composing the functions.

  Args:
    *funs: The functions to be composed.

  Returns:
    The composed function.
  """

  def composed(x):
    for fun in funs:
      x = fun(x)
    return x

  return composed


def split_state_values(
    states: Sequence[bc.State],
    state_values: List[PyTree],
    include_weight: bool = True
) -> (Tuple[WeightVals, HiddenVals, StateVals] | Tuple[HiddenVals, StateVals]):
  """
  Split the state values into the weight values, the hidden values, and the other state values.

  The weight values are the values of the ``braincore.ParamState`` states (including ``ETraceParam``).
  The hidden values are the values of the ``ETraceVar`` states.
  The other state values are the values of the other states.

  Parameters:
  -----------
  states: Sequence[bc.State]
    The states of the model.
  state_values: List[PyTree]
    The values of the states.
  include_weight: bool
    Whether to include the weight values.

  Returns:
  --------
  The weight values, the hidden values, and the other state values.

  Examples:
  ---------
  >>> split_state_values(states, state_values)
  (weight_vals, hidden_vals, other_vals)

  >>> split_state_values(states, state_values, include_weight=False)
  (hidden_vals, other_vals)
  """
  if include_weight:
    weight_vals, hidden_vals, other_vals = [], [], []
    for st, val in zip(states, state_values):
      if isinstance(st, bc.ParamState):
        weight_vals.append(val)
      elif isinstance(st, ETraceVar):
        hidden_vals.append(val)
      else:
        other_vals.append(val)
    return weight_vals, hidden_vals, other_vals
  else:
    hidden_vals, other_vals = [], []
    for st, val in zip(states, state_values):
      if isinstance(st, bc.ParamState):
        pass
      elif isinstance(st, ETraceVar):
        hidden_vals.append(val)
      else:
        other_vals.append(val)
    return hidden_vals, other_vals


class WeightOpTracer(NamedTuple):
  """
  The data structure for the tracing of the ETraceParam operation.
  """
  weight: ETraceParam
  # op: jax.core.ClosedJaxpr
  op: jax.core.JaxprEqn
  x: jax.core.Var
  y: jax.core.Var
  trace: List[jax.core.JaxprEqn]
  hidden_vars: set[jax.core.Var]
  invar_needed_in_oth_eqns: set[jax.core.Var]


class TracedWeightOp(NamedTuple):
  """
  The data structure for the traced weight operation.

  The following fields are included:

  - weight: the instance of ``ETraceParam``
  - op_jaxpr: the jaxpr for the weight operation, instance of ``jax.core.Jaxpr``
  - x: the jax Var for the weight input
  - y: the jax Var for the wight output
  - jaxpr_y2hid: the jaxpr to evaluate y -->  eligibility trace variables
  - hidden_vars: the hidden states connected to the weight
  - hidden2df: the function to convert each hidden stat to the size of weight output
  """
  weight: ETraceParam
  op_jaxpr: jax.core.Jaxpr
  x: jax.core.Var
  y: jax.core.Var
  jaxpr_y2hid: jax.core.Jaxpr
  hidden_vars: List[jax.core.Var]
  hidden2df: List[Callable]


def _check_some_element_exist_in_the_set(
    elements: Sequence[jax.core.Var],
    the_set: Set[jax.core.Var]
) -> bool:
  """
  Checking whether the jaxpr vars contain the weight variables.

  Args:
    elements: The input variables of the equation.
    the_set: The set of the weight variables.
  """
  for invar in elements:
    if isinstance(invar, jax.core.Var) and invar in the_set:
      return True
  return False


def _check_matched(
    invars: Sequence[jax.core.Var],
    invar_needed_in_oth_eqns: Set[jax.core.Var]
) -> List[jax.core.Var]:
  """
  Checking whether the invars are matched with the invar_needed_in_oth_eqns.

  Args:
    invars: The input variables of the equation.
    invar_needed_in_oth_eqns: The variables needed in the other equations.
  """
  matched = []
  for invar in invars:
    if isinstance(invar, jax.core.Var) and invar in invar_needed_in_oth_eqns:
      matched.append(invar)
  return matched


def _trace_simplify(trace: WeightOpTracer) -> TracedWeightOp:
  """
  Simplifying the trace from the weight output to the hidden state.

  Args:
    trace: The traced weight operation.

  Returns:
    The simplified traced weight operation.
  """
  # [first step]
  # Check the hidden states of the given weight. If the hidden states are not
  # used in the model, we raise an error. This is to avoid the situation that
  # the weight is defined but not used in the model.
  if len(trace.hidden_vars) == 0:
    source_info = trace.weight.source_info
    name_stack = source_info_util.current_name_stack() + source_info.name_stack
    with source_info_util.user_context(source_info.traceback, name_stack=name_stack):
      raise CompilationError(
        f'Error: The ETraceParam {trace.weight} does not found the associated hidden states: \n'
        f'There are maybe two kinds of reasons: '
        f'1. The weight is not associated with any hidden states. Therefore it should not be defined \n'
        f'   as a {ETraceParam.__name__}. You can turon of ETrace learning for this weight by setting \n'
        f'   "as_etrace_weight=False". For example, \n'
        f'       \n'
        f'        lin = brainscale.Linear(..., as_etrace_weight=False)". \n'
        f'2. This may be a compilation error. Please report an issue to the developers at {git_issue_addr}. \n'
        f'\n'
        f'Moreover, see the above traceback information for where the weight is defined in your code.'
      )

  # [second step]
  # Remove the unnecessary equations in the trace.
  # The unnecessary equations are the equations
  # that do not contain the hidden states.
  trace.invar_needed_in_oth_eqns.clear()
  new_trace = []
  whole_trace_needed_vars = set(trace.hidden_vars)
  for eqn in reversed(trace.trace):
    need_outvars = []
    for outvar in eqn.outvars:
      if outvar in whole_trace_needed_vars:
        need_outvars.append(outvar)
    if len(need_outvars):
      for outvar in need_outvars:
        whole_trace_needed_vars.remove(outvar)
      new_trace.append(eqn)
      whole_trace_needed_vars.update([invar for invar in eqn.invars if isinstance(invar, jax.core.Var)])

  # [third step]
  # Finding out how the shape of each hidden state is converted to the size of df.
  hidden_vars = list(trace.hidden_vars)
  shape_mapping = []
  y = trace.y
  for hidden_var in hidden_vars:
    # The most direct way when the shapes of "y" and "hidden var" are the same is using "identity()" function.
    # However, there may be bugs, for examples, the output is reshaped to the same shape as the hidden state,
    # or, the split and concatenate operators are used while the shapes are the same between the outputs and
    # hidden states.
    # The most safe way is using automatic shape inverse transformation.
    #
    # However, the automatic inverse transformation may also cause errors, for example, if the following
    # operators are used:
    #     def f(a):
    #         s = jnp.sum(a, axis=[1,2], keepdims=True)
    #         return a / s
    #
    # this will result in the following jaxpr:
    #     { lambda ; a:f32[10,20,5]. let
    #         b:f32[10] = reduce_sum[axes=(1, 2)] a
    #         c:f32[10,1,1] = broadcast_in_dim[broadcast_dimensions=(0,) shape=(10, 1, 1)] b
    #         d:f32[10,20,5] = div a c
    #       in (d,) }
    #
    # It seems that the automatic shape inverse transformation is complex for handling such cases.\
    # Therefore, currently, we only consider the simple cases, and raise an error for the complex cases.

    if y.aval.shape == hidden_var.aval.shape:
      shape_mapping.append(identity)
      continue

    # automatic shape inverse transformation
    trace_need_vars = {hidden_var: [[]]}
    for eqn in new_trace:  # traveling the trace using the reversed order
      if len(eqn.outvars) == 1:
        outvar = eqn.outvars[0]
        if outvar in trace_need_vars:
          shaping = trace_need_vars.pop(outvar)
          rule = shape_changing_rule.get(eqn.primitive, None)
          if rule is None:
            for invar in eqn.invars:
              trace_need_vars[invar] = tuple(list(s) for s in shaping)
          else:
            for invar in eqn.invars:
              olds = tuple(list(s + [partial(rule, **eqn.params)]) for s in shaping)
              news = ([partial(rule, **eqn.params)],) if invar in trace_need_vars else tuple()
              trace_need_vars[invar] = olds + news
    print(trace_need_vars[y])
    if all(len(s) == 0 for s in trace_need_vars[y]):
      raise CompilationError(
        f'Error: The shape of the hidden state {hidden_var} is not converted to the size of the weight output {y}. \n'
        f'This may be caused by the complex shape conversion in the model. \n'
        f'Please report an issue to the developers at {git_issue_addr}. \n'
        f'Moreover, see the above traceback information for where the hidden state is defined in your code.'
      )
    shape_mapping.append(trace_need_vars[y])
    raise NotSupportedError

  # [fourth step]
  # Simplify the trace
  jaxpr_opt = jax.core.Jaxpr(
    # the const vars are not the hidden states, they are
    # intermediate data that are not used in the hidden states
    constvars=[nvar for nvar in whole_trace_needed_vars if nvar != trace.y],
    # the invars are always the weight output
    invars=[trace.y],
    # the outvars are always the connected hidden states of this weight
    outvars=hidden_vars,
    # the new equations which are simplified
    eqns=list(reversed(new_trace)),
  )

  # [final step]
  # Change the "WeightOpTracer" to "TracedWeightOp"
  return TracedWeightOp(weight=trace.weight,
                        op_jaxpr=jax_eqn_to_jaxpr(trace.op),
                        x=trace.x,
                        y=trace.y,
                        jaxpr_y2hid=jaxpr_opt,
                        hidden_vars=hidden_vars,
                        hidden2df=shape_mapping)


def jax_eqn_to_jaxpr(eqn: jax.core.JaxprEqn) -> jax.core.Jaxpr:
  """
  Convert the jax equation to the jaxpr.

  Args:
    eqn: The jax equation.

  Returns:
    The jaxpr.
  """
  return jax.core.Jaxpr(
    constvars=[],
    invars=eqn.invars,
    outvars=eqn.outvars,
    eqns=[eqn]
  )


def _get_element_primitive(eqn: jax.core.JaxprEqn) -> jax.core.Primitive:
  pass


def _fun_to_weight(jaxpr_x2hid, weight_val, hidden_i, hidden_grad, consts, x):
  _, f_vjp = jax.vjp(
    lambda weights: jax.core.eval_jaxpr(jaxpr_x2hid, consts, x, *jax.tree.leaves(weights))[hidden_i],
    weight_val
  )
  d_weights = f_vjp(hidden_grad)[0]
  return d_weights



class JaxprEvaluationForETraceRelation:
  """
  Evaluating the jaxpr for extracting the etrace (weight, hidden, operator) relationships.

  Args:
    jaxpr: The jaxpr for the model.
    weight_id_to_vars: The mapping from the weight id to the jax vars.
    invar_to_weight_id: The mapping from the jax var to the weight id.
    id_to_state: The mapping from the state id to the state.

  Returns:
    The list of the traced weight operations.
  """

  def __init__(
      self,
      jaxpr: jax.core.Jaxpr,
      weight_id_to_vars: Dict[WeightID, List[jax.core.Var]],
      invar_to_weight_id: Dict[jax.core.Var, WeightID],
      id_to_state: Dict[StateID, ETraceParam],
      hidden_invars: List[jax.core.Var],
      hidden_outvars: List[jax.core.Var],
  ):
    # the jaxpr of the original model, assuming that the model is well-defined,
    # see the doc for the model which can be online learning compiled.
    self.jaxpr = jaxpr

    #  the mapping from the weight id to the jax vars, one weight id may contain multiple jax vars
    self.weight_id_to_vars = weight_id_to_vars

    # the mapping from the jax var to the weight id, one jax var for one weight id
    self.invar_to_weight_id = invar_to_weight_id

    # the mapping from the state id to the state
    self.id_to_state = id_to_state

    # jax vars of hidden outputs
    self.hidden_outvars = set(hidden_outvars)

    # jax vars of weights
    self.weight_invars = set([v for vs in weight_id_to_vars.values() for v in vs])

    # jax vars of hidden states
    self.hidden_invars = set(hidden_invars)

  def compile(self) -> List[TracedWeightOp]:
    """
    Compiling the jaxpr for the etrace relationships.
    """

    # TODO:
    # - [x] Add the traceback information for the error messages. [done at 2024-04-06]
    # - [ ] Add the support for the scan, while, cond, pjit, and other operators.
    # - [ ] Add the support for the pytree inputs and outputs within one etrace operator.
    #       Currently, there is no need to consider this.
    # - [ ] If transformation is performed on weights and hiddens during the model computation,
    #       the weight has not a fixed in_var, and the hidden has not a fixed out_var.
    #       How to consider the intermediate data for such transformations?

    # the data structures for the tracing weights, variables and operations
    self.active_tracings: List[WeightOpTracer] = []

    # evaluating the jaxpr
    self._eval_jaxpr(self.jaxpr)

    # finalizing the traces
    final_traces = [_trace_simplify(trace) for trace in self.active_tracings]

    # reset the temporal data structures
    self.active_tracings = []
    return final_traces

  def _eval_jaxpr(self, jaxpr, invars_to_replace=None, outvars_to_replace=None) -> None:
    """
    Evaluating the jaxpr for extracting the etrace relationships.

    ``invars_to_replace`` and ``outvars_to_replace`` are not changing the semantics of the model,
    but are used for the replacement of the jax vars.

    Args:
      jaxpr: The jaxpr for the model.
      invars_to_replace: This means that the invar used in the equation should be directly
                         replaced by the given mapped var.
      outvars_to_replace: The means that the outvar used in the equation should be directly
                          replaced by the given mapped var.
    """
    for eqn in jaxpr.eqns:
      # TODO: add the support for the scan, while, cond, pjit, and other operators
      # Currently, scan, while, and cond are usually not the common operators used in
      # the definition of a brain dynamics model. So we may not need to consider them
      # during the current phase.
      # However, for the long-term maintenance and development, we need to consider them,
      # since users usually create crazy models.

      if invars_to_replace is not None:
        eqn = eqn.replace(invars=[invars_to_replace.get(invar, invar) for invar in eqn.invars])
      if outvars_to_replace is not None:
        eqn = eqn.replace(outvars=[outvars_to_replace.get(outvar, outvar) for outvar in eqn.outvars])

      if eqn.primitive.name == 'pjit':
        self._eval_pjit(eqn)
      elif eqn.primitive.name == 'scan':
        raise NotImplementedError
      elif eqn.primitive.name == 'while':
        raise NotImplementedError
      elif eqn.primitive.name == 'cond':
        raise NotImplementedError
      else:
        self._eval_eqn(eqn)

  def _eval_pjit(self, eqn: jax.core.JaxprEqn) -> None:
    """
    Evaluating the pjit primitive.
    """
    closed_jaxpr = eqn.params['jaxpr']
    if eqn.params['name'] == _etrace_op_name:
      # checking outvars
      if len(eqn.outvars) != 1:
        name_stack = source_info_util.current_name_stack() + eqn.source_info.name_stack
        with source_info_util.user_context(eqn.source_info.traceback, name_stack=name_stack):
          raise NotSupportedError(
            f'Currently, the etrace operator only supports single input and single output. \n'
            f'But we got {len(eqn.outvars)} outputs in the following operator: \n\n'
            f'{eqn} \n\n'
            f'You may need to define the operator as multiple operators, or raise an issue '
            f'to the developers at {git_issue_addr}. \n'
            f'Moreover, see the above traceback information for where the operation is defined in your code.'
          )

      # check old traces
      # If the old traces are valid, we add the new trace to the old
      # traces. If not, we remove the old traces.
      self._eval_old_traces_are_valid_or_not(eqn)

      # input, output, weight checking
      weight_id, x = self._get_state_and_inp_and_checking(eqn)

      # add new trace
      self.active_tracings.append(
        WeightOpTracer(
          weight=self.id_to_state[weight_id],
          x=x,
          y=eqn.outvars[0],
          # --- The jaxpr for the operator [TODO] checking whether there are bugs
          # Although the jaxpr var are not the same are the eqn var, we can still
          # use this closed jaxpr expression, since the ordering of the vars are the same.
          # Therefore, once the arguments and parameters are given correctly, the jaxpr
          # can be used to evaluate the same operator.
          # op=eqn.params['jaxpr'],
          # ---- changed it to the JaxprEqn (@chaoming0625, 16/04/2024)
          op=eqn,
          trace=[],  # the following eqns to hidden states
          hidden_vars=set(),  # the jax var of hidden states
          invar_needed_in_oth_eqns=set(eqn.outvars)  # temporary data for tracing eqn to hidden states
        )
      )
    else:
      if (
          _check_some_element_exist_in_the_set(eqn.invars, self.weight_invars)
          or
          _check_some_element_exist_in_the_set(eqn.outvars, self.hidden_invars)
      ):
        jaxpr = closed_jaxpr.jaxpr
        # TODO: checking whether there are bugs in the following mapping
        invar_to_replace = {var2: var1 for var1, var2 in zip(eqn.invars, jaxpr.invars) if var1 != var2}
        outvar_to_replace = {var2: var1 for var1, var2 in zip(eqn.outvars, jaxpr.outvars) if var1 != var2}
        self._eval_jaxpr(jaxpr, invar_to_replace, outvar_to_replace)
      else:
        # treat the pjit as a normal jaxpr equation
        self._eval_eqn(eqn)

  def _eval_eqn(self, eqn: jax.core.JaxprEqn) -> None:
    """
    Evaluating the normal jaxpr equation.
    """
    for trace in tuple(self.active_tracings):
      matched = _check_matched(eqn.invars, trace.invar_needed_in_oth_eqns)
      # if matched, add the eqn to the trace
      # if not matched, skip
      if len(matched):
        self._add_eqn_in_a_trace(eqn, trace)

  def _eval_old_traces_are_valid_or_not(self, eqn: jax.core.JaxprEqn) -> None:
    for trace in tuple(self.active_tracings):
      # Avoid "Weight -> Hidden -> Weight" pathway.
      # But the "Weight -> Weight -> Hidden" pathway is allowed.
      # However, it is hard to correctly handle the following pathways:
      #              Hidden -> Weight
      #            /
      #     Weight -> Weight -> Hidden
      #
      # This kind of connection pathways may also not be possible in real neural circuits.
      # But we need to consider the possibility of the existence of such pathways in the future (TODO).

      matched = _check_matched(eqn.invars, trace.invar_needed_in_oth_eqns)
      if len(matched) > 0:  # weight -> ? -> weight
        # TODO: how to judge this kind of pathway?
        # The current solution is only applied to the deep neural network models,
        # since the weights, hidden states, and operators are well-defined along the
        # depth. However, for a very complex recurrent graph models, the weights, hidden
        # states, and operators may be connected in a very complex way. Therefore, we
        # need to consider the handling of such complex models in the future.
        if len(trace.hidden_vars) > 0:  # weight -> hidden -> weight:
          pass
        else:  # weight -> weight -> ?
          self._add_eqn_in_a_trace(eqn, trace)

  def _add_eqn_in_a_trace(self, eqn: jax.core.JaxprEqn, trace: WeightOpTracer) -> None:
    trace.trace.append(eqn.replace())
    trace.invar_needed_in_oth_eqns.update(eqn.outvars)
    # check whether the hidden states are needed in the other equations
    for outvar in eqn.outvars:
      if outvar in self.hidden_outvars:
        trace.hidden_vars.add(outvar)

  def _get_state_and_inp_and_checking(self, eqn: jax.core.JaxprEqn) -> Tuple[WeightID, jax.core.Var]:
    # Currently, only single input/output are supported, i.e.,
    #       y = f(x, w1, w2, ...)
    # This may be changed in the future, to support multiple inputs and outputs, i.e.,
    #       y1, y2, ... = f(x1, x2, ..., w1, w2, ...)
    #
    # However, I do not see any possibility or necessity for this kind of design in the
    # current stage. In most situations, single input/output is enough for the brain dynamics model.

    found_invars_in_this_op = set()
    weight_ids = set()
    xs = []
    for invar in eqn.invars:
      weight_id = self.invar_to_weight_id.get(invar, None)
      if weight_id is None:
        xs.append(invar)
      else:
        weight_ids.add(weight_id)
        found_invars_in_this_op.add(invar)

    # --- checking whether the weight variables are all used in the same etrace operation --- #
    if len(weight_ids) == 0:
      name_stack = source_info_util.current_name_stack() + eqn.source_info.name_stack
      with source_info_util.user_context(eqn.source_info.traceback, name_stack=name_stack):
        raise CompilationError(
          f'Error: no ETraceParam are found in this operation: \n\n'
          f'{eqn}\n\n'
          f'See the above traceback information for where the operation is defined in your code.'
        )

    if len(weight_ids) > 1:
      name_stack = source_info_util.current_name_stack() + eqn.source_info.name_stack
      with source_info_util.user_context(eqn.source_info.traceback, name_stack=name_stack):
        raise CompilationError(
          f'Error: multiple ETraceParam ({weight_ids}) are found in this operation. '
          f'This is not allowed for automatic online learning: \n\n'
          f'{eqn}\n\n'
          f'See the above traceback information for where the operation is defined in your code.'
        )

    weight_id = tuple(weight_ids)[0]  # the only ETraceParam found in the operation
    if len(found_invars_in_this_op.difference(set(self.weight_id_to_vars[weight_id]))) > 0:
      name_stack = source_info_util.current_name_stack() + eqn.source_info.name_stack
      with source_info_util.user_context(eqn.source_info.traceback, name_stack=name_stack):
        raise CompilationError(
          f'Error: The found jax vars are {found_invars_in_this_op}, '
          f'but the ETraceParam contains vars {self.weight_id_to_vars[weight_id]}. \n'
          f'This means that the operator has used multiple ETraceParam. '
          f'Please define the trainable weights in a single ETraceParam. \n\n'
          f'{eqn}\n\n'
          f'See the above traceback information for where the operation is defined in your code.'
        )

    if len(xs) != 1:
      name_stack = source_info_util.current_name_stack() + eqn.source_info.name_stack
      with source_info_util.user_context(eqn.source_info.traceback, name_stack=name_stack):
        raise CompilationError(
          'Currently, the etrace operator only supports single input. \n'
          'You may need to define the model as multiple operators, or raise an issue '
          f'to the developers at {git_issue_addr}.\n\n'
          f'{eqn}\n\n'
          f'See the above traceback information for where the operation is defined in your code.'
        )

    # --- get the weight id and the input variable --- #
    return weight_id, xs[0]


class JaxprEvaluationForHiddenPerturbation:
  """
  Adding perturbations to the hidden states in the jaxpr, and replacing the hidden states with the perturbed states.

  Args:
    closed_jaxpr: The closed jaxpr for the model.
    hidden_outvars: The hidden state jax vars.
    outvar_to_state_id: The mapping from the outvar to the state id.
    id_to_state: The mapping from the state id to the state.
    hidden_invars: The hidden state invars.

  Returns:
    The revised closed jaxpr with the perturbations.
  """

  def __init__(
      self,
      closed_jaxpr: jax.core.ClosedJaxpr,
      hidden_outvars: List[jax.core.Var],
      outvar_to_state_id: Dict[jax.core.Var, StateID],
      id_to_state: Dict[StateID, ETraceVar],
      hidden_invars: List[jax.core.Var],
  ):
    # necessary data structures
    self.closed_jaxpr = closed_jaxpr
    self.hidden_outvars = hidden_outvars
    self.outvar_to_state_id = outvar_to_state_id
    self.id_to_state = id_to_state

    # other data structures
    self.hidden_invars = set(hidden_invars)

  def compile(self) -> jax.core.ClosedJaxpr:
    # new invars, the var order is the same as the hidden_outvars
    self.new_invars = {v: self._new_var_like(v) for v in self.hidden_outvars}

    # the hidden states that are not found in the code
    self.hidden_jaxvars_to_remove = set(self.hidden_outvars)

    # final revised equations
    self.revised_eqns = []

    # revising equations
    self._eval_jaxpr(self.closed_jaxpr.jaxpr)

    # [final checking]
    # If there are hidden states that are not found in the code, we raise an error.
    if len(self.hidden_jaxvars_to_remove) > 0:
      raise ValueError(
        f'Error: we did not found your defined hidden state (see the following information) in the code. \n'
        f'Please report an issue to the developers at {git_issue_addr}. \n'
        f'The missed hidden states are: \n'
        f'{state_traceback([self.id_to_state[self.outvar_to_state_id[v]] for v in self.hidden_jaxvars_to_remove])}'
      )

    # new jaxpr
    jaxpr = jax.core.Jaxpr(constvars=list(self.closed_jaxpr.jaxpr.constvars),
                           invars=list(self.closed_jaxpr.jaxpr.invars) + list(self.new_invars.values()),
                           outvars=list(self.closed_jaxpr.jaxpr.outvars),
                           eqns=self.revised_eqns)
    revised_closed_jaxpr = jax.core.ClosedJaxpr(jaxpr, self.closed_jaxpr.literals)

    # remove the temporal data
    self.new_invars = dict()
    self.revised_eqns = []
    self.hidden_jaxvars_to_remove = set()
    return revised_closed_jaxpr

  def _eval_jaxpr(self, jaxpr):
    for eqn in jaxpr.eqns:
      # TODO: add the support for the scan, while, cond, pjit, and other operators
      # If there are no hidden jaxpr vars are used in the equation,
      # then all we can treat it as a normal equation.
      # Therefore, the hidden jaxpr vars are the key to determine whether the equation
      # needs to be revised.

      if eqn.primitive.name == 'pjit':
        # TODO: how to rewrite pjit primitive?
        self.revised_eqns.append(eqn.replace())

      elif eqn.primitive.name == 'scan':
        if _check_some_element_exist_in_the_set(eqn.outvars, self.hidden_invars):
          raise NotImplementedError
        else:
          self.revised_eqns.append(eqn.replace())

      elif eqn.primitive.name == 'while':
        if _check_some_element_exist_in_the_set(eqn.outvars, self.hidden_invars):
          raise NotImplementedError
        else:
          self.revised_eqns.append(eqn.replace())

      elif eqn.primitive.name == 'cond':
        if _check_some_element_exist_in_the_set(eqn.outvars, self.hidden_invars):
          raise NotImplementedError
        else:
          self.revised_eqns.append(eqn.replace())

      else:
        self._eval_eqn(eqn)

  def _add_perturb_eqn(self, eqn: jax.core.JaxprEqn, perturb_var: jax.core.Var):
    # ------------------------------------------------
    # For the hidden var eqn, we want to add a perturbation:
    #    y = f(x)  =>  y = f(x) + perturb_var
    #
    # Particularly, we first define a new variable
    #    new_outvar = f(x)
    # Then, we add a new equation for the perturbation
    #    y = new_outvar + perturb_var
    # ------------------------------------------------

    hidden_var = eqn.outvars[0]

    # Frist step, define the hidden var as a new variable
    new_outvar = self._new_var_like(perturb_var)
    old_eqn = eqn.replace(outvars=[new_outvar])
    self.revised_eqns.append(old_eqn)

    # Second step, add the perturbation equation
    new_eqn = jax.core.JaxprEqn([new_outvar, perturb_var],
                                [hidden_var],
                                jax.lax.add_p,
                                {},
                                set(),
                                eqn.source_info.replace())
    self.revised_eqns.append(new_eqn)

  def _eval_eqn(self, eqn: jax.core.JaxprEqn):
    if len(eqn.outvars) == 1:
      if eqn.outvars[0] in self.hidden_jaxvars_to_remove:
        hidden_var = eqn.outvars[0]
        self.hidden_jaxvars_to_remove.remove(hidden_var)
        self._add_perturb_eqn(eqn, self.new_invars[hidden_var])
        return
    self.revised_eqns.append(eqn.replace())

  def _new_var_like(self, v):
    return jax.core.Var('', jax.core.ShapedArray(v.aval.shape, v.aval.dtype))


@jax.tree_util.register_pytree_node_class
class Residuals:
  """
  The residuals for storing the backward pass data in a VJP function.

  Args:
    jaxpr: The jaxpr for the backward pass.
    in_tree: The input tree structure.
    out_tree: The output tree structure.
    consts: The constants for the backward pass.
  """

  def __init__(self, jaxpr, in_tree, out_tree, consts):
    self.jaxpr = jaxpr
    self.in_tree = in_tree
    self.out_tree = out_tree
    self.consts = consts

  def __iter__(self):
    return iter((self.jaxpr, self.in_tree, self.out_tree, self.consts))

  def tree_flatten(self):
    return self.consts, (self.jaxpr, self.in_tree, self.out_tree)

  @classmethod
  def tree_unflatten(cls, aux, consts):
    jaxpr, in_tree, out_tree = aux
    return cls(jaxpr, in_tree, out_tree, consts)


class ETraceGraph:
  """
  The eligibility trace graph, tracking the relationship between the etrace weights
  :py:class:`ETraceParam`, the etrace variables :py:class:`ETraceVar`, and the etrace
  operations :py:class:`ETraceOp`.

  This class is used for computing the weight spatial gradients and the hidden state residuals.
  It is the most foundational data structure for the RTRL algorithm.

  It is important to note that the graph is built no matter whether the model is
  batched or not. This means that this graph can be applied to any kind of models.


  """
  __module__ = 'brainscale'

  # attributes for the graph
  out_all_jaxvars: List[jax.core.Var]  # all outvars except the function returns
  out_state_jaxvars: List[jax.core.Var]  # the state vars
  out_wx_jaxvars: List[jax.core.Var]  # the weight x
  out_wy2hid_jaxvars: List[jax.core.Var]  # the invars for computing df
  ret_num: int  # the number of function returns
  hidden_id_to_outvar: Dict[StateID, jax.core.Var]  # the hidden state's jax outvar

  # [ KEY ]
  # The most important data structure for the graph, which implementing
  # the relationship between the etrace weights and the etrace variables.
  weight_hidden_relations: List[TracedWeightOp]  # the traced weight operations

  def __init__(self, model: Callable):
    # The original model
    self.model = model

    # stateful model, for extracting states, weights, and variables
    # [ NOTE ]
    # The model does not support "static_argnums" for now.
    # Please always use ``functools.partial`` to fix the static arguments.
    self.stateful_model = bc.transform.StatefulFunction(model)

    # --- jaxpr for the model computation --- #
    # the revised jaxpr to return necessary variables
    self.revised_jaxpr: jax.core.ClosedJaxpr = None
    # the revised jaxpr with hidden state perturbations and return necessary variables
    self.revised_jaxpr_hidden_perturb: jax.core.ClosedJaxpr = None

  @property
  def states(self):
    """
    Getting the states of the model (all instances of ``braincore.State``).
    """
    return self.stateful_model.get_states()

  def _call_org_model(self, *args, **kwargs):
    """
    Calling the original model according to the given inputs and parameters.
    """
    return self.model(*args, **kwargs)

  def _call_org_model_with_jaxpr(self, *args):
    """
    Calling the original model with the model's jaxpr representation.
    """
    return self.stateful_model.jaxpr_call(*args)

  def compile_graph(self, *args, **kwargs):
    """
    Building the eligibility trace graph for the model according to the given inputs.
    """
    # NOTE:
    # The model does not support "static_argnums" for now.
    # Please always use functools.partial to fix the static arguments.
    self.stateful_model.make_jaxpr(*args, **kwargs)

    # -- states -- #
    states = self.stateful_model.get_states()
    id_to_state = {id(st): st for st in states}

    # -- jaxpr -- #
    closed_jaxpr = self.stateful_model.get_jaxpr()
    jaxpr = closed_jaxpr.jaxpr

    # -- finding the corresponding in/out vars of etrace states and weights -- #
    out_shapes = self.stateful_model.get_out_shapes()[0]
    state_vals = [state.value for state in states]
    arg_avals, _ = jax.tree.flatten((args, kwargs))
    ret_avals, _ = jax.tree.flatten(out_shapes)
    ret_num = len(ret_avals)
    state_avals, state_tree = jax.tree.flatten(state_vals)
    assert len(jaxpr.invars) == len(arg_avals) + len(state_avals)
    assert len(jaxpr.outvars) == ret_num + len(state_avals)
    self.ret_num = ret_num
    invars_with_state_tree = jax.tree.unflatten(state_tree, jaxpr.invars[len(arg_avals):])
    outvars_with_state_tree = jax.tree.unflatten(state_tree, jaxpr.outvars[ret_num:])

    # -- checking weights as invar -- #
    eweight_id_to_invar = {id(st): jax.tree.leaves(invar)
                           for invar, st in zip(invars_with_state_tree, states)
                           if isinstance(st, ETraceParam)}
    hidden_id_to_invar = {id(st): invar
                          for invar, st in zip(invars_with_state_tree, states)
                          if isinstance(st, ETraceVar)}  # ETraceVar only contains one Array, "invar" is the jaxpr var
    invar_to_eweight_id = {v: k for k, vs in eweight_id_to_invar.items() for v in vs}
    self.eweight_id_to_invar = eweight_id_to_invar

    # -- checking states as outvar -- #
    hidden_id_to_outvar = {
      id(st): outvar
      for outvar, st in zip(outvars_with_state_tree, states)
      if isinstance(st, ETraceVar)
    }  # ETraceVar only contains one Array, "outvar" is the jaxpr var
    outvar_to_state_id = {v: state_id for state_id, v in hidden_id_to_outvar.items()}
    self.hidden_id_to_outvar = hidden_id_to_outvar

    # -- evaluating the jaxpr for the etrace relationships -- #
    evaluator = JaxprEvaluationForETraceRelation(
      jaxpr=jaxpr,
      weight_id_to_vars=eweight_id_to_invar,
      invar_to_weight_id=invar_to_eweight_id,
      id_to_state=id_to_state,
      hidden_invars=list(hidden_id_to_invar.values()),
      hidden_outvars=list(hidden_id_to_outvar.values()),
    )
    weight_hidden_relation = evaluator.compile()
    self.weight_hidden_relations = weight_hidden_relation

    # --- Collect the Var needed to compute the weight spatial gradients --- #
    # ---      Rewrite the jaxpr for computing the needed variables      --- #

    # -- new outvars -- #
    self.out_state_jaxvars = list(jaxpr.outvars[ret_num:])  # all states jaxpr var
    weight_jaxvar_tree, hidden_jaxvar, other_state_jaxvar_tree = split_state_values(states, outvars_with_state_tree)
    self.out_weight_jaxvars = jax.tree.leaves(weight_jaxvar_tree)  # all weight jaxpr var
    self.out_hidden_jaxvars = list(hidden_jaxvar)  # all hidden jaxpr var, one etrace to one Array
    self.out_othstate_jaxvars = jax.tree.leaves(other_state_jaxvar_tree)  # all other states jaxpr var
    self.out_wx_jaxvars = list(set([relation.x for relation in weight_hidden_relation]))  # all weight x
    self.out_wy2hid_jaxvars = list(
      set([v
           for relation in weight_hidden_relation
           for v in (relation.jaxpr_y2hid.invars + relation.jaxpr_y2hid.constvars)])
    )  # all y-to-hidden vars
    all_outvars = list(set(self.out_state_jaxvars + self.out_wx_jaxvars + self.out_wy2hid_jaxvars))
    self.out_all_jaxvars = jaxpr.outvars[:ret_num] + all_outvars

    # new jaxpr
    jaxpr = jax.core.Jaxpr(constvars=list(jaxpr.constvars),
                           invars=list(jaxpr.invars),
                           outvars=list(self.out_all_jaxvars),
                           eqns=list(jaxpr.eqns))
    closed_jaxpr = jax.core.ClosedJaxpr(jaxpr, closed_jaxpr.consts)
    self.revised_jaxpr = closed_jaxpr

    # ---               add perturbations to the hidden states                  --- #
    # --- new jaxpr with hidden state perturbations for computing the residuals --- #
    evaluator2 = JaxprEvaluationForHiddenPerturbation(
      closed_jaxpr=closed_jaxpr,
      hidden_outvars=self.out_hidden_jaxvars,
      outvar_to_state_id=outvar_to_state_id,
      id_to_state=id_to_state,
      hidden_invars=list(hidden_id_to_invar.values()),
    )
    self.revised_jaxpr_hidden_perturb = evaluator2.compile()

    return self

  def show_graph(self):
    pass

  def _jaxpr_compute_model(self, *args, **kwargs) -> Tuple[PyTree, HiddenVals, StateVals, TempData]:
    """
    Computing the model according to the given inputs and parameters by using the compiled jaxpr.
    """
    # state checking
    old_state_vals = [st.value for st in self.stateful_model.get_states()]

    # parameters
    args = jax.tree.flatten((args, kwargs, old_state_vals))[0]

    # calling the function
    jaxpr_outs = jax.core.eval_jaxpr(self.revised_jaxpr.jaxpr, self.revised_jaxpr.consts, *args)

    # intermediate values
    temps = {v: r for v, r in zip(self.out_all_jaxvars[self.ret_num:], jaxpr_outs[self.ret_num:])}

    # recovery outputs of ``stateful_model``
    state_outs = [temps[v] for v in self.out_state_jaxvars]
    out, new_state_vals = self.stateful_model.get_out_treedef().unflatten(jaxpr_outs[:self.ret_num] + state_outs)

    # state value assignment
    assert len(old_state_vals) == len(new_state_vals), 'State length mismatch.'
    # TODO: [KEY] assuming that the weight values are not changed
    hidden_vals, oth_state_vals = split_state_values(self.stateful_model.get_states(),
                                                     new_state_vals,
                                                     include_weight=False)
    return out, hidden_vals, oth_state_vals, temps

  def _jaxpr_compute_vjp_model(self, *args) -> Tuple[PyTree, HiddenVals, StateVals, TempData, Residuals]:
    """
    Computing the VJP transformed model according to the given inputs and parameters by using the compiled jaxpr.
    """
    _, hidden_states, non_etrace_weight_states, other_states = split_states_v2(self.stateful_model.get_states())

    def fun_for_vjp(inputs, hiddens, non_etrace_weights, oth_states, perturbs):
      # assign and get state values
      assign_state_values(hidden_states, hiddens)
      assign_state_values(non_etrace_weight_states, non_etrace_weights)
      assign_state_values(other_states, oth_states)
      old_state_vals = [st.value for st in self.stateful_model.get_states()]

      # calling the function
      jaxpr_outs = jax.core.eval_jaxpr(self.revised_jaxpr_hidden_perturb.jaxpr,
                                       self.revised_jaxpr_hidden_perturb.consts,
                                       *jax.tree.leaves((inputs, old_state_vals, perturbs)))

      # intermediate values
      temps = {v: r for v, r in zip(self.out_all_jaxvars[self.ret_num:], jaxpr_outs[self.ret_num:])}

      # outputs
      state_outs = [temps[v] for v in self.out_state_jaxvars]
      out, new_state_vals = self.stateful_model.get_out_treedef().unflatten(jaxpr_outs[:self.ret_num] + state_outs)
      new_hiddens, new_others = split_state_values(self.stateful_model.get_states(),
                                                   new_state_vals,
                                                   include_weight=False)
      return (out, new_hiddens, new_others), temps

    # TODO:
    #  [KEY] assuming that the weight values (including etrace weights and normal param weights) are not changed
    hidden_perturbs = [jnp.zeros(v.aval.shape, v.aval.dtype) for v in self.out_hidden_jaxvars]
    hidden_vals = [st.value for st in hidden_states]
    non_etrace_weight_vals = [st.value for st in non_etrace_weight_states]
    other_vals = [st.value for st in other_states]
    # VJP calling
    (out, hidden_vals, other_vals), f_vjp, temps = jax.vjp(
      fun_for_vjp,  # the function
      args, hidden_vals, non_etrace_weight_vals, other_vals, hidden_perturbs,  # the inputs
      has_aux=True
    )
    out_flat, out_tree = jax.tree.flatten(((out, hidden_vals, other_vals),))
    rule, in_tree = jax.api_util.flatten_fun_nokwargs(lu.wrap_init(f_vjp), out_tree)
    out_avals = [jax.core.get_aval(x).at_least_vspace() for x in out_flat]
    jaxpr, _, consts = pe.trace_to_jaxpr_dynamic(rule, out_avals)
    # recovering the other non-etrace weights, although the weights are not changed
    assign_state_values(non_etrace_weight_states, non_etrace_weight_vals)
    return out, hidden_vals, other_vals, temps, Residuals(jaxpr, in_tree(), out_tree, consts)

  def _compute_hid2weight_jacobian(
      self, intermediate_values: dict
  ) -> Tuple[Dict[WeightXVar, jax.Array], Dict[Tuple[WeightYVar, HiddenVar], jax.Array]]:
    """
    Computing the weight x and df values for the spatial gradients.

    Args:
      intermediate_values: The intermediate values of the model.

    Returns:
      The weight x and df values.
    """
    # the weight x
    xs = {v: intermediate_values[v] for v in self.out_wx_jaxvars}

    # the weight df
    dfs = dict()
    for relation in self.weight_hidden_relations:
      consts = [intermediate_values[var] for var in relation.jaxpr_y2hid.constvars]
      invars = [intermediate_values[var] for var in relation.jaxpr_y2hid.invars]  # weight y
      outvar_grads = [jnp.ones(v.aval.shape, v.aval.dtype) for v in relation.jaxpr_y2hid.outvars]  # hidden states

      # [ KEY ]
      #
      # # ---- Method 1: using ``backward_pass`` ---- #
      # Assuming the function is linear. One cheap way is to use
      # the backward pass for computing the gradients of the hidden states.
      # For most situations, the ``y --> hidden`` relation is linear. Therefore,
      # we use ``backward_pass`` to compute the ``Df`` while avoids the overhead
      # of computing the forward pass. Otherwise, we should use ``jax.vjp`` instead.
      # Please also see ``jax.linear_transpose()`` for the same purpose.
      #
      # [df] = backward_pass(relation.jaxpr_y2hid, [], True, consts, invars, outvars)
      #
      # # ---- Method 2: using ``jax.vjp`` ---- #
      # However, for general cases, we choose to use ``jax.vjp`` to compute the gradients.
      #
      # # ---- Method 3: using ``jax.jvp`` ---- #
      assert len(invars) == 1
      _, f_vjp = jax.vjp(lambda x: jax.core.eval_jaxpr(relation.jaxpr_y2hid, consts, x), invars[0])
      df = f_vjp(outvar_grads)[0]

      # get the df we want
      dfs[relation.y] = df

    # all x and df values
    return xs, dfs

  def solve_spatial_gradients(
      self, *args
  ) -> Tuple[Outputs, HiddenVals, StateVals, ETraceVals]:
    """
    Solving the spatial gradients of the weights according to the given inputs and parameters.

    Args:
      *args: The positional arguments for the model.

    Returns:
      The outputs, hidden states, other states, and the spatial gradients of the weights.
    """
    # --- compile the model --- #
    if self.revised_jaxpr is None:
      raise ValueError('The ETraceGraph object has not been built yet.')

    # --- call the model --- #
    out, hiddens, others, temps = self._jaxpr_compute_model(*args)
    hid2weight_jac = self._compute_hid2weight_jacobian(temps)
    return out, hiddens, others, hid2weight_jac

  def solve_spatial_gradients_and_vjp_jaxpr(
      self, *args,
  ) -> Tuple[Outputs, HiddenVals, StateVals, ETraceVals, Residuals]:
    """
    Solving the spatial gradients of the weights and the VJP transformed model according to the given inputs.

    Args:
      *args: The positional arguments for the model.

    Returns:
      The outputs, hidden states, other states, the spatial gradients of the weights, and the residuals.
    """
    if self.revised_jaxpr_hidden_perturb is None:
      raise ValueError('The ETraceGraph object has not been built yet.')

    # --- call the model --- #
    out, hidden_vals, other_vals, temps, vjp_residual = self._jaxpr_compute_vjp_model(*args)
    hid2weight_jac = self._compute_hid2weight_jacobian(temps)
    return out, hidden_vals, other_vals, hid2weight_jac, vjp_residual


@set_module_as('brainscale')
def build_etrace_graph(model, *args, **kwargs) -> ETraceGraph:
  """
  Build the eligibility trace graph of the given model.

  The eligibility trace graph is used to compute the model gradients, including

  - the spatial gradients of the weights
  - the VJP gradients of the etrace variables

  Args:
    model: The model function. Can be any Python callable function.
    *args: The positional arguments for the model.
    **kwargs: The keyword arguments for the model.

  Returns:
    The eligibility trace graph.
  """
  etrace_graph = ETraceGraph(model)
  etrace_graph.compile_graph(*args, **kwargs)
  return etrace_graph

