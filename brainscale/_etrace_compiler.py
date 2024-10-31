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

import itertools
from enum import Enum
from functools import partial
from typing import (Callable, NamedTuple, List, Dict, Sequence, Tuple, Set, Optional)

import brainstate as bst
import brainunit as u
import jax.core
from jax.extend import linear_util as lu
from jax.extend import source_info_util
from jax.interpreters import partial_eval as pe

from ._etrace_concepts import (assign_state_values,
                               split_states_v2)
from ._etrace_concepts import (is_etrace_op,
                               is_etrace_op_enable_gradient,
                               ETraceParam,
                               ETraceState)
from ._jaxpr_to_source_code import jaxpr_to_python_code
from ._misc import (git_issue_addr,
                    state_traceback,
                    set_module_as,
                    NotSupportedError,
                    CompilationError,
                    BaseEnum)
from ._typing import (PyTree,
                      StateID,
                      WeightID,
                      WeightXVar,
                      WeightYVar,
                      HiddenInVar,
                      HiddenOutVar,
                      TempData,
                      Outputs,
                      HiddenVals,
                      StateVals,
                      WeightVals,
                      Hid2WeightJacobian,
                      Hid2HidJacobian)

# TODO
#
# - [x] The visualization of the etrace graph.
# - [ ] Judge whether the `df` is the same for different weight y.
#       For example,
#
#          h = f(x1 @ w1 + x2 @ w2)
#
#       The `df` for w1 and w2 are the same, although them have the different weight y.

__all__ = [
    'ETraceGraph', 'build_etrace_graph',
]


def _remove_quantity(tree):
    """
    Remove the quantity from the tree.

    Args:
      tree: The tree.

    Returns:
      The tree without the quantity.
    """

    def fn(x):
        if isinstance(x, u.Quantity):
            return x.magnitude
        return x

    return jax.tree_map(fn, tree, is_leaf=lambda x: isinstance(x, u.Quantity))


def indent_code(code: str, indent: int = 2) -> str:
    """
    Indent the code.

    Args:
      code: The code to be indented.
      indent: The number of spaces to indent.

    Returns:
      The indented code.
    """
    return '\n'.join(' ' * indent + line for line in code.split('\n'))


def _identity(x):
    """
    Identity function. Return x
    """
    return x


def split_state_values(
    states: Sequence[bst.State],
    state_values: List[PyTree],
    include_weight: bool = True
) -> (Tuple[WeightVals, HiddenVals, StateVals] | Tuple[HiddenVals, StateVals]):
    """
    Split the state values into the weight values, the hidden values, and the other state values.

    The weight values are the values of the ``braincore.ParamState`` states (including ``ETraceParam``).
    The hidden values are the values of the ``ETraceState`` states.
    The other state values are the values of the other states.

    Parameters:
    -----------
    states: Sequence[bst.State]
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
            if isinstance(st, bst.ParamState):
                weight_vals.append(val)
            elif isinstance(st, ETraceState):
                hidden_vals.append(val)
            else:
                other_vals.append(val)
        return weight_vals, hidden_vals, other_vals
    else:
        hidden_vals, other_vals = [], []
        for st, val in zip(states, state_values):
            if isinstance(st, bst.ParamState):
                pass
            elif isinstance(st, ETraceState):
                hidden_vals.append(val)
            else:
                other_vals.append(val)
        return hidden_vals, other_vals


class HiddenToHiddensTracer(NamedTuple):
    """
    The data structure for the tracing of the hidden-to-hidden states.
    """
    hidden_invar: jax.core.Var
    connected_hidden_outvars: set[jax.core.Var]
    other_invars: set[jax.core.Var]
    invar_needed_in_oth_eqns: set[jax.core.Var]
    trace: List[jax.core.JaxprEqn]


class HiddenWeightOpTracer(NamedTuple):
    """
    The data structure for the tracing of the ETraceParam operation.
    """
    op: jax.core.JaxprEqn  # f: how x is transformed into y, i.e., y = f(x, w)
    weight: ETraceParam  # w
    x: jax.core.Var  # y
    y: jax.core.Var  # x
    trace: List[jax.core.JaxprEqn]
    hidden_vars: set[jax.core.Var]
    invar_needed_in_oth_eqns: set[jax.core.Var]

    def replace(
        self,
        weight=None,
        op=None,
        x=None,
        y=None,
        trace=None,
        hidden_vars=None,
        invar_needed_in_oth_eqns=None
    ):
        return HiddenWeightOpTracer(
            op=(op if op is not None else self.op),
            weight=(weight if weight is not None else self.weight),
            x=(x if x is not None else self.x),
            y=(y if y is not None else self.y),
            trace=(trace if trace is not None else self.trace),
            hidden_vars=(hidden_vars if hidden_vars is not None else self.hidden_vars),
            invar_needed_in_oth_eqns=(invar_needed_in_oth_eqns
                                      if invar_needed_in_oth_eqns is not None
                                      else self.invar_needed_in_oth_eqns)
        )


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


def _post_check(trace):
    # Check the hidden states of the given weight. If the hidden states are not
    # used in the model, we raise an error. This is to avoid the situation that
    # the weight is defined but not used in the model.
    if len(trace.hidden_vars) == 0:
        source_info = trace.weight.source_info
        name_stack = source_info_util.current_name_stack() + source_info.name_stack
        with source_info_util.user_context(source_info.traceback, name_stack=name_stack):
            # TODO:
            #    raise error or change it as non-etrace-weight
            raise CompilationError(
                f'Error: The ETraceParam {trace.weight} does not found the associated hidden states: \n'
                f'There are maybe two kinds of reasons: '
                f'1. The weight is not associated with any hidden states. Therefore it should not be defined \n'
                f'   as a {ETraceParam.__name__}. You can turn off ETrace learning for this weight by setting \n'
                f'   "as_etrace_weight=False". For example, \n'
                f'       \n'
                f'        lin = brainscale.Linear(..., as_etrace_weight=False)". \n'
                f'2. This may be a compilation error. Please report an issue to the developers at {git_issue_addr}. \n'
                f'\n'
                f'Moreover, see the above traceback information for where the weight is defined in your code.'
            )
    return trace


def _simplify_hid2hid_tracer(
    tracer: HiddenToHiddensTracer,
    hidden_invar_to_hidden,
    hidden_outvar_to_hidden,
) -> HiddenTransition:
    # [first step]
    # Remove the unnecessary equations in the trace.
    # The unnecessary equations are the equations
    # that do not contain the hidden states.
    tracer.invar_needed_in_oth_eqns.clear()
    new_trace = []
    whole_trace_needed_vars = set(tracer.connected_hidden_outvars)
    visited_needed_vars = set()  # needed_vars has been satisfied
    for eqn in reversed(tracer.trace):
        need_outvars = []
        for outvar in eqn.outvars:
            if outvar in whole_trace_needed_vars:
                need_outvars.append(outvar)
        if len(need_outvars):
            visited_needed_vars.update(need_outvars)
            new_trace.append(eqn)
            whole_trace_needed_vars.update([invar for invar in eqn.invars if isinstance(invar, jax.core.Var)])

    # [second step]
    # Checking whether the shape of each hidden state is consistent.
    # Currently, we only support the element-wise state transition.
    hidden_outvars = tuple(tracer.connected_hidden_outvars)
    for hidden_var in hidden_outvars:
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

        if tracer.hidden_invar.aval.shape != hidden_var.aval.shape:
            raise NotSupportedError(
                f'Currently, we only support the state group that hase the same shape. \n'
                f'However, we got {[var.aval for var in hidden_outvars + (tracer.hidden_invar,)]}'
            )

    # [third step]
    # Simplify the trace
    visited_needed_vars.add(tracer.hidden_invar)
    constvars = list(whole_trace_needed_vars.difference(visited_needed_vars))
    jaxpr_opt = jax.core.Jaxpr(
        # the const vars are not the hidden states, they are
        # intermediate data that are not used in the hidden states
        constvars=constvars,
        # the invars are always the weight output
        invars=[tracer.hidden_invar],
        # the outvars are always the connected hidden states of this weight
        outvars=list(hidden_outvars),
        # the new equations which are simplified
        eqns=list(reversed(new_trace)),
    )

    # [final step]
    # Change the "HiddenWeightOpTracer" to "HiddenWeightOpRelation"
    return HiddenTransition(
        hidden_invar=tracer.hidden_invar,
        hidden=hidden_invar_to_hidden[tracer.hidden_invar],
        connected_hidden_outvars=list(hidden_outvars),
        connected_hiddens=[hidden_outvar_to_hidden[var] for var in hidden_outvars],
        jaxpr=jaxpr_opt,
        other_invars=constvars,
    )


def _trace_simplify(
    tracer: HiddenWeightOpTracer,
    hidden_outvar_to_group: Dict[HiddenOutVar, 'HiddenGroup'],
    hidden_outvar_to_transition: Dict[HiddenOutVar, 'HiddenTransition'],
) -> HiddenWeightOpRelation:
    """
    Simplifying the trace from the weight output to the hidden state.

    Args:
      tracer: The traced weight operation.

    Returns:
      The simplified traced weight operation.
    """
    # [first step]
    _post_check(tracer)

    # [second step]
    # Remove the unnecessary equations in the trace.
    # The unnecessary equations are the equations
    # that do not contain the hidden states.
    tracer.invar_needed_in_oth_eqns.clear()
    new_trace = []
    whole_trace_needed_vars = set(tracer.hidden_vars)
    visited_needed_vars = set()
    for eqn in reversed(tracer.trace):
        need_outvars = []
        for outvar in eqn.outvars:
            if outvar in whole_trace_needed_vars:
                need_outvars.append(outvar)
        if len(need_outvars):
            for outvar in need_outvars:
                visited_needed_vars.add(outvar)
            new_trace.append(eqn)
            whole_trace_needed_vars.update([invar for invar in eqn.invars if isinstance(invar, jax.core.Var)])

    # [third step]
    # Finding out how the shape of each hidden state is converted to the size of df.
    connected_hidden_vars = list(tracer.hidden_vars)
    y = tracer.y
    for hidden_var in connected_hidden_vars:
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
            continue
        else:
            raise NotSupportedError(
                f'Currently, the automatic shape inverse transformation is not supported. \n'
            )

    # [fourth step]
    # Simplify the trace
    visited_needed_vars.add(tracer.y)
    jaxpr_opt = jax.core.Jaxpr(
        # the const vars are not the hidden states, they are
        # intermediate data that are not used in the hidden states
        constvars=[nvar for nvar in whole_trace_needed_vars.difference(visited_needed_vars)],
        # the invars are always the weight output
        invars=[tracer.y],
        # the outvars are always the connected hidden states of this weight
        outvars=connected_hidden_vars,
        # the new equations which are simplified
        eqns=list(reversed(new_trace)),
    )

    # [final step]
    # Change the "HiddenWeightOpTracer" to "HiddenWeightOpRelation"
    hidden_group_ids = set()
    hidden_group_mappings = dict()
    for hidden_var in connected_hidden_vars:
        group = hidden_outvar_to_group[hidden_var]
        group_id = id(group)
        hidden_group_ids.add(group_id)
        hidden_group_mappings[group_id] = group
    hidden_var_to_transition = {
        hidden_var: hidden_outvar_to_transition[hidden_var]
        for hidden_var in connected_hidden_vars
    }

    return HiddenWeightOpRelation(
        weight=tracer.weight,
        op_jaxpr=_jax_eqn_to_jaxpr(tracer.op),
        x=tracer.x,
        y=tracer.y,
        jaxpr_y2hid=jaxpr_opt,
        hidden_vars=connected_hidden_vars,
        hidden_groups=[hidden_group_mappings[gid] for gid in hidden_group_ids],
        hidden_var_to_transition=hidden_var_to_transition
    )


def _jax_eqn_to_jaxpr(eqn: jax.core.JaxprEqn) -> jax.core.Jaxpr:
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


class JaxprEvaluationForHiddenWeightOpRelation:
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
        hidden_outvar_to_group: Dict[HiddenOutVar, 'HiddenGroup'],
        hidden_outvar_to_transition: Dict[HiddenOutVar, 'HiddenTransition'],
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

        self.hidden_outvar_to_group = hidden_outvar_to_group
        self.hidden_outvar_to_transition = hidden_outvar_to_transition

    def compile(self) -> Tuple[HiddenWeightOpRelation, ...]:
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
        self.active_tracings: List[HiddenWeightOpTracer] = []

        # evaluating the jaxpr
        self._eval_jaxpr(self.jaxpr)

        # finalizing the traces
        final_traces = [
            _trace_simplify(_post_check(trace),
                            self.hidden_outvar_to_group,
                            self.hidden_outvar_to_transition)
            for trace in self.active_tracings
        ]

        # reset the temporal data structures
        self.active_tracings = []
        return tuple(final_traces)

    def _eval_jaxpr(
        self,
        jaxpr,
        invars_to_replace=None,
        outvars_to_replace=None
    ) -> None:
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
                if _check_some_element_exist_in_the_set(eqn.invars, self.hidden_invars):
                    raise NotImplementedError(
                        f'Currently, brainscale does not support the "scan" operator with hidden states. '
                        f'Please raise an issue or feature request to the developers at {git_issue_addr}.'
                    )
                self._eval_eqn(eqn)
            elif eqn.primitive.name == 'while':
                if _check_some_element_exist_in_the_set(eqn.invars, self.hidden_invars):
                    raise NotImplementedError(
                        f'Currently, brainscale does not support the "while" operator with hidden states. '
                        f'Please raise an issue or feature request to the developers at {git_issue_addr}.'
                    )
                self._eval_eqn(eqn)
            elif eqn.primitive.name == 'cond':
                if _check_some_element_exist_in_the_set(eqn.invars, self.hidden_invars):
                    raise NotImplementedError(
                        f'Currently, brainscale does not support the "cond" operator with hidden states. '
                        f'Please raise an issue or feature request to the developers at {git_issue_addr}.'
                    )
                self._eval_eqn(eqn)
            else:
                self._eval_eqn(eqn)

    def _eval_pjit(self, eqn: jax.core.JaxprEqn) -> None:
        """
        Evaluating the pjit primitive.
        """
        closed_jaxpr = eqn.params['jaxpr']
        if is_etrace_op(eqn.params['name']):  # is etrace operator
            # checking outvars
            if len(eqn.outvars) != 1:
                name_stack = source_info_util.current_name_stack() + eqn.source_info.name_stack
                with source_info_util.user_context(eqn.source_info.traceback, name_stack=name_stack):
                    raise NotSupportedError(
                        f'Currently, the etrace operator only supports single input and single output. \n'
                        f'But we got {len(eqn.outvars)} outputs in the following operator: \n\n'
                        f'The Jaxpr for the operator: \n\n'
                        f'{eqn} \n\n'
                        f'The corresponding Python code for the operator: \n\n'
                        f'{jaxpr_to_python_code(_jax_eqn_to_jaxpr(eqn))} \n\n'
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
                HiddenWeightOpTracer(
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
        else:  # not etrace operator
            if (
                # checking whether the weight variables are used in the pjit
                _check_some_element_exist_in_the_set(eqn.invars, self.weight_invars)
                or
                # checking whether the hidden variables are computed in the pjit
                _check_some_element_exist_in_the_set(eqn.outvars, self.hidden_outvars)
            ):
                # TODO: checking whether there are bugs in the following mapping
                jaxpr = closed_jaxpr.jaxpr
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
        if eqn.primitive.name == 'stop_gradient':
            return
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

    def _add_eqn_in_a_trace(self, eqn: jax.core.JaxprEqn, trace: HiddenWeightOpTracer) -> None:
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
                    f'The Jaxpr for the operator: \n\n'
                    f'{eqn} \n\n'
                    f'The corresponding Python code for the operator: \n\n'
                    f'{jaxpr_to_python_code(_jax_eqn_to_jaxpr(eqn))} \n\n'
                    f'See the above traceback information for where the operation is defined in your code.'
                )

        if len(weight_ids) > 1:
            name_stack = source_info_util.current_name_stack() + eqn.source_info.name_stack
            with source_info_util.user_context(eqn.source_info.traceback, name_stack=name_stack):
                raise CompilationError(
                    f'Error: multiple ETraceParam ({weight_ids}) are found in this operation. '
                    f'This is not allowed for automatic online learning: \n\n'
                    f'The Jaxpr for the operator: \n\n'
                    f'{eqn} \n\n'
                    f'The corresponding Python code for the operator: \n\n'
                    f'{jaxpr_to_python_code(_jax_eqn_to_jaxpr(eqn))} \n\n'
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
                    f'The Jaxpr for the operator: \n\n'
                    f'{eqn} \n\n'
                    f'The corresponding Python code for the operator: \n\n'
                    f'{jaxpr_to_python_code(_jax_eqn_to_jaxpr(eqn))} \n\n'
                    f'See the above traceback information for where the operation is defined in your code.'
                )

        if len(xs) != 1:
            name_stack = source_info_util.current_name_stack() + eqn.source_info.name_stack
            with source_info_util.user_context(eqn.source_info.traceback, name_stack=name_stack):
                raise CompilationError(
                    'Currently, the etrace operator only supports single input. \n'
                    'You may need to define the model as multiple operators, or raise an issue '
                    f'to the developers at {git_issue_addr}.\n\n'
                    f'The Jaxpr for the operator: \n\n'
                    f'{eqn} \n\n'
                    f'The corresponding Python code for the operator: \n\n'
                    f'{jaxpr_to_python_code(_jax_eqn_to_jaxpr(eqn))} \n\n'
                    f'See the above traceback information for where the operation is defined in your code.'
                )

        # --- get the weight id and the input variable --- #
        return weight_id, xs[0]


def _hpo_tracer_to_relation(
    hid_relation: HiddenGroup,
    hpo_tracer: HiddenWeightOpTracer
) -> HiddenWeightOpRelation:
    hpo_tracer = hpo_tracer.replace(hidden_vars=list(hid_relation.hidden_outvars))
    return _trace_simplify(hpo_tracer)


def _simplify_hidden_eqns(hidden_invars: List[HiddenInVar],
                          hidden_outvars: List[HiddenOutVar],
                          eqns: List[jax.core.JaxprEqn]):
    # remove the unnecessary equations in the trace
    true_eqns = []
    true_hidden_outvars = set()
    true_hidden_invars = set()
    all_invars = set()
    dependent_vars = set(hidden_invars)
    for eqn in eqns:
        temp_vars = []
        true_invars = []
        for invar in eqn.invars:
            if not isinstance(invar, jax.core.Literal):
                if invar in dependent_vars:
                    temp_vars.append(invar)
                true_invars.append(invar)
        if len(temp_vars):
            true_eqns.append(eqn.replace())
            all_invars.update(true_invars)
            dependent_vars.update(eqn.outvars)
            for outvar in eqn.outvars:
                if outvar in hidden_outvars:
                    true_hidden_outvars.add(outvar)
            for invar in true_invars:
                if invar in hidden_invars:
                    true_hidden_invars.add(invar)

    const_vars = list(all_invars.difference(dependent_vars))
    return true_hidden_invars, list(true_hidden_outvars), const_vars, true_eqns


def _format_and_optimize_jaxpr(
    hidden_invars: List[HiddenInVar],
    hidden_outvars: List[HiddenOutVar],
    hidden_outvar_to_invar: Dict,
    hidden_invar_to_outvar: Dict,
    eqns: List[jax.core.JaxprEqn]
) -> Optional[jax.core.Jaxpr]:
    #
    # Several additional things need to pay attention to:
    #
    # 1. Although we found that the weight are associated with the hidden states, the
    #    hidden states may not have the diagonal Jacobian matrix. Therefore, such weight
    #    should be changed to the non-temporal learning weight.
    #
    # 2. There are multiple associated hidden states for one weight. However, some hidden
    #    states does not have the diagonal interaction. Therefore, we need to remove this
    #    hidden states since they should not be associated with the weight.
    #
    # 3. The found "true_hidden_invars" is different from "true_hidden_outvars"
    #
    true_hidden_invars, true_hidden_outvars, constvars, true_eqns = (
        _simplify_hidden_eqns(hidden_invars, hidden_outvars, eqns))
    cor_hidden_invars = [hidden_outvar_to_invar[outvar] for outvar in true_hidden_outvars]
    while set(cor_hidden_invars) != true_hidden_invars:
        hidden_invars = set(cor_hidden_invars).intersection(true_hidden_invars)
        hidden_outvars = [hidden_invar_to_outvar[invar] for invar in hidden_invars]
        true_hidden_invars, true_hidden_outvars, constvars, true_eqns = (
            _simplify_hidden_eqns(hidden_invars, hidden_outvars, eqns))
        cor_hidden_invars = [hidden_outvar_to_invar[outvar] for outvar in true_hidden_outvars]

    if len(true_eqns) == 0:
        return None

    if len(true_hidden_outvars) == 0:
        return None

    # the jaxpr
    jaxpr = jax.core.Jaxpr(
        # the const vars are not the hidden states, they are
        # intermediate data that are not used in the hidden states
        constvars=constvars,
        # the invars are always the weight output
        invars=list(cor_hidden_invars),
        # the outvars are always the connected hidden states of this weight
        outvars=list(true_hidden_outvars),
        # the new equations which are simplified
        eqns=true_eqns,
    )
    return jaxpr


class JaxprEvaluationForHiddenGroup:
    """
    Evaluating the jaxpr for extracting the hidden state ``hidden-to-hidden`` relationships.

    Args:
      jaxpr: The jaxpr for the model.
      hidden_outvars: The hidden output variables.
      hidden_outvar_to_invar: The mapping from the hidden output variable to the hidden input variable.
      hidden_invar_to_hidden: The mapping from the hidden input variable to the hidden state.
      hidden_outvar_to_hidden: The mapping from the hidden output variable to the hidden state.

    Returns:
      The list of the traced weight operations.
    """

    def __init__(
        self,
        jaxpr: jax.core.Jaxpr,
        hidden_outvars: Set[HiddenInVar],
        hidden_outvar_to_invar: Dict[HiddenOutVar, HiddenInVar],
        hidden_invar_to_hidden: Dict,
        hidden_outvar_to_hidden: Dict,
        weight_invars,
    ):
        # the jaxpr of the original model, assuming that the model is well-defined,
        # see the doc for the model which can be online learning compiled.
        self.jaxpr = jaxpr

        # the hidden state groups
        self.hidden_outvar_to_invar = hidden_outvar_to_invar
        self.hidden_invar_to_outvar = {invar: outvar for outvar, invar in hidden_outvar_to_invar.items()}
        self.hidden_invars = set(hidden_outvar_to_invar.values())
        self.hidden_outvars = hidden_outvars

        # the data structures for the tracing hidden-hidden relationships
        self.active_tracings = dict()

        # hidden invar/outvar to hidden itself
        self.hidden_invar_to_hidden = hidden_invar_to_hidden
        self.hidden_outvar_to_hidden = hidden_outvar_to_hidden

        self.weight_invars = weight_invars

    def compile(self) -> Tuple[Sequence, Dict, Dict]:
        """
        Compiling the jaxpr for the etrace relationships.
        """

        # the data structures for the tracing hidden-hidden relationships
        self.active_tracings: Dict[jax.core.Var, HiddenToHiddensTracer] = dict()

        # evaluating the jaxpr
        self._eval_jaxpr(self.jaxpr)

        # post checking
        final_traces = self._post_check()

        # reset the temporal data structures
        self.active_tracings = dict()
        return final_traces

    def _eval_jaxpr(
        self,
        jaxpr,
        invars_to_replace=None,
        outvars_to_replace=None
    ) -> None:
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
                if _check_some_element_exist_in_the_set(eqn.invars, self.hidden_invars):
                    raise NotImplementedError(
                        f'Currently, brainscale does not support the "scan" operator with hidden states. '
                        f'Please raise an issue or feature request to the developers at {git_issue_addr}.'
                    )
                self._eval_eqn(eqn)
            elif eqn.primitive.name == 'while':
                if _check_some_element_exist_in_the_set(eqn.invars, self.hidden_invars):
                    raise NotImplementedError(
                        f'Currently, brainscale does not support the "while" operator with hidden states. '
                        f'Please raise an issue or feature request to the developers at {git_issue_addr}.'
                    )
                self._eval_eqn(eqn)
            elif eqn.primitive.name == 'cond':
                if _check_some_element_exist_in_the_set(eqn.invars, self.hidden_invars):
                    raise NotImplementedError(
                        f'Currently, brainscale does not support the "cond" operator with hidden states. '
                        f'Please raise an issue or feature request to the developers at {git_issue_addr}.'
                    )
                self._eval_eqn(eqn)
            else:
                self._eval_eqn(eqn)

    def _eval_pjit(self, eqn: jax.core.JaxprEqn) -> None:
        """
        Evaluating the pjit primitive.
        """
        closed_jaxpr = eqn.params['jaxpr']
        if is_etrace_op(eqn.params['name']):
            if not is_etrace_op_enable_gradient(eqn.params['name']):
                return
        if (
            # checking whether the weight variables are used in the pjit
            _check_some_element_exist_in_the_set(eqn.invars, self.weight_invars)
            or
            # checking whether the hidden variables are computed in the pjit
            _check_some_element_exist_in_the_set(eqn.outvars, self.hidden_outvars)
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
        if eqn.primitive.name == 'stop_gradient':
            return

        # check whether the invars have one of the hidden states.
        # If it is true, add a new tracer.
        other_invars = []
        hidden_invars = []
        for invar in eqn.invars:
            if isinstance(invar, jax.core.Literal):
                continue
            elif invar in self.hidden_invars:
                hidden_invars.append(invar)
            else:
                other_invars.append(invar)
        if len(hidden_invars) > 0:
            # A hidden invar may be used in multiple places.
            # All places share a common tracer.
            assert len(hidden_invars) == 1
            hidden_var = hidden_invars[0]
            hidden_outvars = set([outvar for outvar in eqn.outvars if outvar in self.hidden_outvars])
            needed_invars = set([outvar for outvar in eqn.outvars if outvar not in self.hidden_outvars])
            if hidden_var in self.active_tracings:
                self.active_tracings[hidden_var].trace.append(eqn.replace())
                self.active_tracings[hidden_var].other_invars.update(other_invars)
                self.active_tracings[hidden_var].invar_needed_in_oth_eqns.update(needed_invars)
                self.active_tracings[hidden_var].connected_hidden_outvars.update(hidden_outvars)
            else:
                tracer = HiddenToHiddensTracer(
                    hidden_invar=hidden_var,
                    connected_hidden_outvars=hidden_outvars,
                    other_invars=set(other_invars),
                    invar_needed_in_oth_eqns=needed_invars,
                    trace=[eqn.replace()]
                )
                self.active_tracings[hidden_var] = tracer

        # check whether this equation is used in other tracers
        for tracer in tuple(self.active_tracings.values()):
            tracer: HiddenToHiddensTracer
            matched = _check_matched(eqn.invars, tracer.invar_needed_in_oth_eqns)
            # if matched, add the eqn to the trace
            # if not matched, skip
            if len(matched):
                self._add_eqn_in_a_trace(eqn, tracer)

    def _add_eqn_in_a_trace(
        self,
        eqn: jax.core.JaxprEqn,
        tracer: HiddenToHiddensTracer
    ) -> None:
        tracer.trace.append(eqn.replace())
        tracer.invar_needed_in_oth_eqns.update(eqn.outvars)
        # check whether the hidden states are needed in the other equations
        for outvar in eqn.outvars:
            if outvar in self.hidden_outvars:
                tracer.connected_hidden_outvars.add(outvar)

    def _post_check(self) -> Tuple[Sequence, Dict, Dict]:
        # [First step]
        # check the following items:
        #
        # 1. the shape of connected hidden states should be the same
        # 2. simplify the trace
        # 3. remove the unnecessary hidden states

        hidden_to_group = [
            _simplify_hid2hid_tracer(
                tracer,
                self.hidden_invar_to_hidden,
                self.hidden_outvar_to_hidden
            )
            for tracer in self.active_tracings.values()
        ]

        # [second step]
        # Find out the hidden group,
        # i.e., the hidden states that are connected to each other, the union of all hidden2group
        groups = [
            set([self.hidden_invar_to_outvar[transition.hidden_invar]] + list(transition.connected_hidden_outvars))
            for transition in hidden_to_group]
        group_sets = self._group_merging(groups)
        # transform the hidden group set to the HiddenGroup
        groups = []
        for group in group_sets:
            hidden_outvars = list(group)
            hidden_invars = [self.hidden_outvar_to_invar[outvar] for outvar in hidden_outvars]
            hidden_states = [self.hidden_outvar_to_hidden[outvar] for outvar in hidden_outvars]
            group = HiddenGroup(hidden_invars=hidden_invars,
                                hidden_outvars=hidden_outvars,
                                hidden_states=hidden_states)
            groups.append(group)
        # hidden_outvar to group
        hid2group = dict()
        for group in groups:
            for hid in group.hidden_outvars:
                hid2group[hid] = group

        # hidden_outvar to transition:
        #
        #   h_1^t, h_2^t, ... = f(h_i^t-1, ....)
        #
        hidden_outvar_to_transition = dict()
        for transition in hidden_to_group:
            transition: HiddenTransition
            hidden_outvar_at_t_minus_1 = self.hidden_invar_to_outvar[transition.hidden_invar]
            hidden_outvar_to_transition[hidden_outvar_at_t_minus_1] = transition

        return groups, hid2group, hidden_outvar_to_transition

    @staticmethod
    def _group_merging(groups) -> Sequence[Set[HiddenOutVar]]:
        """
        Merging the groups.
        """
        previous = frozenset([frozenset(g) for g in groups])
        while True:
            new_groups = []
            old_groups = list(previous)
            not_merged = list(range(len(old_groups)))
            while len(not_merged) > 0:
                i = not_merged.pop()
                merged = False
                for j in tuple(not_merged):
                    if len(old_groups[i].intersection(old_groups[j])) > 0:
                        new_groups.append(old_groups[i].union(old_groups[j]))
                        not_merged.remove(j)
                        merged = True
                if not merged:
                    new_groups.append(old_groups[i])
            new = frozenset([frozenset(g) for g in new_groups])
            if new == previous:
                break
            previous = new
        return list(new)


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
        id_to_state: Dict[StateID, ETraceState],
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
        self.new_invars = {
            v: self._new_var_like(v)
            for v in self.hidden_outvars
        }

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
                f'Error: we did not found your defined hidden state '
                f'(see the following information) in the code. \n'
                f'Please report an issue to the developers at {git_issue_addr}. \n'
                f'The missed hidden states are: \n'
                f'{state_traceback([self.id_to_state[self.outvar_to_state_id[v]] for v in self.hidden_jaxvars_to_remove])}'
            )

        # new jaxpr
        jaxpr = jax.core.Jaxpr(
            constvars=list(self.closed_jaxpr.jaxpr.constvars),
            invars=list(self.closed_jaxpr.jaxpr.invars) + list(self.new_invars.values()),
            outvars=list(self.closed_jaxpr.jaxpr.outvars),
            eqns=self.revised_eqns
        )
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
                self._eval_eqn(eqn)

            elif eqn.primitive.name == 'scan':
                if _check_some_element_exist_in_the_set(eqn.invars, self.hidden_invars):
                    raise NotImplementedError(
                        f'Currently, brainscale does not support the "scan" operator with hidden states. '
                        f'Please raise an issue or feature request to the developers at {git_issue_addr}.'
                    )
                else:
                    self.revised_eqns.append(eqn.replace())

            elif eqn.primitive.name == 'while':
                if _check_some_element_exist_in_the_set(eqn.invars, self.hidden_invars):
                    raise NotImplementedError(
                        f'Currently, brainscale does not support the "while" operator with hidden states. '
                        f'Please raise an issue or feature request to the developers at {git_issue_addr}.'
                    )
                else:
                    self.revised_eqns.append(eqn.replace())

            elif eqn.primitive.name == 'cond':
                if _check_some_element_exist_in_the_set(eqn.invars, self.hidden_invars):
                    raise NotImplementedError(
                        f'Currently, brainscale does not support the "cond" operator with hidden states. '
                        f'Please raise an issue or feature request to the developers at {git_issue_addr}.'
                    )
                else:
                    self.revised_eqns.append(eqn.replace())

            else:
                self._eval_eqn(eqn)

    def _add_perturb_eqn(self, eqn: jax.core.JaxprEqn, perturb_var: jax.core.Var):
        # ------------------------------------------------
        #
        # For the hidden var eqn, we want to add a perturbation:
        #    y = f(x)  =>  y = f(x) + perturb_var
        #
        # Particularly, we first define a new variable
        #    new_outvar = f(x)
        # Then, we add a new equation for the perturbation
        #    y = new_outvar + perturb_var
        #
        # ------------------------------------------------

        hidden_var = eqn.outvars[0]

        # Frist step, define the hidden var as a new variable
        new_outvar = self._new_var_like(perturb_var)
        old_eqn = eqn.replace(outvars=[new_outvar])
        self.revised_eqns.append(old_eqn)

        # Second step, add the perturbation equation
        new_eqn = jax.core.new_jaxpr_eqn([new_outvar, perturb_var],
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


def _summarize_frame(frame) -> str:
    if frame.start_column != 0:
        return (f"{frame.file_name}:{frame.start_line}:{frame.start_column} "
                f"({frame.function_name})")
    else:
        return f"{frame.file_name}:{frame.start_line} ({frame.function_name})"


def _summarize_source_info(
    source_info: source_info_util.SourceInfo,
    start_frame: int = 0,
    num_frames: int = 1
) -> str:
    from jax._src.source_info_util import user_frames
    frames = itertools.islice(
        user_frames(source_info),
        start_frame,
        start_frame + num_frames
    )
    frame_strs = [
        _summarize_frame(frame)
        if frame else "unknown"
        for frame in frames
    ]
    return '\n'.join(reversed(frame_strs))


class _VJPTime(BaseEnum):
    t = 't'
    t_minus_1 = 't_minus_1'


class HiddenTransition(NamedTuple):
    hidden_invar: jax.core.Var
    hidden: ETraceState
    connected_hidden_outvars: List[jax.core.Var]
    connected_hiddens: List[ETraceState]
    jaxpr: jax.core.Jaxpr
    other_invars: List[jax.core.Var]

    def state_transition(
        self,
        old_hidden_val: jax.Array,
        other_input_vals: PyTree,
        return_index: Optional[int] = None
    ) -> HiddenVals | jax.Array:
        """
        Computing the hidden state transitions :math:`h^t = f(h_i^t, x)`.

        Args:
          old_hidden_val: The old hidden state value.
          other_input_vals: The input values.
          return_index: index of the hidden state to return.

        Returns:
          The new hidden state values.
        """
        new_hidden_vals = jax.core.eval_jaxpr(self.jaxpr, other_input_vals, old_hidden_val)
        if return_index is not None:
            return new_hidden_vals[return_index]
        return new_hidden_vals


class HiddenGroup(NamedTuple):
    r"""
    The data structure for recording the hidden-to-hidden relation.

    The following fields are included:x

    - hidden_vars: the hidden states for one neuron population
    - input_vars: the input variables for the computing hidden state transitions
    - jaxpr: the jaxpr for the hidden state transitions

    This relation is used for computing the hidden-to-hidden state transitions::

        h_{t+1} = f(h_t, x_t)

    where ``h_t`` is the hidden state defined in ``hidden_vars``, ``x_t`` is the input at time ``t``
    defined in ``input_vars``, and ``f`` is the hidden state transition function which is defined
    in ``jaxpr``.

    """

    # "hidden_invars", "hidden_outvars", and "hidden_states"
    # sequentially correspond to the order in each other.
    hidden_invars: List[HiddenInVar]  # the input hidden states
    hidden_outvars: List[HiddenOutVar]  # the output hidden states
    hidden_states: List[ETraceState]  # the hidden states

    def hidden_invar_in_this_group(self, invar: jax.core.Var) -> bool:
        """
        Checking whether the input variable is in the hidden states of this group.

        Args:
          invar: The input variable.

        Returns:
          Whether the input variable is in the hidden states of this group.
        """
        return invar in self.hidden_invars

    def hidden_outvar_in_this_group(self, outvar: jax.core.Var) -> bool:
        """
        Checking whether the output variable is in the hidden states of this group.

        Args:
          outvar: The output variable.

        Returns:
          Whether the output variable is in the hidden states of this group.
        """
        return outvar in self.hidden_outvars

    def hidden_state_in_this_group(self, state: ETraceState) -> bool:
        """
        Checking whether the state is in the hidden states of this group.

        Args:
          state: The state.

        Returns:
          Whether the state is in the hidden states of this group.
        """
        return state in self.hidden_states


class HiddenWeightOpRelation(NamedTuple):
    """
    The data structure for recording the weight, operator, and hidden relationship.

    The following fields are included:

    - weight: the instance of ``ETraceParam``
    - op_jaxpr: the jaxpr for the weight operation, instance of ``jax.core.Jaxpr``
    - x: the jax Var for the weight input
    - y: the jax Var for the wight output
    - jaxpr_y2hid: the jaxpr to evaluate y -->  eligibility trace variables
    - hidden_vars: the hidden states connected to the weight
    """

    weight: ETraceParam
    op_jaxpr: jax.core.Jaxpr
    x: WeightXVar
    y: WeightYVar
    jaxpr_y2hid: jax.core.Jaxpr
    hidden_vars: List[HiddenOutVar]
    hidden_groups: List[HiddenGroup]
    hidden_var_to_transition: Dict[HiddenOutVar, HiddenTransition]


_compiler_docstr = '''
  diag_normalize: bool
      Whether to normalize the hidden Jacobian diagonal matrix to the range of ``[-1, 1]``. Default is ``None``.
  vjp_time: str
      The time to compute the loss-to-hidden Jacobian. It should be one of the
      following values:

      - 't': compute the loss-to-hidden Jacobian at the current time step: :math:`\partial L^t / \partial h^t`
      - 't_minus_1': compute the loss-to-hidden Jacobian at the last time step: :math:`\partial L^t / \partial h^{t-1}`
'''


class ETraceGraph:
    r"""
    The eligibility trace graph, tracking the relationship between the etrace weights
    :py:class:`ETraceParam`, the etrace variables :py:class:`ETraceState`, and the etrace
    operations :py:class:`ETraceOp`.

    This class is used for computing the weight spatial gradients and the hidden state residuals.
    It is the most foundational data structure for the ETrace algorithms.

    It is important to note that the graph is built no matter whether the model is
    batched or not. This means that this graph can be applied to any kind of models.
    However, the compilation is sensitive to the shape of hidden states.

    Parameters
    ----------
    {doc}
    """
    __module__ = 'brainscale'

    # [ Attributes for the graph ]
    out_all_jaxvars: List[jax.core.Var]  # all outvars except the function returns
    out_state_jaxvars: List[jax.core.Var]  # the state vars
    out_wx_jaxvars: List[jax.core.Var]  # the weight x
    hid2hid_jaxvars: List[jax.core.Var]  # the hidden to hidden vars
    num_out: int  # the number of function returns

    # hidden invar/outvar to hidden itself
    hidden_invar_to_hidden: Dict[jax.core.Var, ETraceState]
    hidden_outvar_to_hidden: Dict[jax.core.Var, ETraceState]

    # hidden outvar-to-invar, and invar-to-outvar
    hidden_outvar_to_invar: Dict[jax.core.Var, jax.core.Var]
    hidden_invar_to_outvar: Dict[jax.core.Var, jax.core.Var]

    # [ KEY ]
    #
    # 1. The most important data structure for the graph, which implementing
    #    the relationship between the etrace weights and the etrace variables.
    hidden_param_op_relations: Tuple[HiddenWeightOpRelation, ...]
    #
    # 2. The relationship between the hidden states.
    hidden_groups: Sequence[HiddenGroup]
    hidden_outvar_to_group: Dict[HiddenOutVar, HiddenGroup]
    hidden_outvar_to_transition: Dict[HiddenOutVar, HiddenTransition]

    def __init__(
        self,
        model: Callable,
        vjp_time: str | Enum = 't',
    ):
        # The original model
        self.model = model

        # the time for computing the VJP
        self.vjp_time = _VJPTime.get(vjp_time)

        # --- stateful model, for extracting states, weights, and variables --- #
        #
        # [ NOTE ]
        # The model does not support "static_argnums" for now.
        # Please always use ``functools.partial`` to fix the static arguments.
        #
        # wrap the model so that we can track the iteration number
        self.stateful_model = bst.compile.StatefulFunction(model)

        # --- rewrite jaxpr --
        #
        # The augmented jaxpr to return all necessary variables
        self.augmented_jaxpr: jax.core.ClosedJaxpr = None

        # The revised jaxpr with hidden state perturbations and return necessary variables
        # This jaxpr is only needed when the "vjp_time" is "t".
        self.jaxpr_with_hidden_perturb: jax.core.ClosedJaxpr = None

    @property
    def states(self):
        """
        Getting the states of the model (all instances of ``braincore.State``).
        """
        return self.stateful_model.get_states()

    def compile_graph(self, *args, **kwargs):
        """
        Building the eligibility trace graph for the model according to the given inputs.

        This is the most important method for the eligibility trace graph. It builds the
        graph for the model, which is used for computing the weight spatial gradients and
        the hidden state Jacobian.

        """

        # -- compile the model -- #
        #
        # NOTE:
        # The model does not support "static_argnums" for now.
        # Please always use functools.partial to fix the static arguments.
        #
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
        in_avals, _ = jax.tree.flatten((args, kwargs))
        out_avals, _ = jax.tree.flatten(out_shapes)
        num_in = len(in_avals)
        num_out = len(out_avals)
        state_avals, state_tree = jax.tree.flatten(state_vals)
        assert len(jaxpr.invars) == len(in_avals) + len(state_avals)
        assert len(jaxpr.outvars) == num_out + len(state_avals)
        self.num_out = num_out
        invars_with_state_tree = jax.tree.unflatten(state_tree, jaxpr.invars[num_in:])
        outvars_with_state_tree = jax.tree.unflatten(state_tree, jaxpr.outvars[num_out:])

        # remove the quantity from the invars and outvars
        invars_with_state_tree = _remove_quantity(invars_with_state_tree)
        outvars_with_state_tree = _remove_quantity(outvars_with_state_tree)

        # -- checking weights as invar -- #
        weight_id_to_invar = {
            id(st): jax.tree.leaves(invar)
            for invar, st in zip(invars_with_state_tree, states)
            if isinstance(st, ETraceParam)
        }
        hidden_id_to_invar = {
            id(st): invar  # ETraceState only contains one Array, "invar" is the jaxpr var
            for invar, st in zip(invars_with_state_tree, states)
            if isinstance(st, ETraceState)
        }
        invar_to_weight_id = {
            v: k
            for k, vs in weight_id_to_invar.items()
            for v in vs
        }

        # -- checking states as outvar -- #
        hidden_id_to_outvar = {
            id(st): outvar  # ETraceState only contains one Array, "outvar" is the jaxpr var
            for outvar, st in zip(outvars_with_state_tree, states)
            if isinstance(st, ETraceState)
        }
        outvar_to_state_id = {
            v: state_id
            for state_id, v in hidden_id_to_outvar.items()
        }
        self.hidden_outvar_to_invar = {
            outvar: hidden_id_to_invar[hid]
            for hid, outvar in hidden_id_to_outvar.items()
        }
        self.hidden_invar_to_outvar = {
            invar: outvar
            for outvar, invar in self.hidden_outvar_to_invar.items()
        }
        self.hidden_outvar_to_hidden = {
            outvar: st
            for outvar, st in zip(outvars_with_state_tree, states)
            if isinstance(st, ETraceState)
        }
        self.hidden_invar_to_hidden = {
            invar: st
            for invar, st in zip(invars_with_state_tree, states)
            if isinstance(st, ETraceState)
        }

        # -- evaluating the relationship for hidden-to-hidden -- #
        evaluator = JaxprEvaluationForHiddenGroup(
            jaxpr=jaxpr,
            hidden_outvars=set(hidden_id_to_outvar.values()),
            hidden_outvar_to_invar=self.hidden_outvar_to_invar,
            hidden_invar_to_hidden=self.hidden_invar_to_hidden,
            hidden_outvar_to_hidden=self.hidden_outvar_to_hidden,
            weight_invars=set([v for vs in weight_id_to_invar.values() for v in vs])
        )
        groups, hid2group, hidden_outvar_to_transition = evaluator.compile()
        self.hidden_groups = groups
        self.hidden_outvar_to_group = hid2group
        self.hidden_outvar_to_transition = hidden_outvar_to_transition

        # -- evaluating the jaxpr for (hidden, weight, op) relationships -- #
        evaluator = JaxprEvaluationForHiddenWeightOpRelation(
            jaxpr=jaxpr,
            weight_id_to_vars=weight_id_to_invar,
            invar_to_weight_id=invar_to_weight_id,
            id_to_state=id_to_state,
            hidden_invars=list(hidden_id_to_invar.values()),
            hidden_outvars=list(hidden_id_to_outvar.values()),
            hidden_outvar_to_group=self.hidden_outvar_to_group,
            hidden_outvar_to_transition=hidden_outvar_to_transition
        )
        self.hidden_param_op_relations = evaluator.compile()

        # --- Collect the Var needed to compute the weight spatial gradients --- #
        # ---      Rewrite the jaxpr for computing the needed variables      --- #

        # all states jaxpr var
        self.out_state_jaxvars = list(jaxpr.outvars[num_out:])
        weight_jaxvar_tree, hidden_jaxvar, other_state_jaxvar_tree = (
            split_state_values(states, outvars_with_state_tree)
        )

        # all jaxpr var of ETraceState, one etrace to one Array
        # All ETrace Hidden state
        self.out_hidden_jaxvars = list(hidden_jaxvar)

        # all weight x
        self.out_wx_jaxvars = list(
            set([relation.x for relation in self.hidden_param_op_relations])
        )

        # all y-to-hidden vars
        out_wy2hid_jaxvars = list(
            set(
                [v for relation in self.hidden_param_op_relations
                 for v in (relation.jaxpr_y2hid.invars + relation.jaxpr_y2hid.constvars)]
            )
        )

        # hidden-hidden transition vars
        hid2hid_jaxvars = set()
        for group in self.hidden_groups:
            hid2hid_jaxvars.update([v for v in group.hidden_invars])
        for transition in self.hidden_outvar_to_transition.values():
            hid2hid_jaxvars.update([v for v in transition.other_invars])

        # all outvars
        all_outvars = list(
            set(
                self.out_state_jaxvars +  # all state variables
                self.out_wx_jaxvars +  # all weight x
                out_wy2hid_jaxvars +  # all y-to-hidden invars
                list(hid2hid_jaxvars)  # all hidden-hidden transition vars
            )
        )
        self.out_all_jaxvars = jaxpr.outvars[:num_out] + all_outvars

        # Rewrite jaxpr to return all necessary variables, including
        #
        #   1. the original function outputs
        #   2. the hidden states
        #   3. the weight x   ===>  for computing the weight spatial gradients
        #   4. the y-to-hidden variables   ===>  for computing the weight spatial gradients
        #   5. the hidden-hidden transition variables   ===>  for computing the hidden-hidden jacobian
        #
        jaxpr = jax.core.Jaxpr(
            constvars=list(jaxpr.constvars),
            invars=list(jaxpr.invars),
            outvars=list(self.out_all_jaxvars),
            eqns=list(jaxpr.eqns)
        )
        self.augmented_jaxpr = jax.core.ClosedJaxpr(jaxpr, closed_jaxpr.consts)

        if self.vjp_time == _VJPTime.t:
            # ---               add perturbations to the hidden states                  --- #
            # --- new jaxpr with hidden state perturbations for computing the residuals --- #
            evaluator = JaxprEvaluationForHiddenPerturbation(
                closed_jaxpr=self.augmented_jaxpr,
                hidden_outvars=self.out_hidden_jaxvars,
                outvar_to_state_id=outvar_to_state_id,
                id_to_state=id_to_state,
                hidden_invars=list(hidden_id_to_invar.values()),
            )
            self.jaxpr_with_hidden_perturb = evaluator.compile()
        return self

    def show_graph(self, start_frame=1, n_frame=3):
        """
        Showing the graph about the relationship between weight, operator, and hidden states.
        """
        if self.augmented_jaxpr is None:
            raise ValueError(f'Please compile the graph first by calling ".{self.compile_graph.__name__}()" function.')

        for i, hpo_relation in enumerate(self.hidden_param_op_relations):
            msg = '===' * 40 + '\n'
            msg += f'For weight {i}: {hpo_relation.weight}\n\n'
            msg += '1. It is defined at: \n'
            source = indent_code(
                _summarize_source_info(hpo_relation.weight.source_info,
                                       start_frame=start_frame,
                                       num_frames=n_frame),
                indent=3
            )
            msg += f'{source}\n\n'
            msg += '2. The associated hidden states are:\n'
            for hid_var in hpo_relation.hidden_vars:
                hidden: ETraceState = self.hidden_outvar_to_hidden[hid_var]
                msg += f'   {hidden}, which is defined in\n'
                source = indent_code(
                    _summarize_source_info(hidden.source_info,
                                           start_frame=start_frame,
                                           num_frames=n_frame),
                    indent=6
                )
                msg += f'{source}\n'
            msg += '\n'
            msg += '3. The associated etrace operator [ y^t = x^t @ w ] is:\n\n'
            msg += indent_code(
                jaxpr_to_python_code(hpo_relation.op_jaxpr, fn_name='weight_to_hidden_operation'),
                indent=3
            )
            msg += '\n\n'
            msg += '4. The associated hidden states [ h^t = g(h^t-1) ] have the following relationships:\n\n'
            for group in hpo_relation.hidden_groups:
                msg += f'   The hidden states are: {group.hidden_states}:\n\n'
                for hidden_outvar in group.hidden_outvars:
                    transition = self.hidden_outvar_to_transition[hidden_outvar]
                    msg += f'   {hidden_outvar} ==> {transition.connected_hidden_outvars}:\n\n'
                    msg += indent_code(
                        jaxpr_to_python_code(transition.jaxpr, fn_name='hidden_to_hidden_transition'),
                        indent=6
                    )
                    msg += '\n\n'
            msg += '\n\n'
            msg += '---' * 40 + '\n\n'
            print(msg)

    def _jaxpr_compute_model(
        self, *args, **kwargs,
    ) -> Tuple[PyTree, HiddenVals, StateVals, TempData]:
        """
        Computing the model according to the given inputs and parameters by using the compiled jaxpr.
        """
        # state checking
        old_state_vals = [st.value for st in self.stateful_model.get_states()]

        # parameters
        args = jax.tree.flatten((args, kwargs, old_state_vals))[0]

        # calling the function
        jaxpr_outs = jax.core.eval_jaxpr(self.augmented_jaxpr.jaxpr, self.augmented_jaxpr.consts, *args)

        # intermediate values
        temps = {v: r for v, r in zip(self.out_all_jaxvars[self.num_out:], jaxpr_outs[self.num_out:])}

        # recovery outputs of ``stateful_model``
        state_outs = [temps[v] for v in self.out_state_jaxvars]
        out, new_state_vals = self.stateful_model.get_out_treedef().unflatten(jaxpr_outs[:self.num_out] + state_outs)

        # state value assignment
        assert len(old_state_vals) == len(new_state_vals), 'State length mismatch.'
        # TODO: [KEY] assuming that the weight values are not changed
        hidden_vals, oth_state_vals = split_state_values(
            self.stateful_model.get_states(),
            new_state_vals,
            include_weight=False
        )
        return out, hidden_vals, oth_state_vals, temps

    def _compute_hid2weight_jacobian(
        self,
        intermediate_values: dict
    ) -> Tuple[Dict[WeightXVar, jax.Array], Dict[Tuple[WeightYVar, HiddenOutVar], jax.Array]]:
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
        for relation in self.hidden_param_op_relations:
            consts = [intermediate_values[var] for var in relation.jaxpr_y2hid.constvars]
            invars = [intermediate_values[var] for var in relation.jaxpr_y2hid.invars]  # weight y
            assert len(invars) == 1, 'The weight y should be unique.'

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
            primals, tangents = jax.jvp(
                lambda x: jax.core.eval_jaxpr(relation.jaxpr_y2hid, consts, x),
                invars,
                [u.math.ones(invars[0].shape, invars[0].dtype)]
            )

            # get the df we want
            for i, hidden_var in enumerate(relation.jaxpr_y2hid.outvars):  # hidden states
                dfs[(relation.y, hidden_var)] = tangents[i]

        # all x and df values
        return xs, dfs

    def _compute_hidden2hidden_jacobian(
        self, intermediate_values: dict
    ) -> Dict[Tuple[HiddenOutVar, HiddenOutVar], jax.Array]:
        hid2hid_jacobian = dict()

        for transition in self.hidden_outvar_to_transition.values():
            transition: HiddenTransition

            #
            # "primals" is the hidden state values at the previous time step
            primals = intermediate_values[transition.hidden_invar]

            #
            # "tangents" is the hidden-to-hidden Jacobian at the previous time step
            tangents = u.math.ones(primals.aval.shape, primals.aval.dtype)

            # JVP gradients, computing:
            #
            # [∂a^t/∂a^t-1, ∂b^t/∂a^t-1, ∂c^t/∂a^t-1, ...]
            #
            other_input_vals = [intermediate_values[v] for v in transition.other_invars]
            fun = partial(transition.state_transition, other_input_vals=other_input_vals)
            _, jvp_grads = jax.jvp(fun, (primals,), (tangents,))  # produce the new hidden, and the JVP gradients

            # store the gradients
            outvar_t_minus_1 = self.hidden_invar_to_outvar[transition.hidden_invar]
            for outvar_t, grad_data in zip(transition.connected_hidden_outvars, jvp_grads):
                key = (outvar_t_minus_1, outvar_t)
                assert key not in hid2hid_jacobian, 'The key should not exist.'
                hid2hid_jacobian[key] = grad_data

        return hid2hid_jacobian

    def solve_h2w_h2h_jacobian(
        self, *args,
    ) -> Tuple[Outputs, HiddenVals, StateVals, Hid2WeightJacobian, Hid2HidJacobian]:
        r"""
        Solving the hidden-to-weight and hidden-to-hidden Jacobian according to the given inputs and parameters.

        This function is typically used for computing the forward propagation of hidden-to-weight Jacobian.

        Particularly, this function aims to solve:

        1. The Jacobian matrix of hidden-to-weight. That is,
           :math:`\partial h / \partial w`, where :math:`h` is the hidden state and :math:`w` is the weight.
        2. The Jacobian matrix of hidden-to-hidden. That is,
           :math:`\partial h / \partial h`, where :math:`h` is the hidden state.

        Args:
          *args: The positional arguments for the model.

        Returns:
          The outputs, hidden states, other states, and the spatial gradients of the weights.
        """
        # --- compile the model --- #
        if self.augmented_jaxpr is None:
            raise ValueError('The ETraceGraph object has not been built yet.')

        # --- call the model --- #
        out, hiddens, others, temps = self._jaxpr_compute_model(*args)
        hid2weight_jac = self._compute_hid2weight_jacobian(temps)

        # --- other returns --- #
        hid2hid_jac = self._compute_hidden2hidden_jacobian(temps)

        return out, hiddens, others, hid2weight_jac, hid2hid_jac

    def _jaxpr_compute_vjp_model_at_current(
        self, *args
    ) -> Tuple[PyTree, HiddenVals, StateVals, TempData, Residuals]:
        """
        Computing the VJP transformed model according to the given inputs and parameters by using the compiled jaxpr.
        """
        _, hidden_states, non_etrace_weight_states, other_states = split_states_v2(self.stateful_model.get_states())

        def fun_for_vjp(inputs, hiddens, non_etrace_weights, oth_states, perturbs):
            # assign state values
            assign_state_values(hidden_states, hiddens)
            assign_state_values(non_etrace_weight_states, non_etrace_weights)
            assign_state_values(other_states, oth_states)
            # get state values by the "stateful_model", to preserve the order of states
            old_state_vals = [st.value for st in self.stateful_model.get_states()]

            # calling the function
            jaxpr_outs = jax.core.eval_jaxpr(
                self.jaxpr_with_hidden_perturb.jaxpr,
                self.jaxpr_with_hidden_perturb.consts,
                *jax.tree.leaves((inputs, old_state_vals, perturbs))
            )

            # intermediate values
            temps = {
                v: r
                for v, r in zip(self.out_all_jaxvars[self.num_out:], jaxpr_outs[self.num_out:])
            }

            # outputs
            state_outs = [temps[v] for v in self.out_state_jaxvars]
            out, new_state_vals = self.stateful_model.get_out_treedef().unflatten(
                jaxpr_outs[:self.num_out] + state_outs)
            new_hiddens, new_others = split_state_values(
                self.stateful_model.get_states(),
                new_state_vals,
                include_weight=False
            )
            return (out, new_hiddens, new_others), temps

        #  [KEY]
        #  The most important assumption here is
        #  that the weight values (including etrace weights and normal param weights) are not changed

        hidden_perturbs = [u.math.zeros(v.aval.shape, v.aval.dtype) for v in self.out_hidden_jaxvars]
        hidden_vals = [st.value for st in hidden_states]
        non_etrace_weight_vals = [st.value for st in non_etrace_weight_states]
        other_vals = [st.value for st in other_states]

        # VJP calling
        (out, hidden_vals, other_vals), f_vjp, temps = jax.vjp(
            fun_for_vjp,  # the function
            args,  # the inputs
            hidden_vals,
            non_etrace_weight_vals,
            other_vals,
            hidden_perturbs,
            has_aux=True
        )
        out_flat, out_tree = jax.tree.flatten(((out, hidden_vals, other_vals),))
        rule, in_tree = jax.api_util.flatten_fun_nokwargs(lu.wrap_init(f_vjp), out_tree)
        out_avals = [jax.core.get_aval(x).at_least_vspace() for x in out_flat]
        jaxpr, _, consts = pe.trace_to_jaxpr_dynamic(rule, out_avals)

        # recovering the other non-etrace weights, although the weights are not changed
        assign_state_values(non_etrace_weight_states, non_etrace_weight_vals)
        return out, hidden_vals, other_vals, temps, Residuals(jaxpr, in_tree(), out_tree, consts)

    def _jaxpr_compute_vjp_model_at_last(
        self, *args
    ) -> Tuple[PyTree, HiddenVals, StateVals, TempData, Residuals]:
        """
        Computing the VJP transformed model according to the given inputs and parameters by using the compiled jaxpr.
        """
        etrace_param_states, hidden_states, non_etrace_weight_states, other_states = (
            split_states_v2(self.stateful_model.get_states())
        )

        def fun_for_vjp(inputs, hiddens, non_etrace_weights, etrace_weights, oth_states):
            # assign state values
            assign_state_values(hidden_states, hiddens)
            assign_state_values(etrace_param_states, etrace_weights)
            assign_state_values(non_etrace_weight_states, non_etrace_weights)
            assign_state_values(other_states, oth_states)
            # get state values by the "stateful_model", to preserve the order of states
            old_state_vals = [st.value for st in self.stateful_model.get_states()]

            # calling the function
            jaxpr_outs = jax.core.eval_jaxpr(
                self.augmented_jaxpr.jaxpr,
                self.augmented_jaxpr.consts,
                *jax.tree.leaves((inputs, old_state_vals))
            )

            # intermediate values
            temps = {
                v: r
                for v, r in zip(self.out_all_jaxvars[self.num_out:], jaxpr_outs[self.num_out:])
            }

            # outputs
            state_outs = [temps[v] for v in self.out_state_jaxvars]
            out, new_state_vals = self.stateful_model.get_out_treedef().unflatten(
                jaxpr_outs[:self.num_out] + state_outs)
            # get new state values, do not return the weight values, since they are not changed
            new_hiddens, new_others = split_state_values(
                self.stateful_model.get_states(),
                new_state_vals,
                include_weight=False
            )
            return (out, new_hiddens, new_others), temps

        #  [KEY]
        #  The most important assumption here is
        #  that the weight values (including etrace weights and normal param weights) are not changed

        etrace_weight_vals = [st.value for st in etrace_param_states]
        non_etrace_weight_vals = [st.value for st in non_etrace_weight_states]
        hidden_vals = [st.value for st in hidden_states]
        other_vals = [st.value for st in other_states]

        # VJP calling
        (out, hidden_vals, other_vals), f_vjp, temps = jax.vjp(
            fun_for_vjp,  # the function
            args, hidden_vals, non_etrace_weight_vals, etrace_weight_vals, other_vals,  # the inputs
            has_aux=True
        )
        out_flat, out_tree = jax.tree.flatten(((out, hidden_vals, other_vals),))
        rule, in_tree = jax.api_util.flatten_fun_nokwargs(lu.wrap_init(f_vjp), out_tree)
        out_avals = [jax.core.get_aval(x).at_least_vspace() for x in out_flat]
        jaxpr, _, consts = pe.trace_to_jaxpr_dynamic(rule, out_avals)

        # recovering the other non-etrace weights,
        # although the weights are not changed
        assign_state_values(non_etrace_weight_states, non_etrace_weight_vals)
        assign_state_values(etrace_param_states, etrace_weight_vals)
        return out, hidden_vals, other_vals, temps, Residuals(jaxpr, in_tree(), out_tree, consts)

    def solve_h2w_h2h_jacobian_and_l2h_vjp(
        self, *args,
    ) -> Tuple[Outputs, HiddenVals, StateVals, Hid2WeightJacobian, Hid2HidJacobian, Residuals]:
        r"""
        Solving the hidden-to-weight and hidden-to-hidden Jacobian and the VJP transformed loss-to-hidden
        gradients according to the given inputs.

        This function is typically used for computing both the forward propagation of hidden-to-weight Jacobian
        and the loss-to-hidden gradients at the current time-step.

        Particularly, this function aims to solve:

        1. The Jacobian matrix of hidden-to-weight. That is,
           :math:`\partial h / \partial w`, where :math:`h` is the hidden state and :math:`w` is the weight.
        2. The Jacobian matrix of hidden-to-hidden. That is,
           :math:`\partial h / \partial h`, where :math:`h` is the hidden state.
        3. The partial gradients of the loss with respect to the hidden states.
           That is, :math:`\partial L / \partial h`, where :math:`L` is the loss and :math:`h` is the hidden state.

        Args:
          *args: The positional arguments for the model.

        Returns:
          The outputs, hidden states, other states, the spatial gradients of the weights, and the residuals.
        """
        if not hasattr(self, 'jaxpr_with_hidden_perturb'):
            raise ValueError('The ETraceGraph object has not been built yet.')

        # --- call the model --- #
        if self.vjp_time == _VJPTime.t:
            out, hidden_vals, other_vals, temps, vjp_residual = self._jaxpr_compute_vjp_model_at_current(*args)
        elif self.vjp_time == _VJPTime.t_minus_1:
            out, hidden_vals, other_vals, temps, vjp_residual = self._jaxpr_compute_vjp_model_at_last(*args)
        else:
            raise ValueError('The VJP time should be either "current" or "last".')
        hid2weight_jac = self._compute_hid2weight_jacobian(temps)

        # --- other returns --- #
        hid2hid_jac = self._compute_hidden2hidden_jacobian(temps)
        return out, hidden_vals, other_vals, hid2weight_jac, hid2hid_jac, vjp_residual


ETraceGraph.__doc__ = ETraceGraph.__doc__.format(doc=_compiler_docstr)


@set_module_as('brainscale')
def build_etrace_graph(
    model: Callable,
    vjp_time: str | Enum = 't',
) -> Callable[..., ETraceGraph]:
    r"""
    Build the eligibility trace graph of the given model.

    The eligibility trace graph is used to compute the model gradients, including

    - the spatial gradients of the weights
    - the VJP gradients of the etrace variables

    Example:

      ```python
      import jax
      import brainscale

      # the model
      def model(x, w):
        return jax.nn.relu(jnp.dot(x, w))

      # define and compile the etrace graph
      etrace_graph = brainscale.build_etrace_graph(model, vjp_time='t')(x, w)
      ```

    Parameters
    ----------
    {doc}

    Returns
    -------
      graph: The eligibility trace graph.
    """
    etrace_graph = ETraceGraph(model, vjp_time=vjp_time)

    def _compile_graph(*args, **kwargs) -> ETraceGraph:
        etrace_graph.compile_graph(*args, **kwargs)
        return etrace_graph

    return _compile_graph


build_etrace_graph.__doc__ = build_etrace_graph.__doc__.format(doc=_compiler_docstr)
