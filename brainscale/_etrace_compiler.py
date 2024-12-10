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
# ==============================================================================
#
# Author: Chaoming Wang <chao.brain@qq.com>
# Copyright: 2024, Chaoming Wang
# Date: 2024-04-03
#
# Refinement History:
#   [2024-04-03] Created
#   [2024-04-06] Added the traceback information for the error messages.
#   [2024-04-16] Changed the "op" in the "HiddenWeightOpTracer" to "JaxprEqn".
#                Added the support for the "pjit" operator.
#   [2024-05] Add the support for vjp_time == 't_minus_1'
#   [2024-06] Conditionally support control flows, including `scan`, `while`, and `cond`
#   [2024-09] version 0.0.2
#   [2024-11-22] compatible with `brainstate>=0.1.0` (#17)
#   [2024-11-23] Add the support for vjp_time_ahead > 1, it can combine the
#                advantage of etrace learning and backpropagation through time.
#   [2024-11-26] version 0.0.3, a complete new revision for better model debugging.
#   [2024-12-05] change the ETraceWeight to NonETraceWeight if the hidden states are not found;
#                remove the connected hidden states when y=x@w is not shape broadcastable with the hidden states.
#   [2024-12-09] small updates, related to the key items in "CompiledGraph"
#
# ==============================================================================

# -*- coding: utf-8 -*-

from __future__ import annotations

import itertools
import warnings
from typing import (NamedTuple, List, Dict, Sequence, Tuple, Set, Optional)

import brainstate as bst
import brainunit as u
import jax.core
from jax.extend import source_info_util

from ._etrace_concepts import (is_etrace_op,
                               is_etrace_op_enable_gradient,
                               ETraceParam,
                               ETraceState)
from ._etrace_concepts import sequence_split_state_values
from ._jaxpr_to_source_code import jaxpr_to_python_code
from ._misc import (git_issue_addr,
                    NotSupportedError,
                    CompilationError)
from ._typing import (PyTree,
                      StateID,
                      WeightXVar,
                      WeightYVar,
                      HiddenInVar,
                      HiddenOutVar,
                      Path)

__all__ = [
    'compile_graph',
    'HiddenGroupV2',
    'HiddenTransition',
    'WeightOpHiddenRelation',
    'CompiledGraph',
]


# TODO
#
# - [x] The visualization of the etrace graph.
# - [ ] Evaluate whether the `df` is the same for different weights.
#       For example,
#
#          h = f(x1 @ w1 + x2 @ w2)
#
#       The `df` for w1 and w2 are the same, although them have the different weight y.


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


def _check_pjit(eqn, self):
    # checking whether the weight variables are used in the pjit
    invar = _check_some_element_exist_in_the_set(eqn.invars, self.weight_invars)
    if invar is not None:
        raise NotImplementedError(
            'Currently, we do not support the weight states are used within a jit function. \n'
            'Please remove your jit compilation on the intermediate steps. \n\n'
            f'The weight state is: {self.invar_to_hidden_path[invar]}. \n'
            f'The Jaxpr of the JIT function is: \n\n'
            f'{eqn} \n\n'
        )

    # checking whether the hidden variables are computed in the pjit
    outvar = _check_some_element_exist_in_the_set(eqn.outvars, self.hidden_outvars)
    if outvar is not None:
        raise NotImplementedError(
            'Currently, we do not support the hidden states are computed within a jit function. \n'
            'Please remove your jit compilation on the intermediate steps. \n\n'
            f'The hidden state is: {self.outvar_to_hidden_path[outvar]}. \n'
            f'The Jaxpr of the JIT function is: \n\n'
            f'{eqn} \n\n'
        )


class JaxprEvaluation(object):

    def __init__(
        self,
        weight_invars: Set[jax.core.Var],
        hidden_invars: Set[jax.core.Var],
        hidden_outvars: Set[jax.core.Var],
        invar_to_hidden_path: Dict[jax.core.Var, Path],
        outvar_to_hidden_path: Dict[jax.core.Var, Path],
    ):
        self.weight_invars = weight_invars
        self.hidden_invars = hidden_invars
        self.hidden_outvars = hidden_outvars
        self.invar_to_hidden_path = invar_to_hidden_path
        self.outvar_to_hidden_path = outvar_to_hidden_path

    def _eval_jaxpr(self, jaxpr) -> None:
        """
        Evaluating the jaxpr for extracting the etrace relationships.

        Args:
          jaxpr: The jaxpr for the model.
        """

        for eqn in jaxpr.eqns:
            # TODO: add the support for the scan, while, cond, pjit, and other operators
            # Currently, scan, while, and cond are usually not the common operators used in
            # the definition of a brain dynamics model. So we may not need to consider them
            # during the current phase.
            # However, for the long-term maintenance and development, we need to consider them,
            # since users usually create crazy models.

            if eqn.primitive.name == 'pjit':
                self._eval_pjit(eqn)
            elif eqn.primitive.name == 'scan':
                self._eval_scan(eqn)
            elif eqn.primitive.name == 'while':
                self._eval_while(eqn)
            elif eqn.primitive.name == 'cond':
                self._eval_cond(eqn)
            else:
                self._eval_eqn(eqn)

    def _eval_pjit(self, eqn: jax.core.JaxprEqn) -> None:
        """
        Evaluating the pjit primitive.
        """
        if is_etrace_op(eqn.params['name']):
            if is_etrace_op_enable_gradient(eqn.params['name']):
                self._eval_eqn(eqn)
            return

        _check_pjit(eqn, self)

        # treat the pjit as a normal jaxpr equation
        self._eval_eqn(eqn)

    def _eval_scan(self, eqn: jax.core.JaxprEqn) -> None:
        """
        Evaluating the scan primitive.
        """
        invar = _check_some_element_exist_in_the_set(eqn.invars, self.hidden_invars)
        if invar is not None:
            raise NotImplementedError(
                f'Currently, brainscale does not support the "scan" operator with hidden states. '
                f'Please raise an issue or feature request to the developers at {git_issue_addr}. \n'
                f'The hidden state is: {self.invar_to_hidden_path[invar]}. \n'
            )
        outvar = _check_some_element_exist_in_the_set(eqn.outvars, self.hidden_outvars)
        if outvar is not None:
            raise NotImplementedError(
                f'Currently, brainscale does not support the "scan" operator with hidden states. '
                f'Please raise an issue or feature request to the developers at {git_issue_addr}. \n'
                f'The hidden state is: {self.outvar_to_hidden_path[outvar]}. \n'
            )
        self._eval_eqn(eqn)

    def _eval_while(self, eqn: jax.core.JaxprEqn) -> None:
        """
        Evaluating the while primitive.
        """
        invar = _check_some_element_exist_in_the_set(eqn.invars, self.hidden_invars)
        if invar is not None:
            raise NotImplementedError(
                f'Currently, brainscale does not support the "while" operator with hidden states. '
                f'Please raise an issue or feature request to the developers at {git_issue_addr}. \n'
                f'The hidden state is: {self.invar_to_hidden_path[invar]}. \n'
            )
        outvar = _check_some_element_exist_in_the_set(eqn.outvars, self.hidden_outvars)
        if outvar is not None:
            raise NotImplementedError(
                f'Currently, brainscale does not support the "while" operator with hidden states. '
                f'Please raise an issue or feature request to the developers at {git_issue_addr}. \n'
                f'The hidden state is: {self.outvar_to_hidden_path[outvar]}. \n'
            )
        self._eval_eqn(eqn)

    def _eval_cond(self, eqn: jax.core.JaxprEqn) -> None:
        """
        Evaluating the cond primitive.
        """
        invar = _check_some_element_exist_in_the_set(eqn.invars, self.hidden_invars)
        if invar is not None:
            raise NotImplementedError(
                f'Currently, brainscale does not support the "cond" operator with hidden states. '
                f'Please raise an issue or feature request to the developers at {git_issue_addr}. \n'
                f'The hidden state is: {self.invar_to_hidden_path[invar]}. \n'
            )
        outvar = _check_some_element_exist_in_the_set(eqn.outvars, self.hidden_outvars)
        if outvar is not None:
            raise NotImplementedError(
                f'Currently, brainscale does not support the "cond" operator with hidden states. '
                f'Please raise an issue or feature request to the developers at {git_issue_addr}. \n'
                f'The hidden state is: {self.outvar_to_hidden_path[outvar]}. \n'
            )
        self._eval_eqn(eqn)

    def _eval_eqn(self, eqn):
        raise NotImplementedError(
            'The method "_eval_eqn" should be implemented in the subclass.'
        )


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
    weight_path: Path  # w
    x: jax.core.Var  # y
    y: jax.core.Var  # x
    trace: List[jax.core.JaxprEqn]
    hidden_vars: set[jax.core.Var]
    invar_needed_in_oth_eqns: set[jax.core.Var]

    def replace(
        self,
        weight=None,
        weight_path=None,
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
            weight_path=(weight_path if weight_path is not None else self.weight_path),
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
) -> jax.core.Var | None:
    """
    Checking whether the jaxpr vars contain the weight variables.

    Args:
      elements: The input variables of the equation.
      the_set: The set of the weight variables.
    """
    for invar in elements:
        if isinstance(invar, jax.core.Var) and invar in the_set:
            return invar
    return None


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


def _post_check(trace: HiddenWeightOpTracer) -> HiddenWeightOpTracer:
    # Check the hidden states of the given weight. If the hidden states are not
    # used in the model, we raise an error. This is to avoid the situation that
    # the weight is defined but not used in the model.
    if len(trace.hidden_vars) == 0:
        source_info = trace.weight.source_info
        name_stack = source_info_util.current_name_stack() + source_info.name_stack
        with source_info_util.user_context(source_info.traceback, name_stack=name_stack):
            trace.weight.is_not_etrace = True
            msg = (
                f'\n'
                f'Warning: The ETraceParam {trace.weight_path} does not found the associated hidden states. \n'
                f'We have changed is as a weight that is not trained with eligibility trace. However, if you \n'
                f'found this is a compilation error. Please report an issue to the developers at '
                f'{git_issue_addr}. \n\n'
            )
            warnings.warn(msg, UserWarning)
            # trace.weight._raise_error_with_source_info(
            #     CompilationError(
            #
            #     )
            # )
    return trace


def _simplify_hid2hid_tracer(
    tracer: HiddenToHiddensTracer,
    hidden_invar_to_path: Dict[HiddenInVar, Path],
    hidden_outvar_to_path: Dict[HiddenOutVar, Path],
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
    # Change the "HiddenWeightOpTracer" to "HiddenTransition"
    return HiddenTransition(
        hidden_invar=tracer.hidden_invar,
        hidden_path=hidden_invar_to_path[tracer.hidden_invar],
        connected_hidden_outvars=list(hidden_outvars),
        connected_hidden_paths=[hidden_outvar_to_path[var] for var in hidden_outvars],
        transition_jaxpr=jaxpr_opt,
        other_invars=constvars,
    )


def _trace_simplify(
    tracer: HiddenWeightOpTracer,
    hid_path_to_group: Dict[Path, 'HiddenGroupV2'],
    hid_path_to_transition: Dict[Path, 'HiddenTransition'],
    state_id_to_path: Dict[int, Path],
    outvar_to_hidden_path: Dict[jax.core.Var, Path],
) -> WeightOpHiddenRelation | None:
    """
    Simplifying the trace from the weight output to the hidden state.

    Args:
      tracer: The traced weight operation.

    Returns:
      The simplified traced weight operation.
    """

    # [First step]
    # Finding out how the shape of each hidden state is converted to the size of df.
    y = tracer.y
    connected_hidden_vars = []
    connected_hidden_paths = []
    for hidden_var in list(tracer.hidden_vars):
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

        try:
            jax.numpy.broadcast_shapes(y.aval.shape, hidden_var.aval.shape)
            connected_hidden_paths.append(outvar_to_hidden_path[hidden_var])
            connected_hidden_vars.append(hidden_var)
        except ValueError:
            msg = (
                f'\n'
                f'In your computational graph, we found ETraceWeight {tracer.weight_path} is connected '
                f'with hidden state {outvar_to_hidden_path[hidden_var]}.\n'
                f'However, our online learning is only applied to the case of: h^t = f(y), y = x @ w, '
                f'where y has the broadcastable shape with h^t. \n'
                f'while we found the shape of y is {y.aval.shape} and the shape of h^t is {hidden_var.aval.shape}.\n'
                f'Therefore, we remove the connection between the weight {tracer.weight_path} and the hidden state '
                f'{outvar_to_hidden_path[hidden_var]}.\n'
                f'If you found this is a bug, please report an issue to the developers at {git_issue_addr}. \n'
            )
            warnings.warn(msg, UserWarning)

            # raise NotSupportedError(
            #     f'Currently, the automatic shape inverse transformation is not supported. \n '
            #     f'Got {y.aval.shape} != {hidden_var.aval.shape}. \n'
            #     f'The weight is: {state_id_to_path[id(tracer.weight)]}. \n'
            #     f'The hidden state is: {outvar_to_hidden_path[hidden_var]}. \n'
            # )

    # [Second step]
    tracer = tracer.replace(hidden_vars=connected_hidden_vars)
    _post_check(tracer)
    if len(connected_hidden_paths) == 0:
        return None

    # [Third step]
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

    # [fourth step]
    equations = list(reversed(new_trace))

    # [fifth step]
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
        eqns=equations,
    )

    # [final step]
    # Change the "HiddenWeightOpTracer" to "WeightOpHiddenRelation"
    hidden_group_ids = set()
    hidden_group_mappings = dict()
    for path in connected_hidden_paths:
        group = hid_path_to_group[path]
        group_id = id(group)
        hidden_group_ids.add(group_id)
        hidden_group_mappings[group_id] = group

    hidden_path_to_transition = {
        path: hid_path_to_transition[path]
        for path in connected_hidden_paths
    }

    return WeightOpHiddenRelation(
        weight=tracer.weight,
        path=state_id_to_path[id(tracer.weight)],
        op_jaxpr=_jax_eqn_to_jaxpr(tracer.op),
        x=tracer.x,
        y=tracer.y,
        jaxpr_y2hid=jaxpr_opt,
        hidden_paths=connected_hidden_paths,
        hidden_groups=[hidden_group_mappings[gid] for gid in hidden_group_ids],
        hidden_path_to_transition=hidden_path_to_transition
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


class JaxprEvaluationForWeightOpHiddenRelation(JaxprEvaluation):
    """
    Evaluating the jaxpr for extracting the etrace (hidden, operator, weight) relationships.

    Args:
        jaxpr: The jaxpr for the model.
        weight_path_to_vars: The mapping from the weight id to the jax vars.
        invar_to_weight_path: The mapping from the jax var to the weight id.
        path_to_state: The mapping from the state id to the state.

    Returns:
        The list of the traced weight operations.
    """

    def __init__(
        self,
        jaxpr: jax.core.Jaxpr,
        hidden_outvar_to_invar: Dict[HiddenOutVar, HiddenInVar],
        weight_path_to_vars: Dict[Path, List[jax.core.Var]],
        invar_to_weight_path: Dict[jax.core.Var, Path],
        path_to_state: Dict[Path, bst.State],
        hid_path_to_group: Dict[Path, 'HiddenGroupV2'],
        hid_path_to_transition: Dict[Path, 'HiddenTransition'],
        invar_to_hidden_path: Dict[HiddenInVar, Path],
        outvar_to_hidden_path: Dict[HiddenOutVar, Path],
    ):
        # the jaxpr of the original model, assuming that the model is well-defined,
        # see the doc for the model which can be online learning compiled.
        self.jaxpr = jaxpr

        #  the mapping from the weight id to the jax vars, one weight id may contain multiple jax vars
        self.weight_path_to_vars = weight_path_to_vars

        # the mapping from the jax var to the weight id, one jax var for one weight id
        self.invar_to_weight_path = invar_to_weight_path

        # the mapping from the state id to the state
        self.path_to_state = path_to_state
        self.state_id_to_path = {id(state): path for path, state in path_to_state.items()}

        # jax vars of weights
        weight_invars = set([v for vs in weight_path_to_vars.values() for v in vs])

        self.hid_path_to_group = hid_path_to_group
        self.hid_path_to_transition = hid_path_to_transition

        super().__init__(
            weight_invars=weight_invars,
            hidden_invars=set(hidden_outvar_to_invar.values()),
            hidden_outvars=set(hidden_outvar_to_invar.keys()),
            invar_to_hidden_path=invar_to_hidden_path,
            outvar_to_hidden_path=outvar_to_hidden_path
        )

    def compile(self) -> Tuple[WeightOpHiddenRelation, ...]:
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
            _trace_simplify(
                _post_check(trace),
                hid_path_to_group=self.hid_path_to_group,
                hid_path_to_transition=self.hid_path_to_transition,
                state_id_to_path=self.state_id_to_path,
                outvar_to_hidden_path=self.outvar_to_hidden_path,
            )
            for trace in self.active_tracings
        ]

        # reset the temporal data structures
        self.active_tracings = []
        return tuple([trace for trace in final_traces if trace is not None])

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
            weight_path, x = self._get_state_and_inp_and_checking(eqn)

            # add new trace
            self.active_tracings.append(
                HiddenWeightOpTracer(
                    weight=self.path_to_state[weight_path],
                    weight_path=weight_path,
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
            _check_pjit(eqn, self)

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
                    if is_etrace_op_enable_gradient(eqn.params['name']):
                        # weight -> diagnal weight -> ?
                        self._add_eqn_in_a_trace(eqn, trace)
                    else:
                        # avoid off " weight -> weight -> ? "  pathway
                        pass

    def _add_eqn_in_a_trace(
        self,
        eqn: jax.core.JaxprEqn,
        trace: HiddenWeightOpTracer
    ) -> None:
        trace.trace.append(eqn.replace())
        trace.invar_needed_in_oth_eqns.update(eqn.outvars)
        # check whether the hidden states are needed in the other equations
        for outvar in eqn.outvars:
            if outvar in self.hidden_outvars:
                trace.hidden_vars.add(outvar)

    def _get_state_and_inp_and_checking(
        self,
        eqn: jax.core.JaxprEqn
    ) -> Tuple[Path, jax.core.Var]:

        # Currently, only single input/output are supported, i.e.,
        #       y = f(x, w1, w2, ...)
        # This may be changed in the future, to support multiple inputs and outputs, i.e.,
        #       y1, y2, ... = f(x1, x2, ..., w1, w2, ...)
        #
        # However, I do not see any possibility or necessity for this kind of design in the
        # current stage. In most situations, single input/output is enough for the brain dynamics model.

        found_invars_in_this_op = set()
        weight_paths = set()
        xs = []
        for invar in eqn.invars:
            if isinstance(invar, jax.core.Literal):
                xs.append(invar)
                continue
            weight_path = self.invar_to_weight_path.get(invar, None)
            if weight_path is None:
                xs.append(invar)
            else:
                weight_paths.add(weight_path)
                found_invars_in_this_op.add(invar)

        # --- checking whether the weight variables are all used in the same etrace operation --- #
        if len(weight_paths) == 0:
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

        if len(weight_paths) > 1:
            name_stack = source_info_util.current_name_stack() + eqn.source_info.name_stack
            with source_info_util.user_context(eqn.source_info.traceback, name_stack=name_stack):
                raise CompilationError(
                    f'Error: multiple ETraceParam ({weight_paths}) are found in this operation. '
                    f'This is not allowed for automatic online learning: \n\n'
                    f'The Jaxpr for the operator: \n\n'
                    f'{eqn} \n\n'
                    f'The corresponding Python code for the operator: \n\n'
                    f'{jaxpr_to_python_code(_jax_eqn_to_jaxpr(eqn))} \n\n'
                    f'See the above traceback information for where the operation is defined in your code.'
                )

        weight_path = tuple(weight_paths)[0]  # the only ETraceParam found in the operation
        if len(found_invars_in_this_op.difference(set(self.weight_path_to_vars[weight_path]))) > 0:
            name_stack = source_info_util.current_name_stack() + eqn.source_info.name_stack
            with source_info_util.user_context(eqn.source_info.traceback, name_stack=name_stack):
                raise CompilationError(
                    f'Error: The found jax vars are {found_invars_in_this_op}, '
                    f'but the ETraceParam contains vars {self.weight_path_to_vars[weight_path]}. \n'
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
        return weight_path, xs[0]


def _hpo_tracer_to_relation(
    hid_relation: HiddenGroup,
    hpo_tracer: HiddenWeightOpTracer,
    hid_path_to_group: Dict[Path, 'HiddenGroupV2'],
    hid_path_to_transition: Dict[Path, 'HiddenTransition'],
    state_id_to_path: Dict[int, Path],
    outvar_to_hidden_path: Dict[jax.core.Var, Path]
) -> WeightOpHiddenRelation | None:
    hpo_tracer = hpo_tracer.replace(hidden_vars=list(hid_relation.hidden_outvars))
    return _trace_simplify(
        hpo_tracer,
        hid_path_to_group=hid_path_to_group,
        hid_path_to_transition=hid_path_to_transition,
        state_id_to_path=state_id_to_path,
        outvar_to_hidden_path=outvar_to_hidden_path
    )


def _simplify_hidden_eqns(
    hidden_invars: List[HiddenInVar],
    hidden_outvars: List[HiddenOutVar],
    eqns: List[jax.core.JaxprEqn]
):
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


class JaxprEvaluationForHiddenGroup(JaxprEvaluation):
    """
    Evaluating the jaxpr for extracting the hidden state ``hidden-to-hidden`` relationships.

    Args:
        jaxpr: The jaxpr for the model.
        hidden_outvar_to_invar: The mapping from the hidden output variable to the hidden input variable.
        weight_invars: The weight input variables.
        invar_to_hidden_path: The mapping from the weight input variable to the hidden state path.
        outvar_to_hidden_path: The mapping from the hidden output variable to the hidden state path.
    """

    def __init__(
        self,
        jaxpr: jax.core.Jaxpr,
        hidden_outvar_to_invar: Dict[HiddenOutVar, HiddenInVar],
        weight_invars: Set[jax.core.Var],
        invar_to_hidden_path: Dict[HiddenInVar, Path],
        outvar_to_hidden_path: Dict[HiddenOutVar, Path],
    ):
        # the jaxpr of the original model, assuming that the model is well-defined,
        # see the doc for the model which can be online learning compiled.
        self.jaxpr = jaxpr

        # the hidden state groups
        self.hidden_outvar_to_invar = hidden_outvar_to_invar
        self.hidden_invar_to_outvar = {invar: outvar for outvar, invar in hidden_outvar_to_invar.items()}
        hidden_invars = set(hidden_outvar_to_invar.values())
        hidden_outvars = set(hidden_outvar_to_invar.keys())

        # the data structures for the tracing hidden-hidden relationships
        self.active_tracings = dict()

        super().__init__(
            weight_invars=weight_invars,
            hidden_invars=hidden_invars,
            hidden_outvars=hidden_outvars,
            invar_to_hidden_path=invar_to_hidden_path,
            outvar_to_hidden_path=outvar_to_hidden_path
        )

    def compile(self) -> Tuple[
        Sequence[HiddenGroupV2],
        Dict[Path, HiddenGroupV2],
        Dict[Path, HiddenTransition]
    ]:
        """
        Compiling the jaxpr for the etrace relationships.
        """

        # the data structures for the tracing hidden-hidden relationships
        self.active_tracings: Dict[jax.core.Var, HiddenToHiddensTracer] = dict()

        # evaluating the jaxpr
        self._eval_jaxpr(self.jaxpr)

        # post checking
        hid_groups, hid_path_to_group, hid_path_to_transition = self._post_check()

        # reset the temporal data structures
        self.active_tracings = dict()
        return hid_groups, hid_path_to_group, hid_path_to_transition

    def _eval_jaxpr(self, jaxpr) -> None:
        """
        Evaluating the jaxpr for extracting the etrace relationships.

        Args:
          jaxpr: The jaxpr for the model.
        """

        for eqn in jaxpr.eqns:
            # TODO: add the support for the scan, while, cond, pjit, and other operators
            # Currently, scan, while, and cond are usually not the common operators used in
            # the definition of a brain dynamics model. So we may not need to consider them
            # during the current phase.
            # However, for the long-term maintenance and development, we need to consider them,
            # since users usually create crazy models.

            if eqn.primitive.name == 'pjit':
                self._eval_pjit(eqn)
            elif eqn.primitive.name == 'scan':
                self._eval_scan(eqn)
            elif eqn.primitive.name == 'while':
                self._eval_while(eqn)
            elif eqn.primitive.name == 'cond':
                self._eval_cond(eqn)
            else:
                self._eval_eqn(eqn)

    def _eval_pjit(self, eqn: jax.core.JaxprEqn) -> None:
        """
        Evaluating the pjit primitive.
        """
        if is_etrace_op(eqn.params['name']):
            if is_etrace_op_enable_gradient(eqn.params['name']):
                self._eval_eqn(eqn)
            return

        # checking whether the weight variables are used in the pjit
        invar = _check_some_element_exist_in_the_set(eqn.invars, self.weight_invars)
        if invar is not None:
            raise NotImplementedError(
                'Currently, we do not support the weight states are used within a jit function. \n'
                'Please remove your jit compilation on the intermediate steps. \n\n'
                f'The weight state is: {self.invar_to_hidden_path[invar]}. \n'
                f'The Jaxpr of the JIT function is: \n\n'
                f'{eqn} \n\n'
            )

        # checking whether the hidden variables are computed in the pjit
        outvar = _check_some_element_exist_in_the_set(eqn.outvars, self.hidden_outvars)
        if outvar is not None:
            raise NotImplementedError(
                'Currently, we do not support the hidden states are computed within a jit function. \n'
                'Please remove your jit compilation on the intermediate steps. \n\n'
                f'The hidden state is: {self.outvar_to_hidden_path[outvar]}. \n'
                f'The Jaxpr of the JIT function is: \n\n'
                f'{eqn} \n\n'
            )

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

    def _post_check(self) -> Tuple[
        Sequence[HiddenGroupV2],
        Dict[Path, HiddenGroupV2],
        Dict[Path, HiddenTransition]
    ]:
        # [First step]
        # check the following items:
        #
        # 1. the shape of connected hidden states should be the same
        # 2. simplify the trace
        # 3. remove the unnecessary hidden states

        hidden_to_group = [
            _simplify_hid2hid_tracer(
                tracer,
                self.invar_to_hidden_path,
                self.outvar_to_hidden_path,
            )
            for tracer in self.active_tracings.values()
        ]

        # [second step]
        # Find out the hidden group,
        # i.e., the hidden states that are connected to each other, the union of all hidden2group
        groups = [
            set(
                [self.hidden_invar_to_outvar[transition.hidden_invar]] +
                list(transition.connected_hidden_outvars)
            )
            for transition in hidden_to_group
        ]
        group_sets = self._group_merging(groups)

        # transform the hidden group set to the HiddenGroup
        groups = []
        # for group in group_sets:
        #     hidden_outvars = list(group)
        #     hidden_invars = [self.hidden_outvar_to_invar[outvar] for outvar in hidden_outvars]
        #     hidden_states = [self.hidden_outvar_to_hidden[outvar] for outvar in hidden_outvars]
        #     group = HiddenGroup(
        #         hidden_invars=hidden_invars,
        #         hidden_outvars=hidden_outvars,
        #         hidden_states=hidden_states
        #     )
        #     groups.append(group)

        for group in group_sets:
            hidden_outvars = list(group)
            hidden_invars = [self.hidden_outvar_to_invar[outvar] for outvar in hidden_outvars]
            group = HiddenGroupV2(
                hidden_invars=hidden_invars,
                hidden_outvars=hidden_outvars,
                hidden_paths=[self.outvar_to_hidden_path[outvar] for outvar in hidden_outvars],
            )
            groups.append(group)

        # hidden_outvar to group
        hidden_path_to_group = dict()
        for group in groups:
            for hid in group.hidden_outvars:
                path = self.outvar_to_hidden_path[hid]
                if path in hidden_path_to_group:
                    raise ValueError(
                        f'Error: the hidden state {path} '
                        f'is found in multiple groups. \n'
                        f'{hidden_path_to_group[path]} '
                        f'\n\n'
                        f'{group}'
                    )
                hidden_path_to_group[path] = group

        # hidden_outvar to transition:
        #
        #   h_1^t, h_2^t, ... = f(h_i^t-1, ....)
        #
        hidden_path_to_transition = dict()
        for transition in hidden_to_group:
            transition: HiddenTransition
            path = self.invar_to_hidden_path[transition.hidden_invar]
            hidden_path_to_transition[path] = transition

        return groups, hidden_path_to_group, hidden_path_to_transition

    @staticmethod
    def _group_merging(groups) -> List[frozenset[HiddenOutVar]]:
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


class JaxprEvaluationForHiddenPerturbation(JaxprEvaluation):
    """
    Adding perturbations to the hidden states in the jaxpr, and replacing the hidden states with the perturbed states.

    Args:
        closed_jaxpr: The closed jaxpr for the model.
        outvar_to_hidden_path: The mapping from the outvar to the state id.
        hidden_outvar_to_invar: The mapping from the hidden output variable to the hidden input variable.
        weight_invars: The weight input variables.
        invar_to_hidden_path: The mapping from the weight input variable to the hidden state path.

    Returns:
        The revised closed jaxpr with the perturbations.

    """

    def __init__(
        self,
        closed_jaxpr: jax.core.ClosedJaxpr,
        hidden_outvar_to_invar: Dict[HiddenOutVar, HiddenInVar],
        weight_invars: Set[jax.core.Var],
        invar_to_hidden_path: Dict[HiddenInVar, Path],
        outvar_to_hidden_path: Dict[jax.core.Var, Path],
    ):
        # necessary data structures
        self.closed_jaxpr = closed_jaxpr

        # initialize the super class
        super().__init__(
            weight_invars=weight_invars,
            hidden_invars=set(hidden_outvar_to_invar.values()),
            hidden_outvars=set(hidden_outvar_to_invar.keys()),
            invar_to_hidden_path=invar_to_hidden_path,
            outvar_to_hidden_path=outvar_to_hidden_path
        )

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
                f'{[self.outvar_to_hidden_path[v] for v in self.hidden_jaxvars_to_remove]}'
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


class HiddenTransition(NamedTuple):
    """
    Hidden state transition.
    """

    # the hidden state h_i^{t-1}
    hidden_invar: jax.core.Var
    hidden_path: Path

    # the connected hidden states h_1^t, h_2^t, ...
    connected_hidden_outvars: List[jax.core.Var]
    connected_hidden_paths: List[Path]

    # the jaxpr for computing hidden state transitions
    #
    # h_1^t, h_2^t, ... = f(h_i^{t-1}, x)
    #
    transition_jaxpr: jax.core.Jaxpr

    # the other input variables for jaxpr evaluation
    other_invars: List[jax.core.Var]

    def state_transition(
        self,
        old_hidden_val: jax.Array,
        other_input_vals: PyTree,
        return_index: Optional[int] = None
    ) -> List[jax.Array] | jax.Array:
        """
        Computing the hidden state transitions :math:`h^t = f(h_i^t, x)`.

        Args:
          old_hidden_val: The old hidden state value.
          other_input_vals: The input values.
          return_index: index of the hidden state to return.

        Returns:
          The new hidden state values.
        """
        new_hidden_vals = jax.core.eval_jaxpr(self.transition_jaxpr, other_input_vals, old_hidden_val)
        if return_index is not None:
            return new_hidden_vals[return_index]
        return new_hidden_vals


class HiddenGroup(NamedTuple):
    r"""
    The data structure for recording the hidden-to-hidden relation.

    The following fields are included:

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


class HiddenGroupV2(NamedTuple):
    r"""
    The data structure for recording the hidden-to-hidden relation.

    The following fields are included:

    - hidden_paths: the path to each hidden state

    This relation is used for computing the hidden-to-hidden state transitions::

        h_{t+1} = f(h_t, x_t)

    where ``h_t`` is the hidden state defined in ``hidden_vars``, ``x_t`` is the input at time ``t``
    defined in ``input_vars``, and ``f`` is the hidden state transition function which is defined
    in ``jaxpr``.

    """

    hidden_invars: List[HiddenInVar]  # the input hidden states
    hidden_outvars: List[HiddenOutVar]  # the output hidden states
    hidden_paths: List[Path]  # the input hidden states


class WeightOpHiddenRelation(NamedTuple):
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
    path: Path
    op_jaxpr: jax.core.Jaxpr
    x: WeightXVar
    y: WeightYVar
    jaxpr_y2hid: jax.core.Jaxpr
    # hidden_vars: List[HiddenOutVar]
    # hidden_groups: List[HiddenGroup]
    # hidden_var_to_transition: Dict[HiddenOutVar, HiddenTransition]
    hidden_paths: List[Path]
    hidden_groups: List[HiddenGroupV2]
    hidden_path_to_transition: Dict[Path, HiddenTransition]


class CompiledGraph(NamedTuple):
    """
    The compiled graph for the eligibility trace.

    The following fields are included:

    - augmented_jaxpr: the jaxpr that return necessary intermediate variables
    - jaxpr_perturb_hidden: the jaxpr the add hidden perturbation
    - stateful_fn_states: the states of the stateful function
    - stateful_fn_outtree: the output tree of the stateful function
    - hidden_param_op_relations: the hidden-to-weight relation
    - hid_invar_to_path: the mapping from the hidden input variable to the hidden state path
    - hid_outvar_to_path: the mapping from the hidden output variable to the hidden state path
    - hid_path_to_transition: the mapping from the hidden state path to the hidden state transition
    - out_hidden_jaxvars: the output hidden jax variables
    - out_wx_jaxvars: the output weight x jax variables
    - out_all_jaxvars: the output all jax variables
    - out_state_jaxvars: the output state jax variables
    - num_out: the number of outputs

    """
    augmented_jaxpr: jax.core.ClosedJaxpr  # the jaxpr that return necessary intermediate variables
    jaxpr_perturb_hidden: jax.core.ClosedJaxpr  # the jaxpr the add hidden perturbation
    stateful_fn_states: Sequence[bst.State]
    stateful_fn_outtree: jax.tree_util.PyTreeDef
    hidden_groups: Sequence[HiddenGroupV2]
    hidden_param_op_relations: Sequence[WeightOpHiddenRelation]
    hid_path_to_transition: Dict[Path, HiddenTransition]
    hid_invar_to_path: Dict[jax.core.Var, Path]
    hid_outvar_to_path: Dict[jax.core.Var, Path]
    out_hidden_jaxvars: List[jax.core.Var]
    out_wx_jaxvars: List[jax.core.Var]
    out_all_jaxvars: List[jax.core.Var]
    out_state_jaxvars: List[jax.core.Var]
    num_out: int


def compile_graph(
    model: bst.nn.Module,
    multi_step: bool,
    *model_args,
    **model_kwargs
) -> CompiledGraph:
    """
    Building the eligibility trace graph for the model according to the given inputs.

    This is the most important method for the eligibility trace graph. It builds the
    graph for the model, which is used for computing the weight spatial gradients and
    the hidden state Jacobian.

    """

    assert isinstance(model, bst.nn.Module), (
        "The model should be an instance of bst.nn.Module. "
        "Since it allows the explicit definition of the model structure."
    )
    states_ = bst.graph.states(model)
    path_to_states: Dict[Path, bst.State] = {
        path: state
        for path, state in states_.items()
    }
    state_id_to_path: Dict[StateID, Path] = {
        id(state): path
        for path, state in states_.items()
    }
    del states_

    def model_to_check_weight_assign(*args_, **kwargs_):
        with bst.StateTraceStack() as trace:
            out = model(*args_, **kwargs_)

        for st, write in zip(trace.states, trace.been_writen):
            if isinstance(st, bst.ParamState) and write:
                raise NotSupportedError(
                    f'The parameter state "{st}" is rewritten in the model. Currently, the '
                    f'online learning method we provided does not support the dynamical '
                    f'weight parameters. '
                )
        return out

    # --- stateful model, for extracting states, weights, and variables --- #
    #
    # [ NOTE ]
    # The model does not support "static_argnums" for now.
    # Please always use ``functools.partial`` to fix the static arguments.
    #
    # wrap the model so that we can track the iteration number
    stateful_model = bst.compile.StatefulFunction(model_to_check_weight_assign)

    # -- compile the model -- #
    #
    # NOTE:
    # The model does not support "static_argnums" for now.
    # Please always use functools.partial to fix the static arguments.
    #
    stateful_model.make_jaxpr(*model_args, **model_kwargs)

    # -- states -- #
    states = stateful_model.get_states()

    # -- jaxpr -- #
    closed_jaxpr = stateful_model.get_jaxpr()
    jaxpr = closed_jaxpr.jaxpr

    # -- finding the corresponding in/out vars of etrace states and weights -- #
    out_shapes = stateful_model.get_out_shapes()[0]
    state_vals = [state.value for state in states]
    in_avals, _ = jax.tree.flatten((model_args, model_kwargs))
    out_avals, _ = jax.tree.flatten(out_shapes)
    num_in = len(in_avals)
    num_out = len(out_avals)
    state_avals, state_tree = jax.tree.flatten(state_vals)
    assert len(jaxpr.invars) == len(in_avals) + len(state_avals)
    assert len(jaxpr.outvars) == num_out + len(state_avals)
    invars_with_state_tree = jax.tree.unflatten(state_tree, jaxpr.invars[num_in:])
    outvars_with_state_tree = jax.tree.unflatten(state_tree, jaxpr.outvars[num_out:])

    # remove the quantity from the invars and outvars
    invars_with_state_tree = _remove_quantity(invars_with_state_tree)
    outvars_with_state_tree = _remove_quantity(outvars_with_state_tree)

    # -- checking weights as invar -- #
    weight_path_to_invar = {
        state_id_to_path[id(st)]: jax.tree.leaves(invar)
        for invar, st in zip(invars_with_state_tree, states)
        if isinstance(st, ETraceParam)
    }
    hidden_path_to_invar = {  # one-to-many mapping
        state_id_to_path[id(st)]: invar  # ETraceState only contains one Array, "invar" is the jaxpr var
        for invar, st in zip(invars_with_state_tree, states)
        if isinstance(st, ETraceState)
    }
    invar_to_hidden_path = {
        invar: path
        for path, invar in hidden_path_to_invar.items()
    }
    invar_to_weight_path = {  # many-to-one mapping
        v: k
        for k, vs in weight_path_to_invar.items()
        for v in vs
    }

    # -- checking states as outvar -- #
    hidden_path_to_outvar = {  # one-to-one mapping
        state_id_to_path[id(st)]: outvar  # ETraceState only contains one Array, "outvar" is the jaxpr var
        for outvar, st in zip(outvars_with_state_tree, states)
        if isinstance(st, ETraceState)
    }
    outvar_to_hidden_path = {  # one-to-one mapping
        v: state_id
        for state_id, v in hidden_path_to_outvar.items()
    }
    hidden_outvar_to_invar = {
        outvar: hidden_path_to_invar[hid]
        for hid, outvar in hidden_path_to_outvar.items()
    }
    hidden_outvar_to_hidden = {
        outvar: st
        for outvar, st in zip(outvars_with_state_tree, states)
        if isinstance(st, ETraceState)
    }
    hidden_invar_to_hidden = {
        invar: st
        for invar, st in zip(invars_with_state_tree, states)
        if isinstance(st, ETraceState)
    }

    # -- evaluating the relationship for hidden-to-hidden -- #
    weight_invars = set([v for vs in weight_path_to_invar.values() for v in vs])
    (
        hidden_groups,
        hid_path_to_group,
        hid_path_to_transition
    ) = JaxprEvaluationForHiddenGroup(
        jaxpr=jaxpr,
        hidden_outvar_to_invar=hidden_outvar_to_invar,
        weight_invars=weight_invars,
        invar_to_hidden_path=invar_to_hidden_path,
        outvar_to_hidden_path=outvar_to_hidden_path,
    ).compile()

    # -- evaluating the jaxpr for (hidden, weight, op) relationships -- #

    hidden_param_op_relations = JaxprEvaluationForWeightOpHiddenRelation(
        jaxpr=jaxpr,
        hidden_outvar_to_invar=hidden_outvar_to_invar,
        weight_path_to_vars=weight_path_to_invar,
        invar_to_weight_path=invar_to_weight_path,
        path_to_state=path_to_states,
        hid_path_to_group=hid_path_to_group,
        hid_path_to_transition=hid_path_to_transition,
        invar_to_hidden_path=invar_to_hidden_path,
        outvar_to_hidden_path=outvar_to_hidden_path,
    ).compile()

    # --- Collect the Var needed to compute the weight spatial gradients --- #
    # ---      Rewrite the jaxpr for computing the needed variables      --- #

    # all states jaxpr var
    out_state_jaxvars = list(jaxpr.outvars[num_out:])
    (
        weight_jaxvar_tree,
        hidden_jaxvar,
        other_state_jaxvar_tree
    ) = sequence_split_state_values(states, outvars_with_state_tree)

    # all jaxpr var of ETraceState, one etrace to one Array
    # All ETrace Hidden state
    out_hidden_jaxvars = list(hidden_jaxvar)

    # all weight x
    out_wx_jaxvars = list(
        set([relation.x for relation in hidden_param_op_relations])
    )

    # all y-to-hidden vars
    out_wy2hid_jaxvars = list(
        set(
            [v for relation in hidden_param_op_relations
             for v in (relation.jaxpr_y2hid.invars +
                       relation.jaxpr_y2hid.constvars)]
        )
    )

    # hidden-hidden transition vars
    hid2hid_jaxvars = set()
    for group in hidden_groups:
        hid2hid_jaxvars.update([v for v in group.hidden_invars])
    for transition in hid_path_to_transition.values():
        hid2hid_jaxvars.update([v for v in transition.other_invars])

    # all outvars
    all_outvars = list(
        set(
            out_state_jaxvars +  # all state variables
            out_wx_jaxvars +  # all weight x
            out_wy2hid_jaxvars +  # all y-to-hidden invars
            list(hid2hid_jaxvars)  # all hidden-hidden transition vars
        )
    )
    out_all_jaxvars = jaxpr.outvars[:num_out] + all_outvars

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
        outvars=list(out_all_jaxvars),
        eqns=list(jaxpr.eqns)
    )
    augmented_jaxpr = jax.core.ClosedJaxpr(jaxpr, closed_jaxpr.consts)

    if multi_step:
        jaxpr_with_hidden_perturb = None

    else:
        # ---               add perturbations to the hidden states                  --- #
        # --- new jaxpr with hidden state perturbations for computing the residuals --- #

        jaxpr_with_hidden_perturb = JaxprEvaluationForHiddenPerturbation(
            closed_jaxpr=augmented_jaxpr,
            hidden_outvar_to_invar=hidden_outvar_to_invar,
            weight_invars=weight_invars,
            invar_to_hidden_path=invar_to_hidden_path,
            outvar_to_hidden_path=outvar_to_hidden_path,
        ).compile()

    cache_key = stateful_model.get_arg_cache_key(*model_args, **model_kwargs)

    return CompiledGraph(
        augmented_jaxpr=augmented_jaxpr,
        jaxpr_perturb_hidden=jaxpr_with_hidden_perturb,
        stateful_fn_states=stateful_model.get_states(cache_key),
        stateful_fn_outtree=stateful_model.get_out_treedef(cache_key),
        hidden_groups=hidden_groups,
        hidden_param_op_relations=hidden_param_op_relations,
        hid_path_to_transition=hid_path_to_transition,
        hid_invar_to_path=invar_to_hidden_path,
        hid_outvar_to_path=outvar_to_hidden_path,
        out_hidden_jaxvars=out_hidden_jaxvars,
        out_wx_jaxvars=out_wx_jaxvars,
        out_all_jaxvars=out_all_jaxvars,
        out_state_jaxvars=out_state_jaxvars,
        num_out=num_out,
    )
