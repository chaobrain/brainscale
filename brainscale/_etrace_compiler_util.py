# Copyright 2025 BDP Ecosystem Limited. All Rights Reserved.
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

import functools
from typing import Dict, Sequence, Set, List, NamedTuple

import brainstate as bst
import jax.core

from ._etrace_concepts import (
    ETraceParam,
    ETraceState,
)
from ._etrace_operators import is_etrace_op, is_etrace_op_enable_gradient
from ._misc import NotSupportedError, unknown_state_path
from ._misc import _remove_quantity
from ._typing import Path, StateID

if jax.__version_info__ < (0, 4, 38):
    from jax.core import Var, JaxprEqn
else:
    from jax.extend.core import Var, JaxprEqn


def find_matched_vars(
    invars: Sequence[Var],
    invar_needed_in_oth_eqns: Set[Var]
) -> List[Var]:
    """
    Checking whether the invars are matched with the invar_needed_in_oth_eqns.

    Args:
      invars: The input variables of the equation.
      invar_needed_in_oth_eqns: The variables needed in the other equations.
    """
    matched = []
    for invar in invars:
        if isinstance(invar, Var) and invar in invar_needed_in_oth_eqns:
            matched.append(invar)
    return matched


def find_element_exist_in_the_set(
    elements: Sequence[Var],
    the_set: Set[Var]
) -> Var | None:
    """
    Checking whether the jaxpr vars contain the weight variables.

    Args:
      elements: The input variables of the equation.
      the_set: The set of the weight variables.
    """
    for invar in elements:
        if isinstance(invar, Var) and invar in the_set:
            return invar
    return None


def check_unsupported_op(
    self,
    eqn: JaxprEqn,
    op_name: str
):
    # checking whether the weight variables are used in the equation
    invar = find_element_exist_in_the_set(eqn.invars, self.weight_invars)
    if invar is not None:
        raise NotImplementedError(
            f'Currently, we do not support the weight states are used within a {op_name} function. \n'
            f'Please remove your {op_name} on the intermediate steps. \n\n'
            f'The weight state is: {self.invar_to_hidden_path[invar]}. \n'
            f'The Jaxpr of the {op_name} function is: \n\n'
            f'{eqn} \n\n'
        )

    # checking whether the hidden variables are computed in the equation
    outvar = find_element_exist_in_the_set(eqn.outvars, self.hidden_outvars)
    if outvar is not None:
        raise NotImplementedError(
            f'Currently, we do not support the hidden states are computed within a {op_name} function. \n'
            f'Please remove your {op_name} on the intermediate steps. \n\n'
            f'The hidden state is: {self.outvar_to_hidden_path[outvar]}. \n'
            f'The Jaxpr of the {op_name} function is: \n\n'
            f'{eqn} \n\n'
        )


class JaxprEvaluation(object):
    """
    The base class for evaluating the jaxpr for extracting the etrace relationships.

    Args:
        weight_invars: The input variables of the weight.
        hidden_invars: The input variables of the hidden states.
        hidden_outvars: The output variables of the hidden states.
        invar_to_hidden_path: The mapping from the input variables to the hidden states.
        outvar_to_hidden_path: The mapping from the output variables to the hidden states.
    """

    def __init__(
        self,
        weight_invars: Set[Var],
        hidden_invars: Set[Var],
        hidden_outvars: Set[Var],
        invar_to_hidden_path: Dict[Var, Path],
        outvar_to_hidden_path: Dict[Var, Path],
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

    def _eval_pjit(self, eqn: JaxprEqn) -> None:
        """
        Evaluating the pjit primitive.
        """
        if is_etrace_op(eqn.params['name']):
            if is_etrace_op_enable_gradient(eqn.params['name']):
                self._eval_eqn(eqn)
            return

        check_unsupported_op(self, eqn, 'jit')

        # treat the pjit as a normal jaxpr equation
        self._eval_eqn(eqn)

    def _eval_scan(self, eqn: JaxprEqn) -> None:
        """
        Evaluating the scan primitive.
        """
        check_unsupported_op(self, eqn, 'while')
        self._eval_eqn(eqn)

    def _eval_while(self, eqn: JaxprEqn) -> None:
        """
        Evaluating the while primitive.
        """
        check_unsupported_op(self, eqn, 'scan')
        self._eval_eqn(eqn)

    def _eval_cond(self, eqn: JaxprEqn) -> None:
        """
        Evaluating the cond primitive.
        """
        check_unsupported_op(self, eqn, 'cond')
        self._eval_eqn(eqn)

    def _eval_eqn(self, eqn):
        raise NotImplementedError(
            'The method "_eval_eqn" should be implemented in the subclass.'
        )


def _model_to_check_weight_assign(model, *args_, **kwargs_):
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


def _check_consistent_states_between_model_and_compiler(
    compiled_model_states: Sequence[bst.State],
    retrieved_model_states: Dict[Path, bst.State],
    verbose: bool = True,  # whether to print the information
):
    id_to_compiled_state = {
        id(st): st
        for st in compiled_model_states
    }
    id_to_path = {
        id(st): path
        for path, st in retrieved_model_states.items()
    }
    for id_ in id_to_path:
        if id_ not in id_to_compiled_state:
            path = id_to_path[id_]
            retrieved_model_states.pop(path)
            if verbose:
                print(f"Warning: the state {path} is not found in the compiled model.")
    i_unknown = 0
    for id_ in id_to_compiled_state:
        if id_ not in id_to_path:
            st = id_to_compiled_state[id_]
            if verbose:
                print(f"Warning: the state {st} is not found in the retrieved model. "
                      f"We have added this state.")
            retrieved_model_states[(unknown_state_path(i=i_unknown),)] = st
            i_unknown += 1


def abstractify_model(
    model: bst.nn.Module,
    *model_args,
    **model_kwargs
):
    assert isinstance(model, bst.nn.Module), (
        "The model should be an instance of bst.nn.Module. "
        "Since it allows the explicit definition of the model structure."
    )
    model_retrieved_states = bst.graph.states(model)

    # --- stateful model, for extracting states, weights, and variables --- #
    #
    # [ NOTE ]
    # The model does not support "static_argnums" for now.
    # Please always use ``functools.partial`` to fix the static arguments.
    #
    # wrap the model so that we can track the iteration number
    stateful_model = bst.compile.StatefulFunction(
        functools.partial(_model_to_check_weight_assign, model)
    )

    # -- compile the model -- #
    #
    # NOTE:
    # The model does not support "static_argnums" for now.
    # Please always use functools.partial to fix the static arguments.
    #
    stateful_model.make_jaxpr(*model_args, **model_kwargs)

    # -- states -- #
    cache_key = stateful_model.get_arg_cache_key(*model_args, **model_kwargs)
    compiled_states = stateful_model.get_states(cache_key)

    # check the consistency between the model and the compiler
    _check_consistent_states_between_model_and_compiler(
        compiled_states,
        model_retrieved_states
    )

    return stateful_model, model_retrieved_states


class ModelInfo(NamedTuple):
    stateful_model: bst.compile.StatefulFunction
    retrieved_model_states: Dict[Path, bst.State]
    compiled_model_states: Sequence[bst.State]
    state_id_to_path: Dict[StateID, Path]
    closed_jaxpr: jax.core.ClosedJaxpr
    jaxpr: jax.core.Jaxpr
    hidden_path_to_invar: Dict[Path, Var]
    invar_to_hidden_path: Dict[Var, Path]
    hidden_path_to_outvar: Dict[Path, Var]
    outvar_to_hidden_path: Dict[Var, Path]
    hidden_outvar_to_invar: Dict[Var, Var]
    weight_invars: Set[Var]


def extract_model_info(
    model: bst.nn.Module,
    *model_args,
    **model_kwargs
) -> ModelInfo:
    (
        stateful_model,
        model_retrieved_states
    ) = abstractify_model(
        model,
        *model_args,
        **model_kwargs
    )

    cache_key = stateful_model.get_arg_cache_key(*model_args, **model_kwargs)
    compiled_states = stateful_model.get_states(cache_key)

    state_id_to_path: Dict[StateID, Path] = {
        id(state): path
        for path, state in model_retrieved_states.items()
    }

    closed_jaxpr = stateful_model.get_jaxpr(cache_key)
    jaxpr = closed_jaxpr.jaxpr

    out_shapes = stateful_model.get_out_shapes(cache_key)[0]
    state_vals = [state.value for state in compiled_states]
    in_avals, _ = jax.tree.flatten((model_args, model_kwargs))
    out_avals, _ = jax.tree.flatten(out_shapes)
    num_in = len(in_avals)
    num_out = len(out_avals)
    state_avals, state_tree = jax.tree.flatten(state_vals)
    invars_with_state_tree = jax.tree.unflatten(state_tree, jaxpr.invars[num_in:])
    outvars_with_state_tree = jax.tree.unflatten(state_tree, jaxpr.outvars[num_out:])

    # remove the quantity from the invars and outvars
    invars_with_state_tree = _remove_quantity(invars_with_state_tree)
    outvars_with_state_tree = _remove_quantity(outvars_with_state_tree)

    # -- checking weights as invar -- #
    weight_path_to_invar = {
        state_id_to_path[id(st)]: jax.tree.leaves(invar)
        for invar, st in zip(invars_with_state_tree, compiled_states)
        if isinstance(st, ETraceParam)
    }
    hidden_path_to_invar = {  # one-to-many mapping
        state_id_to_path[id(st)]: invar  # ETraceState only contains one Array, "invar" is the jaxpr var
        for invar, st in zip(invars_with_state_tree, compiled_states)
        if isinstance(st, ETraceState)
    }
    invar_to_hidden_path = {
        invar: path
        for path, invar in hidden_path_to_invar.items()
    }

    # -- checking states as outvar -- #
    hidden_path_to_outvar = {  # one-to-one mapping
        state_id_to_path[id(st)]: outvar  # ETraceState only contains one Array, "outvar" is the jaxpr var
        for outvar, st in zip(outvars_with_state_tree, compiled_states)
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
    weight_invars = set([v for vs in weight_path_to_invar.values() for v in vs])

    return ModelInfo(
        stateful_model=stateful_model,
        retrieved_model_states=model_retrieved_states,
        compiled_model_states=compiled_states,
        state_id_to_path=state_id_to_path,
        closed_jaxpr=closed_jaxpr,
        jaxpr=jaxpr,
        hidden_path_to_invar=hidden_path_to_invar,
        invar_to_hidden_path=invar_to_hidden_path,
        hidden_path_to_outvar=hidden_path_to_outvar,
        outvar_to_hidden_path=outvar_to_hidden_path,
        hidden_outvar_to_invar=hidden_outvar_to_invar,
        weight_invars=weight_invars
    )
