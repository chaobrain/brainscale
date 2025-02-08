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
from typing import Dict, Sequence, Set, List, NamedTuple, Tuple, Any

import brainstate as bst
import jax.core

from ._etrace_concepts import (
    ETraceParam,
    ETraceState,
)
from ._etrace_operators import is_etrace_op, is_etrace_op_enable_gradient
from ._misc import NotSupportedError, unknown_state_path
from ._misc import _remove_quantity
from ._state_managment import sequence_split_state_values
from ._typing import (
    Path,
    StateID,
    Inputs,
    Outputs,
    ETraceVals,
    StateVals,
    TempData,
)

if jax.__version_info__ < (0, 4, 38):
    from jax.core import Var, JaxprEqn, Jaxpr, ClosedJaxpr
else:
    from jax.extend.core import Var, JaxprEqn, ClosedJaxpr


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
    __module__ = 'brainscale'

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
    """
    The model information for the etrace compiler.

    The model information contains the at least five categories of information:

    1. The stateful model.

         - ``stateful_model``: The stateful model is the model that compiles the model
           into abstract jaxpr representation.

    2. The jaxpr.

         The jaxpr is the abstract representation of the model.

         - ``closed_jaxpr``: The closed jaxpr is the closed jaxpr representation of the model.
         - ``jaxpr``: The jaxpr is the open jaxpr representation of the model.

    3. The states.

        - ``retrieved_model_states``: The model states that are retrieved from the ``model.states()`` function,
           which has well-defined paths and structures.
        - ``compiled_model_states``: The model states that are compiled from the stateful model, which is
           accurate and consistent with the model jaxpr, but loss the path information.
        - ``state_id_to_path``: The mapping from the state id to the state path.

    4. The hidden states.

        - ``hidden_path_to_invar``: The mapping from the hidden path to the input variable.
        - ``hidden_path_to_outvar``: The mapping from the hidden path to the output variable.
        - ``invar_to_hidden_path``: The mapping from the input variable to the hidden path.
        - ``outvar_to_hidden_path``: The mapping from the output variable to the hidden path.
        - ``hidden_outvar_to_invar``: The mapping from the output variable to the input variable.

    5. The parameter weights.

        - ``weight_invars``: The weight input variables.
        - ``weight_path_to_invars``: The mapping from the weight path to the input variables.
        - ``invar_to_weight_path``: The mapping from the input variable to the weight path.

    """
    # stateful model
    stateful_model: bst.compile.StatefulFunction

    # jaxpr
    closed_jaxpr: jax.core.ClosedJaxpr

    # states
    retrieved_model_states: bst.util.FlattedDict[Path, bst.State]
    compiled_model_states: Sequence[bst.State]
    state_id_to_path: Dict[StateID, Path]
    state_tree_invars: bst.typing.PyTree[Var]
    state_tree_outvars: bst.typing.PyTree[Var]

    # hidden states
    hidden_path_to_invar: Dict[Path, Var]
    hidden_path_to_outvar: Dict[Path, Var]
    invar_to_hidden_path: Dict[Var, Path]
    outvar_to_hidden_path: Dict[Var, Path]
    hidden_outvar_to_invar: Dict[Var, Var]

    # parameter weights
    weight_invars: Set[Var]
    weight_path_to_invars: Dict[Path, List[Var]]
    invar_to_weight_path: Dict[Var, Path]

    # output
    num_var_out: int  # number of original output variables
    num_var_state: int  # number of state variable outputs

    def dict(self) -> Dict[str, Any]:
        return dict(self._asdict())

    @property
    def jaxpr(self) -> jax.core.Jaxpr:
        """
        The jaxpr of the model.
        """
        return self.closed_jaxpr.jaxpr

    def add_jaxpr_outs(
        self,
        jax_vars: Sequence[Var],
        inplace: bool = True,
    ) -> ModelInfo:
        assert all(isinstance(v, Var) for v in jax_vars), 'The jax_vars should be the instance of Var.'
        jaxpr = Jaxpr(
            constvars=list(self.jaxpr.constvars),
            invars=list(self.jaxpr.invars),
            outvars=list(jax_vars),
            eqns=list(self.jaxpr.eqns),
            effects=self.jaxpr.effects,
            debug_info=self.jaxpr.debug_info,
        )
        closed_jaxpr = ClosedJaxpr(jaxpr, self.closed_jaxpr.consts)

        if inplace:
            items = self.dict()
            items['closed_jaxpr'] = closed_jaxpr
            return ModelInfo(**items)

        else:
            self.closed_jaxpr = closed_jaxpr
            return self

    def split_state_outvars(self):
        """
        Splitting the state outvars into three parts: weight, hidden, and other states.

        Returns:
            weight_jaxvar_tree: The weight tree of jax Var.
            hidden_jaxvar: The hidden tree of jax Var.
            other_state_jaxvar_tree: The other state tree of jax Var.
        """
        (
            weight_jaxvar_tree,
            hidden_jaxvar,
            other_state_jaxvar_tree
        ) = sequence_split_state_values(self.compiled_model_states, self.state_tree_outvars)
        return weight_jaxvar_tree, hidden_jaxvar, other_state_jaxvar_tree

    def jaxpr_call(
        self,
        *args: Inputs,
    ) -> Tuple[
        Outputs,
        ETraceVals,
        StateVals,
        TempData,
    ]:
        """
        Computing the model according to the given inputs and parameters by using the compiled jaxpr.

        Args:
            args: The inputs of the model.

        Returns:
            out: The output of the model.
            etrace_vals: The values for etrace states.
            oth_state_vals: The other state values.
            temps: The temporary intermediate values.
        """

        # state checking
        old_state_vals = [st.value for st in self.compiled_model_states]

        # parameters
        args = jax.tree.leaves((args, old_state_vals))

        # calling the function
        jaxpr_outs = jax.core.eval_jaxpr(
            self.closed_jaxpr.jaxpr,
            self.closed_jaxpr.consts,
            *args
        )

        # intermediate values
        #
        # "jaxpr_outs[:self.num_out]" corresponds to model original outputs
        #     - Outputs
        # "jaxpr_outs[self.num_out:]" corresponds to extra output in  "augmented_jaxpr"
        #     - others
        temps = {
            v: r for v, r in
            zip(
                self.jaxpr.outvars[self.num_var_out:],
                jaxpr_outs[self.num_var_out:]
            )
        }

        #
        # recovery outputs of ``stateful_model``
        #
        cache_key = self.stateful_model.get_arg_cache_key(*args)
        i_start = self.num_var_out
        i_end = i_start + self.num_var_state
        out, new_state_vals = self.stateful_model.get_out_treedef(cache_key).unflatten(jaxpr_outs[:i_end])

        #
        # check state value
        assert len(old_state_vals) == len(new_state_vals), 'State length mismatch.'

        #
        # split the state values
        #
        etrace_vals = dict()
        oth_state_vals = dict()
        for st, st_val in zip(self.compiled_model_states, new_state_vals):
            if isinstance(st, ETraceState):
                etrace_vals[self.state_id_to_path[id(st)]] = st_val
            elif isinstance(st, bst.ParamState):
                # assume they are not changed
                pass
            else:
                oth_state_vals[self.state_id_to_path[id(st)]] = st_val
        return out, etrace_vals, oth_state_vals, temps


ModelInfo.__module__ = 'brainscale'


def extract_model_info(
    model: bst.nn.Module,
    *model_args,
    **model_kwargs
) -> ModelInfo:
    """
    Extracting the model information for the etrace compiler.

    Args:
        model: The model to extract the information.
        model_args: The arguments of the model.
        model_kwargs: The keyword arguments of the model.

    Returns:
        The model information.
    """

    # abstract the model
    (
        stateful_model,
        model_retrieved_states
    ) = abstractify_model(
        model,
        *model_args,
        **model_kwargs
    )

    # state information
    cache_key = stateful_model.get_arg_cache_key(*model_args, **model_kwargs)
    compiled_states = stateful_model.get_states(cache_key)

    state_id_to_path: Dict[StateID, Path] = {
        id(state): path
        for path, state in model_retrieved_states.items()
    }

    closed_jaxpr = stateful_model.get_jaxpr(cache_key)
    jaxpr = closed_jaxpr.jaxpr

    # out information
    out_shapes = stateful_model.get_out_shapes(cache_key)[0]
    state_vals = [state.value for state in compiled_states]
    in_avals, _ = jax.tree.flatten((model_args, model_kwargs))
    out_avals, _ = jax.tree.flatten(out_shapes)
    num_in = len(in_avals)
    num_out = len(out_avals)
    state_avals, state_tree = jax.tree.flatten(state_vals)
    state_tree_invars = jax.tree.unflatten(state_tree, jaxpr.invars[num_in:])
    state_tree_outvars = jax.tree.unflatten(state_tree, jaxpr.outvars[num_out:])

    # remove the quantity from the invars and outvars
    state_tree_invars = _remove_quantity(state_tree_invars)
    state_tree_outvars = _remove_quantity(state_tree_outvars)

    # -- checking weights as invar -- #
    weight_path_to_invars = {
        state_id_to_path[id(st)]: jax.tree.leaves(invar)
        for invar, st in zip(state_tree_invars, compiled_states)
        if isinstance(st, ETraceParam)
    }
    hidden_path_to_invar = {  # one-to-many mapping
        state_id_to_path[id(st)]: invar  # ETraceState only contains one Array, "invar" is the jaxpr var
        for invar, st in zip(state_tree_invars, compiled_states)
        if isinstance(st, ETraceState)
    }
    invar_to_hidden_path = {
        invar: path
        for path, invar in hidden_path_to_invar.items()
    }
    invar_to_weight_path = {  # many-to-one mapping
        v: k
        for k, vs in weight_path_to_invars.items()
        for v in vs
    }

    # -- checking states as outvar -- #
    hidden_path_to_outvar = {  # one-to-one mapping
        state_id_to_path[id(st)]: outvar  # ETraceState only contains one Array, "outvar" is the jaxpr var
        for outvar, st in zip(state_tree_outvars, compiled_states)
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
    weight_invars = set([v for vs in weight_path_to_invars.values() for v in vs])

    return ModelInfo(
        # stateful model
        stateful_model=stateful_model,

        # jaxpr
        closed_jaxpr=closed_jaxpr,

        # states
        retrieved_model_states=model_retrieved_states,
        compiled_model_states=compiled_states,
        state_id_to_path=state_id_to_path,
        state_tree_invars=state_tree_invars,
        state_tree_outvars=state_tree_outvars,

        # hidden states
        hidden_path_to_invar=hidden_path_to_invar,
        invar_to_hidden_path=invar_to_hidden_path,
        hidden_path_to_outvar=hidden_path_to_outvar,
        outvar_to_hidden_path=outvar_to_hidden_path,
        hidden_outvar_to_invar=hidden_outvar_to_invar,

        # parameter weights
        weight_invars=weight_invars,
        weight_path_to_invars=weight_path_to_invars,
        invar_to_weight_path=invar_to_weight_path,

        # output parameters
        num_var_out=num_out,
        num_var_state=len(jaxpr.outvars[num_out:]),
    )
