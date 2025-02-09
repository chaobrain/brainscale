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

from typing import Dict, Sequence, Set, List

import jax

from ._etrace_operators import (
    is_etrace_op,
    is_etrace_op_enable_gradient,
)
from ._typing import Path

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
