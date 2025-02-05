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

from enum import Enum
from typing import Callable, Optional, Any

import brainstate as bst
import brainunit as u
import jax.lax
import numpy as np

from ._etrace_operators import ETraceOp
from ._misc import BaseEnum
from ._typing import WeightVals

__all__ = [
    # eligibility trace states
    'ETraceState',  # the hidden state for the etrace-based learning
    'ETraceGroupState',  # the hidden state group for the etrace-based learning

    # eligibility trace parameters and operations
    'ETraceParam',  # the parameter/weight for the etrace-based learning
    'ETraceParamOp',  # the parameter and operator for the etrace-based learning, combining ETraceParam and ETraceOp
    'NonTempParamOp',  # the parameter state with an associated operator without temporal dependent gradient learning

    # element-wise eligibility trace parameters
    'ElemWiseParamOp',  # the element-wise weight parameter for the etrace-based learning

    # fake parameter state
    'FakeParamOp',
    'FakeElemWiseParamOp',
]


# -------------------------------------------------------------------------------------- #
# Eligibility Trace Related Concepts
# -------------------------------------------------------------------------------------- #


class ETraceState(bst.ShortTermState):
    """
    The Hidden State for Eligibility Trace-based Learning.

    .. note::

        Currently, the hidden state only supports `jax.Array` or `brainunit.Quantity`.
        This means that each instance of :py:class:`ETraceState` should define
        single hidden variable.

        If you want to define multiple hidden variables within a single instance of
        :py:class:`ETraceState`, you can :py:class:`ETraceGroupState` instead.

    Args:
        value: The value of the hidden state.
               Currently only support a `jax.Array` or `brainunit.Quantity`.
        name: The name of the hidden state.
    """
    __module__ = 'brainscale'

    def __init__(
        self,
        value: bst.typing.ArrayLike,
        name: Optional[str] = None
    ):
        self._check_value(value)
        super().__init__(value, name=name)

    def _check_value(self, value: bst.typing.ArrayLike):
        if not isinstance(value, (np.ndarray, jax.Array, u.Quantity)):
            raise TypeError(
                f'Currently, {ETraceState.__name__} only supports '
                f'numpy.ndarray, jax.Array or brainunit.Quantity. '
                f'But we got {type(value)}.'
            )


class ETraceGroupState(bst.ShortTermState):
    """
    The hidden state group for eligibility trace-based learning.

    Args:
        value: The values of the hidden states.
    """

    __module__ = 'brainscale'

    def _check_value_tree(self, v):
        pass


class ETraceParam(bst.ParamState):
    """
    The Eligibility Trace Weight Parameter.

    Args:
        value: The value of the weight. Can be a PyTree.
        name: The name of the weight.
    """
    __module__ = 'brainscale'

    is_not_etrace: bool

    def __init__(
        self,
        value: bst.typing.PyTree,
        name: Optional[str] = None
    ):
        super().__init__(value, name=name)
        self.is_not_etrace = False


class _ETraceGrad(BaseEnum):
    full = 'full'
    approx = 'approx'
    adaptive = 'adaptive'


class ETraceParamOp(ETraceParam):
    """
    The Eligibility Trace Weight and its Associated Operator.

    Args:
      weight: The weight of the ETrace.
      op: The operator for the ETrace. See `ETraceOp`.
    """
    __module__ = 'brainscale'
    op: ETraceOp  # operator

    def __init__(
        self,
        weight: bst.typing.PyTree,
        op: Callable[[jax.Array, WeightVals], jax.Array],
        grad: Optional[str | Enum] = None,
        is_diagonal: bool = None,
        name: Optional[str] = None
    ):
        # weight value
        super().__init__(weight, name=name)

        # gradient
        if grad is None:
            grad = 'adaptive'
        self.gradient = _ETraceGrad.get(grad)

        # operation
        if isinstance(op, ETraceOp):
            self.op = op
            if is_diagonal is not None:
                self.op.is_diagonal = is_diagonal
        else:
            self.op = ETraceOp(op, is_diagonal=is_diagonal if is_diagonal is not None else False)

    def execute(self, x: jax.Array) -> jax.Array:
        return self.op(x, self.value)


class ElemWiseParamOp(ETraceParamOp):
    """
    The Element-wise Eligibility Trace Weight and its Associated Operator.

    Args:
      weight: The weight of the ETrace.
    """
    __module__ = 'brainscale'

    def __init__(
        self,
        weight: bst.typing.PyTree,
    ):
        from ._etrace_operators import ElementWiseOpV2
        super().__init__(weight, ElementWiseOpV2(), grad=_ETraceGrad.full, is_diagonal=True)

    def execute(self) -> Any:
        ones = u.math.zeros(1)
        return self.op(ones, self.value)


class NonTempParamOp(bst.ParamState):
    r"""
    The Parameter State with an Associated Operator with no temporal dependent gradient learning.

    This class behaves the same as :py:class:`ETraceParamOp`, but will not build the
    eligibility trace graph when using online learning. Therefore, in a sequence
    learning task, the weight gradient can only be computed with the spatial gradients.
    That is,

        $$
        \nabla \theta = \sum_t \partial L^t / \partial \theta^t
        $$

    Instead, the gradient of the weight $\theta$ which is labeled as :py:class:`ETraceParamOp` is
    computed as:

        $$
        \nabla \theta = \sum_t \partial L^t / \partial \theta = \sum_t \sum_i^t \partial L^t / \partial \theta^i
        $$

    Args:
      value: The value of the parameter.
      op: The operator for the parameter. See `ETraceOp`.
    """
    __module__ = 'brainscale'
    op: Callable  # operator

    def __init__(
        self,
        value: bst.typing.PyTree,
        op: Callable,
        name: Optional[str] = None,
        **kwargs
    ):
        super().__init__(value, name=name)

        # operation
        if isinstance(op, ETraceOp):
            op = op.fun
        self.op = op

    def execute(self, x: jax.Array) -> jax.Array:
        return self.op(x, self.value)


class FakeParamOp(object):
    """
    The Parameter State with an Associated Operator that does not require to compute gradients.

    This class corresponds to the :py:class:`NonTempParamOp` but does not require to compute gradients.
    It has the same usage interface with :py:class:`NonTempParamOp`.

    Args:
      value: The value of the parameter.
      op: The operator for the parameter.
    """
    __module__ = 'brainscale'

    def __init__(
        self,
        value: bst.typing.PyTree,
        op: Callable
    ):
        super().__init__()

        self.value = value
        if isinstance(op, ETraceOp):
            op = op.fun
        self.op = op

    def execute(self, x: bst.typing.ArrayLike) -> bst.typing.ArrayLike:
        return self.op(x, self.value)


class FakeElemWiseParamOp(object):
    """
    The fake element-wise parameter state with an associated operator.

    This class corresponds to the :py:class:`ElemWiseParamOp` but does not require to compute gradients.
    It has the same usage interface with :py:class:`ElemWiseParamOp`.

    Args:
        weight: The weight of the ETrace.
    """
    __module__ = 'brainscale'

    def __init__(
        self,
        weight: bst.typing.PyTree,
    ):
        super().__init__()
        self.value = weight

    def execute(self) -> bst.typing.ArrayLike:
        return self.value
