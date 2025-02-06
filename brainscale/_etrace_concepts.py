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
from typing import Callable, Optional, Dict, Sequence

import brainstate as bst
import brainunit as u
import jax
import numpy as np
from brainstate.typing import ArrayLike

from ._etrace_operators import ETraceOp, ElemWiseOp
from ._misc import BaseEnum

__all__ = [
    # eligibility trace states
    'ETraceState',  # single hidden state for the etrace-based learning
    'ETraceMultiState',  # multiple hidden state for the etrace-based learning
    'ETraceDictState',  # dictionary of hidden states for the etrace-based learning

    # eligibility trace parameters and operations
    # 'ETraceParam',  # the parameter/weight for the etrace-based learning
    'ETraceParam',  # the parameter and operator for the etrace-based learning, combining ETraceParam and ETraceOp
    'NonTempParam',  # the parameter state with an associated operator without temporal dependent gradient learning

    # element-wise eligibility trace parameters
    'ElemWiseParam',  # the element-wise weight parameter for the etrace-based learning

    # fake parameter state
    'FakeETraceParam',  # the fake parameter state with an associated operator
    'FakeElemWiseParam',  # the fake element-wise parameter state with an associated operator
]

X = bst.typing.ArrayLike
W = bst.typing.PyTree
Y = bst.typing.ArrayLike


class ETraceGrad(BaseEnum):
    """
    The Gradient Type for the Eligibility Trace.

    This defines how the weight gradient is computed in the eligibility trace-based learning.

    - `full`: The full eligibility trace gradient is computed.
    - `approx`: The approximated eligibility trace gradient is computed.
    - `adaptive`: The adaptive eligibility trace gradient is computed.

    """
    full = 'full'
    approx = 'approx'
    adaptive = 'adaptive'


class ETraceState(bst.ShortTermState):
    """
    The Hidden State for Eligibility Trace-based Learning.

    .. note::

        Currently, the hidden state only supports `jax.Array` or `brainunit.Quantity`.
        This means that each instance of :py:class:`ETraceState` should define
        single hidden variable.

        If you want to define multiple hidden variables within a single instance of
        :py:class:`ETraceState`, you can try :py:class:`ETraceMultiState` or
        :py:class:`ETraceDictState` instead.

    Args:
        value: The value of the hidden state.
               Currently only support a `jax.Array` or `brainunit.Quantity`.
        name: The name of the hidden state.
    """
    __module__ = 'brainscale'

    value: bst.typing.ArrayLike

    def __init__(
        self,
        value: bst.typing.PyTree,
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


class ETraceMultiState(ETraceState):
    """
    A group of multiple hidden states for eligibility trace-based learning.

    This class is used to define multiple hidden states within a single instance
    of :py:class:`ETraceState`. Normally, you should define multiple instances
    of :py:class:`ETraceState` to represent multiple hidden states. But
    :py:class:`ETraceMultiState` let your define multiple hidden states within
    a single instance.

    Args:
        value: The values of the hidden states.
        name: The name of the hidden states.
    """

    __module__ = 'brainscale'
    value: bst.typing.ArrayLike

    def __init__(
        self,
        value: bst.typing.PyTree,
        names: Optional[Sequence[str]] = None
    ):
        self.names = names
        self.name2index = (
            None
            if names is None else
            {name: i for i, name in enumerate(names)}
        )
        self._check_value(value)
        bst.ShortTermState.__init__(self, value)

    def _check_value(self, value):
        if not isinstance(value, (np.ndarray, jax.Array, u.Quantity)):
            raise TypeError(
                f'Currently, {ETraceMultiState.__name__} only supports '
                f'numpy.ndarray, jax.Array or brainunit.Quantity. '
                f'But we got {type(value)}.'
            )
        if value.ndim < 2:
            raise ValueError(
                f'Currently, {ETraceMultiState.__name__} only supports '
                f'hidden states with more than 2 dimensions, where the last '
                f'dimension is the number of state size and the other dimensions '
                f'are the hidden shape. '
                f'But we got {value.ndim} dimensions.'
            )

    def get_value(self, item: int | str) -> bst.typing.ArrayLike:
        """
        Get the value of the hidden state with the item.

        Args:
            item: int or str. The index of the hidden state.
                - If int, the index of the hidden state.
                - If str, the name of the hidden state.
        Returns:
            The value of the hidden state.
        """
        if isinstance(item, int):
            assert item < self.value.shape[-1], (f'Index {item} out of range. '
                                                 f'The maximum index is {self.value.shape[-1] - 1}.')
            return self.value[..., item]
        elif isinstance(item, str):
            assert self.names is not None, (f'Hidden state names are not defined. '
                                            f'Please define the names when initializing '
                                            f'{ETraceMultiState.__name__}.')
            assert item in self.name2index, (f'Hidden state name {item} not found. '
                                             f'Please check the hidden state names.')
            index = self.name2index[item]
            return self.value[..., index]
        else:
            raise TypeError(
                f'Currently, {ETraceMultiState.__name__} only supports '
                f'int or str for getting the hidden state. '
                f'But we got {type(item)}.'
            )


class ETraceDictState(ETraceMultiState):
    """
    A dictionary of multiple hidden states for eligibility trace-based learning.

    .. note::

        This value in this state class behaves likes a dictionary of hidden states.
        However, the state is actually stored as a single dimensionless array.

    Args:
        value: The values of the hidden states.
    """

    __module__ = 'brainscale'
    value: Dict[str, bst.typing.ArrayLike]

    def __init__(
        self,
        value: Dict[str, bst.typing.ArrayLike],
    ):
        self._check_value(value)
        self.name2unit = {
            k: u.get_unit(v)
            for k, v in value.items()
        }
        self.name2index = {
            k: i
            for i, k in enumerate(value.keys())
        }
        value = u.math.stack([u.get_magnitude(v) for v in value.values()], axis=-1)
        bst.ShortTermState.__init__(self, value)

    def _check_value(self, value: dict):
        if not isinstance(value, dict):
            raise TypeError(
                f'Currently, {ETraceDictState.__name__} only supports '
                f'dictionary of hidden states. '
                f'But we got {type(value)}.'
            )
        shapes = []
        for k, v in value.items():
            if not isinstance(v, (np.ndarray, jax.Array, u.Quantity)):
                raise TypeError(
                    f'Currently, {ETraceDictState.__name__} only supports '
                    f'numpy.ndarray, jax.Array or brainunit.Quantity. '
                    f'But we got {type(v)} for key {k}.'
                )
            shapes.append(v.shape)
        if len(set(shapes)) > 1:
            raise ValueError(
                f'Currently, {ETraceDictState.__name__} only supports '
                f'hidden states with the same shape. '
                f'But we got {dict(k=v.shape for k, v in value.items())}.'
            )

    def get_value(self, item: str | int) -> bst.typing.ArrayLike:
        """
        Get the value of the hidden state with the key.

        Args:
            item: The key of the hidden state.
                - If int, the index of the hidden state.
                - If str, the name of the hidden state.
        """
        if isinstance(item, int):
            assert item < self._value.shape[-1], (f'Index {item} out of range. '
                                                  f'The maximum index is {self._value.shape[-1] - 1}.')
            return self._value[..., item]
        elif isinstance(item, str):
            assert item in self.name2index, (f'Hidden state name {item} not found. '
                                             f'Please check the hidden state names.')
            index = self.name2index[item]
            return self._value[..., index]
        else:
            raise TypeError(
                f'Currently, {self.__class__.__name__} only supports '
                f'int or str for getting the hidden state. '
                f'But we got {type(item)}.'
            )

    def _read_value(self) -> Dict[str, ArrayLike]:
        res = dict()
        for i, k in enumerate(self.name2unit.keys()):
            v = self._value[..., i]
            res[k] = u.maybe_decimal(u.Quantity(v, unit=self.name2unit[k]))
        return res

    def _write_value(self, value: Dict[str, ArrayLike]) -> None:
        self._check_value(value)
        res = []
        for k, v in value.items():
            res.append(u.get_magnitude(u.Quantity(v).to(self.name2unit[k])))
        value = u.math.stack(res, axis=-1)
        self._value = value


# class ETraceParam(bst.ParamState):
#     """
#     The Eligibility Trace Weight Parameter.
#
#     Args:
#         value: The value of the weight. Can be a PyTree.
#         name: The name of the weight.
#     """
#     __module__ = 'brainscale'
#
#     def __init__(
#         self,
#         value: bst.typing.PyTree,
#         name: Optional[str] = None
#     ):
#         super().__init__(value, name=name)
#
#         self.is_etrace = False


class ETraceParam(bst.ParamState):
    """
    The Eligibility Trace Weight and its Associated Operator.

    .. note::

        Although one weight is defined as :py:class:`ETraceParam`,
        whether eligibility traces are used for training with temporal
        dependencies depends on the final compilation result of the
        compiler regarding this parameter. If no hidden states are
        found to associate this parameter, the training based on
        eligibility traces will not be performed.
        Then, this parameter will perform the same as :py:class:`NonTempParam`.

    Args:
        weight: The weight of the ETrace.
        op: The operator for the ETrace. See :py:class:`ETraceOp`.
        grad: The gradient type for the ETrace. Default is `adaptive`.
        is_diagonal: Whether the operator is diagonal. Default is `None`.
        name: The name of the weight-operator.
    """
    __module__ = 'brainscale'
    op: ETraceOp  # operator
    is_etrace: bool

    def __init__(
        self,
        weight: bst.typing.PyTree,
        op: ETraceOp | Callable[[X, W], Y],
        grad: Optional[str | Enum] = None,
        is_diagonal: bool = None,
        name: Optional[str] = None
    ):
        # weight value
        super().__init__(weight, name=name)

        # gradient
        if grad is None:
            grad = 'adaptive'
        self.gradient = ETraceGrad.get(grad)

        # operation
        if not isinstance(op, ETraceOp):
            op = ETraceOp(op, is_diagonal=False if is_diagonal is None else is_diagonal)
        self.op = op
        if is_diagonal is not None:
            self.op.is_diagonal = is_diagonal

        # check if the operator is not an eligibility trace
        self.is_etrace = True

    def execute(self, x: X) -> Y:
        """
        Execute the operator with the input.
        """
        return self.op(x, self.value)


class ElemWiseParam(ETraceParam):
    r"""
    The Element-wise Eligibility Trace Weight and its Associated Operator.

    .. note::

        The ``element-wise`` is called with the correspondence to the hidden state.
        That means the operator performs element-wise operations with the hidden state.

    It supports all element-wise operations for the eligibility trace-based learning.
    For example, if the parameter weight has the shape with the same as the hidden state,

        $$
        I = \theta_1 * h_1
        $$

    where $\theta_1 \in \mathbb{R}^H$ is the weight and $h_1 \in \mathbb{R}^H$ is the
    hidden state. The element-wise operation is defined as:

    .. code-block:: python

       op = ElemWiseParam(weight, op=lambda w: w)

    If the parameter weight is a scalar,

        $$
        I = \theta * h
        $$

    where $\theta \in \mathbb{R}$ is the weight and $h \in \mathbb{R}^H$ is the hidden state.
    Then the element-wise operation can be defined as:

    .. code-block:: python

         h = 100  # hidden size
         op = ElemWiseParam(weight, op=lambda w: w * jax.numpy.ones(h))

    Other element-wise operations can be defined in the same way.

    Args:
        weight: The weight of the ETrace.
        op: The operator for the ETrace. See :py:class:`ElemWiseOp`.
        name: The name of the weight-operator.
    """
    __module__ = 'brainscale'

    def __init__(
        self,
        weight: bst.typing.PyTree,
        op: ElemWiseOp | Callable[[W], Y] = (lambda w: w),
        name: Optional[str] = None,
    ):
        super().__init__(
            weight,
            op=ElemWiseOp(op),
            grad=ETraceGrad.full,
            is_diagonal=True,
            name=name
        )

    def execute(self) -> Y:
        return self.op(self.value)


class NonTempParam(bst.ParamState):
    r"""
    The Parameter State with an Associated Operator with no temporal dependent gradient learning.

    This class behaves the same as :py:class:`ETraceParam`, but will not build the
    eligibility trace graph when using online learning. Therefore, in a sequence
    learning task, the weight gradient can only be computed with the spatial gradients.
    That is,

        $$
        \nabla \theta = \sum_t \partial L^t / \partial \theta^t
        $$

    Instead, the gradient of the weight $\theta$ which is labeled as :py:class:`ETraceParam` is
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


class FakeETraceParam(object):
    """
    The Parameter State with an Associated Operator that does not require to compute gradients.

    This class corresponds to the :py:class:`NonTempParam` but does not require to compute gradients.
    It has the same usage interface with :py:class:`NonTempParam`.

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


class FakeElemWiseParam(object):
    """
    The fake element-wise parameter state with an associated operator.

    This class corresponds to the :py:class:`ElemWiseParam` but does not require to compute gradients.
    It has the same usage interface with :py:class:`ElemWiseParam`. For usage, please see
    :py:class:`ElemWiseParam`.

    Args:
        weight: The weight of the ETrace.
        op: The operator for the ETrace. See :py:class:`ElemWiseOp`.
        name: The name of the weight-operator.
    """
    __module__ = 'brainscale'

    def __init__(
        self,
        weight: bst.typing.PyTree,
        op: ElemWiseOp | Callable[[W], Y] = (lambda w: w),
        name: Optional[str] = None,
    ):
        super().__init__()
        self.value = weight
        self.op = op
        self.name = name

    def execute(self) -> bst.typing.ArrayLike:
        return self.op(self.value)
