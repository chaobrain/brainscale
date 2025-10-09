# Copyright 2024 BrainX Ecosystem Limited. All Rights Reserved.
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

# -*- coding: utf-8 -*-

from typing import Callable, Union, Sequence, Optional

import brainstate
import brainunit as u
from braintools import init

from brainscale._etrace_concepts import ETraceParam
from brainscale._etrace_operators import MatMulOp, LoraOp, SpMatMulOp
from brainscale._typing import ArrayLike

__all__ = [
    'Linear',
    'SignedWLinear',
    'SparseLinear',
    'LoRA',
]


class Linear(brainstate.nn.Module):
    """
    A Linear layer that performs a linear transformation on the input data.

    This class represents a fully connected linear layer, which applies a linear
    transformation to the incoming data: `y = xW + b`, where `x` is the input,
    `W` is the weight matrix, and `b` is the bias vector.

    Attributes
    ----------
    in_size : Union[int, Sequence[int]]
        The size of the input features.
    out_size : Union[int, Sequence[int]]
        The size of the output features.
    w_mask : Optional[Union[ArrayLike, Callable]]
        An optional mask for the weights.
    weight_op : ETraceParam
        The parameter object that holds the weights and the operation to be
        performed on them.
        
    Methods
    -------
    update(x)
        Applies the linear transformation to the input data.
    """
    __module__ = 'brainscale.nn'

    def __init__(
        self,
        in_size: Union[int, Sequence[int]],
        out_size: Union[int, Sequence[int]],
        w_init: Union[Callable, ArrayLike] = init.KaimingNormal(),
        b_init: Optional[Union[Callable, ArrayLike]] = init.ZeroInit(),
        w_mask: Optional[Union[ArrayLike, Callable]] = None,
        name: Optional[str] = None,
        param_type: type = ETraceParam,
    ):
        """
        Initializes a Linear layer.

        Parameters
        ----------
        in_size : Union[int, Sequence[int]]
            The size of the input features.
        out_size : Union[int, Sequence[int]]
            The size of the output features.
        w_init : Union[Callable, ArrayLike], optional
            The initializer for the weights. Defaults to KaimingNormal.
        b_init : Optional[Union[Callable, ArrayLike]], optional
            The initializer for the bias. Defaults to ZeroInit.
        w_mask : Optional[Union[ArrayLike, Callable]], optional
            An optional mask for the weights.
        name : Optional[str], optional
            The name of the layer.
        param_type : type, optional
            The type of the parameter, defaults to ETraceParam.

        """
        super().__init__(name=name)

        # input and output shape
        self.in_size = in_size
        self.out_size = out_size

        # w_mask
        w_shape = (self.in_size[-1], self.out_size[-1])
        b_shape = (self.out_size[-1],)
        self.w_mask = init.param(w_mask, w_shape)

        # weights
        params = dict(weight=init.param(w_init, w_shape, allow_none=False))
        if b_init is not None:
            params['bias'] = init.param(b_init, b_shape, allow_none=False)

        # weight + op
        self.weight_op = param_type(params, op=MatMulOp(self.w_mask))

    def update(self, x):
        """
        Updates the layer with the given input.

        Parameters
        ----------
        x : ArrayLike
            The input data to be processed by the layer.

        Returns
        -------
        ArrayLike
            The result of the linear transformation applied to the input.
        """
        return self.weight_op.execute(x)


class SignedWLinear(brainstate.nn.Module):
    """
    A Linear layer with signed weights.

    This class represents a linear layer where the weights can be constrained
    to have specific signs. It applies a linear transformation to the input
    data, with the option to mask the weights with a sign matrix.

    Attributes
    ----------
    in_size : Union[int, Sequence[int]]
        The size of the input features.
    out_size : Union[int, Sequence[int]]
        The size of the output features.
    weight_op : ETraceParam
        The parameter object that holds the weights and the operation to be
        performed on them.

    Methods
    -------
    update(x)
        Applies the signed weight linear transformation to the input data.
    """
    __module__ = 'brainscale.nn'

    def __init__(
        self,
        in_size: Union[int, Sequence[int]],
        out_size: Union[int, Sequence[int]],
        w_init: Union[Callable, ArrayLike] = init.KaimingNormal(),
        w_sign: Optional[ArrayLike] = None,
        name: Optional[str] = None,
        param_type: type = ETraceParam,
    ):
        super().__init__(name=name)

        # input and output shape
        self.in_size = in_size
        self.out_size = out_size

        # weights
        w_shape = (self.in_size[-1], self.out_size[-1])
        weight = init.param(w_init, w_shape, allow_none=False)
        op = MatMulOp(
            weight_mask=w_sign,
            weight_fn=u.math.abs,
        )
        self.weight_op = param_type({'weight': weight}, op=op)

    def update(self, x):
        """
        Applies the signed weight linear transformation to the input data.

        Parameters
        ----------
        x : ArrayLike
            The input data to be processed by the layer.

        Returns
        -------
        ArrayLike
            The result of the signed weight linear transformation applied to the input.
        """
        return self.weight_op.execute(x)


class ScaledWSLinear(brainstate.nn.Module):
    """
    Linear Layer with Weight Standardization.

    Applies weight standardization to the weights of the linear layer.

    Parameters
    ----------
    in_size: int, sequence of int
      The input size.
    out_size: int, sequence of int
      The output size.
    w_init: Callable, ArrayLike
      The initializer for the weights.
    b_init: Callable, ArrayLike
      The initializer for the bias.
    w_mask: ArrayLike, Callable
      The optional mask of the weights.
    ws_gain: bool
      Whether to use gain for the weights. The default is True.
    eps: float
      The epsilon value for the weight standardization.
    name: str
      The name of the object.

    """
    __module__ = 'brainscale.nn'

    def __init__(
        self,
        in_size: Union[int, Sequence[int]],
        out_size: Union[int, Sequence[int]],
        w_init: Callable = init.KaimingNormal(),
        b_init: Callable = init.ZeroInit(),
        w_mask: Optional[Union[ArrayLike, Callable]] = None,
        ws_gain: bool = True,
        eps: float = 1e-4,
        name: Optional[str] = None,
        param_type: type = ETraceParam,
    ):
        super().__init__(name=name)

        # input and output shape
        self.in_size = in_size
        self.out_size = out_size

        # w_mask
        self.w_mask = init.param(w_mask, (self.in_size[0], 1))

        # parameters
        self.eps = eps

        # weights
        w_shape = (self.in_size[-1], self.out_size[-1])
        b_shape = (self.out_size[-1],)
        params = dict(weight=init.param(w_init, w_shape, allow_none=False))
        if b_init is not None:
            params['bias'] = init.param(b_init, b_shape, allow_none=False)
        # gain
        if ws_gain:
            params['gain'] = u.math.ones((1, w_shape[-1]), dtype=params['weight'].dtype)

        # operation
        op = MatMulOp(
            weight_mask=self.w_mask,
            weight_fn=lambda w: brainstate.functional.weight_standardization(w['weight'], self.eps,
                                                                             w.get('gain', None)),
            apply_weight_fn_before_mask=True
        )

        # weight + op
        self.weight_op = param_type(params, op=op)

    def update(self, x):
        return self.weight_op.execute(x)


class SparseLinear(brainstate.nn.Module):
    """
    A Linear layer that utilizes a sparse matrix for efficient computation.

    This class represents a linear transformation layer where the weight matrix
    is sparse, allowing for efficient storage and computation. It supports various
    sparse matrix formats such as CSR, CSC, and COO, provided by the `brainunit.sparse`
    module.

    Linear layer with Sparse Matrix (can be ``brainunit.sparse.CSR``,
    ``brainunit.sparse.CSC``, ``brainunit.sparse.COO``, or any other sparse matrix).


    Attributes
    ----------
    in_size : brainstate.typing.Size, optional
        The size of the input features. If provided, it must match the first n-1
        dimensions of the output size.
    out_size : int
        The size of the output features, determined by the last dimension of the
        sparse matrix.
    weight_op : ETraceParam
        The parameter object that holds the sparse weights and the operation to
        be performed on them.

    Methods
    -------
    update(x)
        Applies the sparse linear transformation to the input data.

    Parameters
    ----------
    sparse_mat : u.sparse.SparseMatrix
        The sparse weight matrix to be used in the linear transformation.
    b_init : Optional[Union[Callable, ArrayLike]], optional
        The initializer for the bias. If None, no bias is used.
    in_size : brainstate.typing.Size, optional
        The size of the input features. If provided, it must match the first n-1
        dimensions of the output size.
    name : Optional[str], optional
        The name of the layer.
    param_type : type, optional
        The type of the parameter, defaults to ETraceParam.

    Raises
    ------
    AssertionError
        If the first n-1 dimensions of "in_size" and "out_size" do not match.
        If "sparse_mat" is not an instance of u.sparse.SparseMatrix.

    """
    __module__ = 'brainscale.nn'

    def __init__(
        self,
        sparse_mat: u.sparse.SparseMatrix,
        b_init: Optional[Union[Callable, ArrayLike]] = None,
        in_size: brainstate.typing.Size = None,
        name: Optional[str] = None,
        param_type: type = ETraceParam,
    ):
        """
        Initializes a SparseLinear layer with a sparse weight matrix.

        Parameters
        ----------
        sparse_mat : u.sparse.SparseMatrix
            The sparse weight matrix to be used in the linear transformation.
        b_init : Optional[Union[Callable, ArrayLike]], optional
            The initializer for the bias. If None, no bias is used.
        in_size : brainstate.typing.Size, optional
            The size of the input features. If provided, it must match the
            first n-1 dimensions of the output size.
        name : Optional[str], optional
            The name of the layer.
        param_type : type, optional
            The type of the parameter, defaults to ETraceParam.

        Raises
        ------
        AssertionError
            If the first n-1 dimensions of "in_size" and "out_size" do not match.
            If "sparse_mat" is not an instance of u.sparse.SparseMatrix.
        """
        super().__init__(name=name)

        # input and output shape
        if in_size is not None:
            self.in_size = in_size
        self.out_size = sparse_mat.shape[-1]
        if in_size is not None:
            assert self.in_size[:-1] == self.out_size[:-1], (
                'The first n-1 dimensions of "in_size" '
                'and "out_size" must be the same.'
            )

        # weights
        assert isinstance(sparse_mat, u.sparse.SparseMatrix), '"weight" must be a brainunit.sparse.SparseMatrix.'
        params = dict(weight=sparse_mat.data)
        if b_init is not None:
            params['bias'] = init.param(b_init, self.out_size[-1], allow_none=False)
        op = SpMatMulOp(sparse_mat=sparse_mat)  # x @ sparse matrix
        self.weight_op = param_type(params, op=op)

    def update(self, x):
        """
        Applies the sparse linear transformation to the input data.

        Parameters
        ----------
        x : ArrayLike
            The input data to be processed by the layer.

        Returns
        -------
        ArrayLike
            The result of the sparse linear transformation applied to the input.
        """
        return self.weight_op.execute(x)


class LoRA(brainstate.nn.Module):
    r"""A standalone LoRA layer.

    LoRA (Low-Rank Adaptation) is a technique used to adapt pre-trained models
    by introducing low-rank matrices into the model's weight matrices. This
    allows for efficient fine-tuning of large models with a reduced number of
    parameters.

    $$
        W_\mathrm{L o R A}=W_{\text {orig }}+\frac{\alpha}{r} B A
    $$

    The LoRA layer modifies the original weight matrix $ W $ by adding a
    low-rank component $ \frac{\alpha}{r} B A $, where $ B $ and $ A $
    are learnable matrices of rank $ r $, and $ \alpha $ is a scaling factor.


    Example usage::

        >>> import brainstate as brainstate
        >>> import brainscale
        >>> import jax, jax.numpy as jnp
        >>> layer = brainscale.nn.LoRA(3, 2, 4)
        >>> layer.weight_op.value
        {'lora_a': Array([[ 0.25141352, -0.09826107],
                [ 0.2328382 ,  0.38869813],
                [ 0.27069277,  0.7678282 ]], dtype=float32),
         'lora_b': Array([[-0.8372317 ,  0.21012013, -0.52999765, -0.31939325],
                [ 0.64234126, -0.42980042,  1.2549229 , -0.47134295]],      dtype=float32)}
        >>> # Wrap around existing layer
        >>> linear = brainstate.nn.Linear(3, 4)
        >>> wrapper = brainscale.nn.LoRA(3, 2, 4, base_module=linear)
        >>> assert wrapper.base_module == linear
        >>> y = layer(jnp.ones((16, 3)))
        >>> y.shape
        (16, 4)

    Attributes
    ----------
    in_features : brainstate.typing.Size
        The number of input features.
    lora_rank : int
        The rank of the LoRA dimension.
    out_features : brainstate.typing.Size
        The number of output features.
    alpha : float
        A scaling factor for the LoRA operation.
    base_module : Optional[Callable]
        A base module to call and substitute, if possible.
    weight_op : ETraceParam
        The parameter object that holds the LoRA weights and the operation to
        be performed on them.

    Methods
    -------
    update(x)
        Applies the LoRA transformation to the input data.

    Parameters
    ----------
    in_features : brainstate.typing.Size
        The number of input features.
    lora_rank : int
        The rank of the LoRA dimension.
    out_features : brainstate.typing.Size
        The number of output features.
    alpha : float, optional
        A scaling factor for the LoRA operation, by default 1.
    base_module : Optional[Callable], optional
        A base module to call and substitute, if possible, by default None.
    B_init : Union[Callable, ArrayLike], optional
        Initializer function for the weight matrix B, by default ZeroInit.
    A_init : Union[Callable, ArrayLike], optional
        Initializer function for the weight matrix A, by default LecunNormal.
    param_type : type, optional
        The type of the LoRA parameters, by default ETraceParam.
    """

    def __init__(
        self,
        in_features: brainstate.typing.Size,
        lora_rank: int,
        out_features: brainstate.typing.Size,
        *,
        alpha: float = 1.,
        base_module: Optional[Callable] = None,
        B_init: Union[Callable, ArrayLike] = init.ZeroInit(),
        A_init: Union[Callable, ArrayLike] = init.LecunNormal(),
        param_type: type = ETraceParam,
    ):
        """
        Initializes a LoRA (Low-Rank Adaptation) layer.

        This constructor sets up the LoRA layer with the specified input and output
        features, LoRA rank, and other optional parameters. It initializes the
        weight matrices B and A using the provided initializers and sets up the
        LoRA operation.

        Parameters
        ----------
        in_features : brainstate.typing.Size
            The number of input features.
        lora_rank : int
            The rank of the LoRA dimension.
        out_features : brainstate.typing.Size
            The number of output features.
        alpha : float, optional
            A scaling factor for the LoRA operation, by default 1.
        base_module : Optional[Callable], optional
            A base module to call and substitute, if possible, by default None.
        B_init : Union[Callable, ArrayLike], optional
            Initializer function for the weight matrix B, by default ZeroInit.
        A_init : Union[Callable, ArrayLike], optional
            Initializer function for the weight matrix A, by default LecunNormal.
        param_type : type, optional
            The type of the LoRA parameters, by default ETraceParam.
        """
        super().__init__()

        # input and output shape
        self.in_size = in_features
        self.out_size = out_features
        self.in_features = in_features
        self.out_features = out_features
        self.lora_rank = lora_rank
        self.alpha = alpha

        # others
        if base_module is not None:
            assert callable(base_module), '"base_module" must be callable.'
        self.base_module = base_module

        # weights
        param = dict(
            B=B_init((self.in_size[-1], lora_rank)),
            A=A_init((lora_rank, self.out_size[-1]))
        )
        op = LoraOp(alpha=self.alpha / self.lora_rank)
        self.weight_op = param_type(param, op=op)

    def update(self, x: ArrayLike):
        """
        Applies the LoRA transformation to the input data.

        This method executes the LoRA operation on the input data and, if a base
        module is provided, adds its output to the result of the LoRA operation.

        Parameters
        ----------
        x : ArrayLike
            The input data to be processed by the LoRA layer.

        Returns
        -------
        ArrayLike
            The result of the LoRA transformation applied to the input, optionally
            combined with the output of the base module if it is provided.
        """
        out = self.weight_op.execute(x)
        if self.base_module is not None:
            out += self.base_module(x)
        return out
