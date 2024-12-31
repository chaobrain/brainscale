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
# ==============================================================================
# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Callable, Union

import brainstate as bst
import brainunit as u
from brainstate import init, functional, nn

from brainscale._etrace_concepts import ETraceState, ETraceParamOp, ElementWiseParamOp
from brainscale._typing import ArrayLike
from ._linear import Linear

__all__ = [
    'ValinaRNNCell',
    'GRUCell',
    'MGUCell',
    'LSTMCell',
    'URLSTMCell',
    'MinimalRNNCell',
    'MiniGRU',
    'MiniLSTM',
    'LRUCell',
]


class ValinaRNNCell(nn.RNNCell):
    """
    Vanilla RNN cell.
  
    Args:
      in_size: bst.typing.Size. The number of input units.
      out_size: bst.typing.Size. The number of hidden units.
      state_init: callable, ArrayLike. The state initializer.
      w_init: callable, ArrayLike. The input weight initializer.
      b_init: optional, callable, ArrayLike. The bias weight initializer.
      activation: str, callable. The activation function. It can be a string or a callable function.
      name: optional, str. The name of the module.
    """
    __module__ = 'brainscale.nn'

    def __init__(
        self,
        in_size: bst.typing.Size,
        out_size: bst.typing.Size,
        state_init: Union[ArrayLike, Callable] = init.ZeroInit(),
        w_init: Union[ArrayLike, Callable] = init.XavierNormal(),
        b_init: Union[ArrayLike, Callable] = init.ZeroInit(),
        activation: str | Callable = 'relu',
        name: str = None,
    ):
        super().__init__(name=name)

        # parameters
        self._state_initializer = state_init
        self.out_size = out_size
        self.in_size = in_size

        # activation function
        if isinstance(activation, str):
            self.activation = getattr(functional, activation)
        else:
            assert callable(activation), "The activation function should be a string or a callable function. "
            self.activation = activation

        # weights
        self.W = Linear(self.in_size[-1] + self.out_size[-1], self.out_size[-1], w_init=w_init, b_init=b_init)

    def init_state(self, batch_size: int = None, **kwargs):
        self.h = ETraceState(init.param(self._state_initializer, self.out_size, batch_size))

    def reset_state(self, batch_size: int = None, **kwargs):
        self.h.value = init.param(self._state_initializer, self.out_size, batch_size)

    def update(self, x):
        xh = u.math.concatenate([x, self.h.value], axis=-1)
        self.h.value = self.activation(self.W(xh))
        return self.h.value


class GRUCell(nn.RNNCell):
    r"""
    Gated Recurrent Unit (GRU) cell, implemented as in
    `Learning Phrase Representations using RNN Encoder-Decoder for
    Statistical Machine Translation <https://arxiv.org/abs/1406.1078>`_.


    Args:
      in_size: bst.typing.Size. The number of input units.
      out_size: bst.typing.Size. The number of hidden units.
      state_init: callable, ArrayLike. The state initializer.
      w_init: callable, ArrayLike. The input weight initializer.
      b_init: optional, callable, ArrayLike. The bias weight initializer.
      activation: str, callable. The activation function. It can be a string or a callable function.
      name: optional, str. The name of the module.
    """
    __module__ = 'brainscale.nn'

    def __init__(
        self,
        in_size: bst.typing.Size,
        out_size: bst.typing.Size,
        w_init: Union[ArrayLike, Callable] = init.Orthogonal(),
        b_init: Union[ArrayLike, Callable] = init.ZeroInit(),
        state_init: Union[ArrayLike, Callable] = init.ZeroInit(),
        activation: str | Callable = 'tanh',
        name: str = None,
    ):
        super().__init__(name=name)

        # parameters
        self._state_initializer = state_init
        self.out_size = out_size
        self.in_size = in_size

        # activation function
        if isinstance(activation, str):
            self.activation = getattr(functional, activation)
        else:
            assert callable(activation), "The activation function should be a string or a callable function. "
            self.activation = activation

        # weights
        self.Wz = Linear(self.in_size[-1] + self.out_size[-1], self.out_size[-1], w_init=w_init, b_init=b_init)
        self.Wr = Linear(self.in_size[-1] + self.out_size[-1], self.out_size[-1], w_init=w_init, b_init=b_init)
        self.Wh = Linear(self.in_size[-1] + self.out_size[-1], self.out_size[-1], w_init=w_init, b_init=b_init)

    def init_state(self, batch_size: int = None, **kwargs):
        self.h = ETraceState(init.param(self._state_initializer, self.out_size, batch_size))

    def reset_state(self, batch_size: int = None, **kwargs):
        self.h.value = init.param(self._state_initializer, self.out_size, batch_size)

    def update(self, x):
        old_h = self.h.value
        xh = u.math.concatenate([x, old_h], axis=-1)
        z = functional.sigmoid(self.Wz(xh))
        r = functional.sigmoid(self.Wr(xh))
        rh = r * old_h
        h = self.activation(self.Wh(u.math.concatenate([x, rh], axis=-1)))
        h = (1 - z) * old_h + z * h
        self.h.value = h
        return h


class CFNCell(nn.RNNCell):
    r"""
    Chaos Free Networks (CFN) cell, implemented as in
    `A recurrent neural network without chaos <https://arxiv.org/abs/1612.06212>`_.

    Args:
      in_size: bst.typing.Size. The number of input units.
      out_size: bst.typing.Size. The number of hidden units.
      state_init: callable, ArrayLike. The state initializer.
      w_init: callable, ArrayLike. The input weight initializer.
      b_init: optional, callable, ArrayLike. The bias weight initializer.
      activation: str, callable. The activation function. It can be a string or a callable function.
      name: optional, str. The name of the module.
    """
    __module__ = 'brainscale.nn'

    def __init__(
        self,
        in_size: bst.typing.Size,
        out_size: bst.typing.Size,
        w_init: Union[ArrayLike, Callable] = init.Orthogonal(),
        b_init: Union[ArrayLike, Callable] = init.ZeroInit(),
        state_init: Union[ArrayLike, Callable] = init.ZeroInit(),
        activation: str | Callable = 'tanh',
        name: str = None,
    ):
        super().__init__(name=name)

        # parameters
        self._state_initializer = state_init
        self.out_size = out_size
        self.in_size = in_size

        # activation function
        if isinstance(activation, str):
            self.activation = getattr(functional, activation)
        else:
            assert callable(activation), "The activation function should be a string or a callable function. "
            self.activation = activation

        # weights
        self.Wf = Linear(self.in_size[-1] + self.out_size[-1], self.out_size[-1], w_init=w_init, b_init=b_init)
        self.Wi = Linear(self.in_size[-1] + self.out_size[-1], self.out_size[-1], w_init=w_init, b_init=b_init)
        self.Wh = Linear(self.out_size[-1], self.out_size[-1], w_init=w_init, b_init=b_init)

    def init_state(self, batch_size: int = None, **kwargs):
        self.h = ETraceState(init.param(self._state_initializer, self.out_size, batch_size))

    def reset_state(self, batch_size: int = None, **kwargs):
        self.h.value = init.param(self._state_initializer, self.out_size, batch_size)

    def update(self, x):
        old_h = self.h.value
        xh = u.math.concatenate([x, old_h], axis=-1)
        f = functional.sigmoid(self.Wf(xh))
        i = functional.sigmoid(self.Wi(xh))
        h = f * self.activation(old_h) + i * self.activation(self.Wh(x))
        self.h.value = h
        return h


class MGUCell(nn.RNNCell):
    r"""
    Minimal Gated Recurrent Unit (MGU) cell, implemented as in
    `Minimal Gated Unit for Recurrent Neural Networks <https://arxiv.org/abs/1603.09420>`_.
  
    .. math::
  
       \begin{aligned}
       f_{t}&=\sigma (W_{f}x_{t}+U_{f}h_{t-1}+b_{f})\\
       {\hat {h}}_{t}&=\phi (W_{h}x_{t}+U_{h}(f_{t}\odot h_{t-1})+b_{h})\\
       h_{t}&=(1-f_{t})\odot h_{t-1}+f_{t}\odot {\hat {h}}_{t}
       \end{aligned}
  
    where:
  
    - :math:`x_{t}`: input vector
    - :math:`h_{t}`: output vector
    - :math:`{\hat {h}}_{t}`: candidate activation vector
    - :math:`f_{t}`: forget vector
    - :math:`W, U, b`: parameter matrices and vector
  
    Args:
      in_size: bst.typing.Size. The number of input units.
      out_size: bst.typing.Size. The number of hidden units.
      state_init: callable, ArrayLike. The state initializer.
      w_init: callable, ArrayLike. The input weight initializer.
      b_init: optional, callable, ArrayLike. The bias weight initializer.
      activation: str, callable. The activation function. It can be a string or a callable function.
      name: optional, str. The name of the module.
    """
    __module__ = 'brainscale.nn'

    def __init__(
        self,
        in_size: bst.typing.Size,
        out_size: bst.typing.Size,
        w_init: Union[ArrayLike, Callable] = init.Orthogonal(),
        b_init: Union[ArrayLike, Callable] = init.ZeroInit(),
        state_init: Union[ArrayLike, Callable] = init.ZeroInit(),
        activation: str | Callable = 'tanh',
        name: str = None,
    ):
        super().__init__(name=name)

        # parameters
        self._state_initializer = state_init
        self.out_size = out_size
        self.in_size = in_size

        # activation function
        if isinstance(activation, str):
            self.activation = getattr(functional, activation)
        else:
            assert callable(activation), "The activation function should be a string or a callable function. "
            self.activation = activation

        # weights
        self.Wf = Linear(self.in_size[-1] + self.out_size[-1], self.out_size[-1], w_init=w_init, b_init=b_init)
        self.Wh = Linear(self.in_size[-1] + self.out_size[-1], self.out_size[-1], w_init=w_init, b_init=b_init)

    def init_state(self, batch_size: int = None, **kwargs):
        self.h = ETraceState(init.param(self._state_initializer, self.out_size, batch_size))

    def reset_state(self, batch_size: int = None, **kwargs):
        self.h.value = init.param(self._state_initializer, self.out_size, batch_size)

    def update(self, x):
        old_h = self.h.value
        xh = u.math.concatenate([x, old_h], axis=-1)
        f = functional.sigmoid(self.Wf(xh))
        fh = f * old_h
        h = self.activation(self.Wh(u.math.concatenate([x, fh], axis=-1)))
        self.h.value = (1 - f) * self.h.value + f * h
        return self.h.value


class LSTMCell(nn.RNNCell):
    r"""Long short-term memory (LSTM) RNN core.
  
    The implementation is based on (zaremba, et al., 2014) [1]_. Given
    :math:`x_t` and the previous state :math:`(h_{t-1}, c_{t-1})` the core
    computes
  
    .. math::
  
       \begin{array}{ll}
       i_t = \sigma(W_{ii} x_t + W_{hi} h_{t-1} + b_i) \\
       f_t = \sigma(W_{if} x_t + W_{hf} h_{t-1} + b_f) \\
       g_t = \tanh(W_{ig} x_t + W_{hg} h_{t-1} + b_g) \\
       o_t = \sigma(W_{io} x_t + W_{ho} h_{t-1} + b_o) \\
       c_t = f_t c_{t-1} + i_t g_t \\
       h_t = o_t \tanh(c_t)
       \end{array}
  
    where :math:`i_t`, :math:`f_t`, :math:`o_t` are input, forget and
    output gate activations, and :math:`g_t` is a vector of cell updates.
  
    The output is equal to the new hidden, :math:`h_t`.
  
    Notes
    -----
  
    Forget gate initialization: Following (Jozefowicz, et al., 2015) [2]_ we add 1.0
    to :math:`b_f` after initialization in order to reduce the scale of forgetting in
    the beginning of the training.
  
  
    Parameters
    ----------
    in_size: bst.typing.Size
      The dimension of the input vector
    out_size: bst.typing.Size
      The number of hidden unit in the node.
    state_init: callable, ArrayLike
      The state initializer.
    w_init: callable, ArrayLike
      The input weight initializer.
    b_init: optional, callable, ArrayLike
      The bias weight initializer.
    activation: str, callable
      The activation function. It can be a string or a callable function.
  
    References
    ----------
  
    .. [1] Zaremba, Wojciech, Ilya Sutskever, and Oriol Vinyals. "Recurrent neural
           network regularization." arXiv preprint arXiv:1409.2329 (2014).
    .. [2] Jozefowicz, Rafal, Wojciech Zaremba, and Ilya Sutskever. "An empirical
           exploration of recurrent network architectures." In International conference
           on machine learning, pp. 2342-2350. PMLR, 2015.
    """
    __module__ = 'brainscale.nn'

    def __init__(
        self,
        in_size: bst.typing.Size,
        out_size: bst.typing.Size,
        w_init: Union[ArrayLike, Callable] = init.XavierNormal(),
        b_init: Union[ArrayLike, Callable] = init.ZeroInit(),
        state_init: Union[ArrayLike, Callable] = init.ZeroInit(),
        activation: str | Callable = 'tanh',
        name: str = None,
    ):
        super().__init__(name=name)

        # parameters
        self.out_size = out_size
        self.in_size = in_size

        # initializers
        self._state_initializer = state_init

        # activation function
        if isinstance(activation, str):
            self.activation = getattr(functional, activation)
        else:
            assert callable(activation), "The activation function should be a string or a callable function. "
            self.activation = activation

        # weights
        self.Wi = Linear(self.in_size[-1] + self.out_size[-1], self.out_size[-1], w_init=w_init, b_init=b_init)
        self.Wg = Linear(self.in_size[-1] + self.out_size[-1], self.out_size[-1], w_init=w_init, b_init=b_init)
        self.Wf = Linear(self.in_size[-1] + self.out_size[-1], self.out_size[-1], w_init=w_init, b_init=b_init)
        self.Wo = Linear(self.in_size[-1] + self.out_size[-1], self.out_size[-1], w_init=w_init, b_init=b_init)

    def init_state(self, batch_size: int = None, **kwargs):
        self.c = ETraceState(init.param(self._state_initializer, self.out_size, batch_size))
        self.h = ETraceState(init.param(self._state_initializer, self.out_size, batch_size))

    def reset_state(self, batch_size: int = None, **kwargs):
        self.c.value = init.param(self._state_initializer, self.out_size, batch_size)
        self.h.value = init.param(self._state_initializer, self.out_size, batch_size)

    def update(self, x):
        h, c = self.h.value, self.c.value
        xh = u.math.concatenate([x, h], axis=-1)
        i = self.Wi(xh)
        g = self.Wg(xh)
        f = self.Wf(xh)
        o = self.Wo(xh)
        c = functional.sigmoid(f + 1.) * c + functional.sigmoid(i) * self.activation(g)
        h = functional.sigmoid(o) * self.activation(c)
        self.h.value = h
        self.c.value = c
        return h


class URLSTMCell(nn.RNNCell):
    __module__ = 'brainscale.nn'

    def __init__(
        self,
        in_size: bst.typing.Size,
        out_size: bst.typing.Size,
        w_init: Union[ArrayLike, Callable] = init.XavierNormal(),
        state_init: Union[ArrayLike, Callable] = init.ZeroInit(),
        activation: str | Callable = 'tanh',
        name: str = None,
    ):
        super().__init__(name=name)

        # parameters
        self.out_size = out_size
        self.in_size = in_size

        # initializers
        self._state_initializer = state_init

        # activation function
        if isinstance(activation, str):
            self.activation = getattr(functional, activation)
        else:
            assert callable(activation), "The activation function should be a string or a callable function. "
            self.activation = activation

        # weights
        self.Wu = Linear(self.in_size[-1] + self.out_size[-1], self.out_size[-1], w_init=w_init, b_init=None)
        self.Wf = Linear(self.in_size[-1] + self.out_size[-1], self.out_size[-1], w_init=w_init, b_init=None)
        self.Wr = Linear(self.in_size[-1] + self.out_size[-1], self.out_size[-1], w_init=w_init, b_init=None)
        self.Wo = Linear(self.in_size[-1] + self.out_size[-1], self.out_size[-1], w_init=w_init, b_init=None)
        self.bias = ETraceParamOp(self._forget_bias(), op=u.math.add, grad='full')

    def _forget_bias(self):
        u = bst.random.uniform(1 / self.out_size[-1], 1 - 1 / self.out_size[1], (self.out_size[-1],))
        return -u.math.log(1 / u - 1)

    def init_state(self, batch_size: int = None, **kwargs):
        self.c = ETraceState(init.param(self._state_initializer, self.out_size, batch_size))
        self.h = ETraceState(init.param(self._state_initializer, self.out_size, batch_size))

    def reset_state(self, batch_size: int = None, **kwargs):
        self.c.value = init.param(self._state_initializer, self.out_size, batch_size)
        self.h.value = init.param(self._state_initializer, self.out_size, batch_size)

    def update(self, x: ArrayLike) -> ArrayLike:
        h, c = self.h.value, self.c.value
        xh = u.math.concatenate([x, h], axis=-1)
        f = self.Wf(xh)
        r = self.Wr(xh)
        u_ = self.Wu(xh)
        o = self.Wo(xh)
        f_ = functional.sigmoid(self.bias.execute(f))
        r_ = functional.sigmoid(-self.bias.execute(-r))
        g = 2 * r_ * f_ + (1 - 2 * r_) * f_ ** 2
        next_cell = g * c + (1 - g) * self.activation(u_)
        next_hidden = functional.sigmoid(o) * self.activation(next_cell)
        self.h.value = next_hidden
        self.c.value = next_cell
        return next_hidden


class MinimalRNNCell(nn.RNNCell):
    r"""
    Minimal RNN Cell, implemented as in
    `MinimalRNN: Toward More Interpretable and Trainable Recurrent Neural Networks <https://arxiv.org/abs/1711.06788>`_
  
    Model
    -----
  
    At each step $t$, the model first maps its input $\mathbf{x}_t$ to a
    latent space through
    $$\mathbf{z}_t=\Phi(\mathbf{x}_t)$$
    $\Phi(\cdot)$ here can be any highly flexible functions such  as neural networks.
    Default, we take $\Phi(\cdot)$ as a fully connected layer with tanh activation. That
    is,  $\Phi ( \mathbf{x} _t) = \tanh ( \mathbf{W} _x\mathbf{x} _t+ \mathbf{b} _z) .$
  
    Given the latent representation $\mathbf{z}_t$ of the input, MinimalRNN then updates its states simply as:
  
    $$\mathbf{h}_t=\mathbf{u}_t\odot\mathbf{h}_{t-1}+(\mathbf{1}-\mathbf{u}_t)\odot\mathbf{z}_t$$
  
    where $\mathbf{u}_t=\sigma(\mathbf{U}_h\mathbf{h}_{t-1}+\mathbf{U}_z\mathbf{z}_t+\mathbf{b}_u)$ is the update
    gate.
  
    Parameters
    ----------
    in_size: bst.typing.Size
      The number of input units.
    out_size: bst.typing.Size
      The number of hidden units.
    w_init: callable, ArrayLike
      The input weight initializer.
    b_init: callable, ArrayLike
      The bias weight initializer.
    state_init: callable, ArrayLike
      The state initializer.
    phi: callable
      The input activation function.
    name: optional, str
      The name of the module.
  
    """
    __module__ = 'brainscale.nn'

    def __init__(
        self,
        in_size: bst.typing.Size,
        out_size: bst.typing.Size,
        w_init: Union[ArrayLike, Callable] = init.Orthogonal(),
        b_init: Union[ArrayLike, Callable] = init.ZeroInit(),
        state_init: Union[ArrayLike, Callable] = init.ZeroInit(),
        phi: Callable = None,
        name: str = None,
    ):
        super().__init__(name=name)

        # parameters
        self._state_initializer = state_init
        self.out_size = out_size
        self.in_size = in_size

        # functions
        if phi is None:
            phi = Linear(self.in_size[-1], self.out_size[-1], w_init=w_init, b_init=b_init)
        assert callable(phi), f"The phi function should be a callable function. But got {phi}"
        self.phi = phi

        # weights
        self.W_u = Linear(self.out_size[-1] * 2, self.out_size[-1], w_init=w_init, b_init=b_init)

    def init_state(self, batch_size: int = None, **kwargs):
        self.h = ETraceState(init.param(self._state_initializer, self.out_size, batch_size))

    def reset_state(self, batch_size: int = None, **kwargs):
        self.h.value = init.param(self._state_initializer, self.out_size, batch_size)

    def update(self, x):
        z = self.phi(x)
        f = functional.sigmoid(self.W_u(u.math.concatenate([z, self.h.value], axis=-1)))
        self.h.value = f * self.h.value + (1 - f) * z
        return self.h.value


class MiniGRU(nn.RNNCell):
    r"""
    Minimal RNN Cell, implemented as in
    `MinimalRNN: Toward More Interpretable and Trainable Recurrent Neural Networks <https://arxiv.org/abs/1711.06788>`_

    Model
    -----

    At each step $t$, the model first maps its input $\mathbf{x}_t$ to a
    latent space through
    $$\mathbf{z}_t=\Phi(\mathbf{x}_t)$$
    $\Phi(\cdot)$ here can be any highly flexible functions such  as neural networks.
    Default, we take $\Phi(\cdot)$ as a fully connected layer with tanh activation. That
    is,  $\Phi ( \mathbf{x} _t) = \tanh ( \mathbf{W} _x\mathbf{x} _t+ \mathbf{b} _z) .$

    Given the latent representation $\mathbf{z}_t$ of the input, MinimalRNN then updates its states simply as:

    $$\mathbf{h}_t=\mathbf{u}_t\odot\mathbf{h}_{t-1}+(\mathbf{1}-\mathbf{u}_t)\odot\mathbf{z}_t$$

    where $\mathbf{u}_t=\sigma(\mathbf{U}_h\mathbf{h}_{t-1}+\mathbf{U}_z\mathbf{z}_t+\mathbf{b}_u)$ is the update
    gate.


    Parameters
    ----------
    in_size: bst.typing.Size
        The number of input units.
    out_size: bst.typing.Size
        The number of hidden units.
    w_init: callable, ArrayLike
        The input weight initializer.
    b_init: callable, ArrayLike
        The bias weight initializer.
    state_init: callable, ArrayLike
        The state initializer.
    name: optional, str
        The name of the module.

    """
    __module__ = 'brainscale.nn'

    def __init__(
        self,
        in_size: bst.typing.Size,
        out_size: bst.typing.Size,
        w_init: Union[ArrayLike, Callable] = init.Orthogonal(),
        b_init: Union[ArrayLike, Callable] = init.ZeroInit(),
        state_init: Union[ArrayLike, Callable] = init.ZeroInit(),
        name: str = None,
    ):
        super().__init__(name=name)

        # parameters
        self._state_initializer = state_init
        self.out_size = out_size
        self.in_size = in_size

        # functions
        self.W_x = Linear(self.in_size[-1], self.out_size[-1], w_init=w_init, b_init=b_init)

        # weights
        self.W_z = Linear(self.in_size[-1] + self.out_size[-1], self.out_size[-1], w_init=w_init, b_init=b_init)

    def init_state(self, batch_size: int = None, **kwargs):
        self.h = ETraceState(init.param(self._state_initializer, self.out_size, batch_size))

    def reset_state(self, batch_size: int = None, **kwargs):
        self.h.value = init.param(self._state_initializer, self.out_size, batch_size)

    def update(self, x):
        z = functional.sigmoid(self.W_z(u.math.concatenate([x, self.h.value], axis=-1)))
        self.h.value = (1 - z) * self.h.value + z * self.W_x(x)
        return self.h.value


class MiniLSTM(nn.RNNCell):
    r"""
    Minimal RNN Cell, implemented as in
    `MinimalRNN: Toward More Interpretable and Trainable Recurrent Neural Networks <https://arxiv.org/abs/1711.06788>`_

    Model
    -----

    At each step $t$, the model first maps its input $\mathbf{x}_t$ to a
    latent space through
    $$\mathbf{z}_t=\Phi(\mathbf{x}_t)$$
    $\Phi(\cdot)$ here can be any highly flexible functions such  as neural networks.
    Default, we take $\Phi(\cdot)$ as a fully connected layer with tanh activation. That
    is,  $\Phi ( \mathbf{x} _t) = \tanh ( \mathbf{W} _x\mathbf{x} _t+ \mathbf{b} _z) .$

    Given the latent representation $\mathbf{z}_t$ of the input, MinimalRNN then updates its states simply as:

    $$\mathbf{h}_t=\mathbf{u}_t\odot\mathbf{h}_{t-1}+(\mathbf{1}-\mathbf{u}_t)\odot\mathbf{z}_t$$

    where $\mathbf{u}_t=\sigma(\mathbf{U}_h\mathbf{h}_{t-1}+\mathbf{U}_z\mathbf{z}_t+\mathbf{b}_u)$ is the update
    gate.


    Parameters
    ----------
    in_size: bst.typing.Size
        The number of input units.
    out_size: bst.typing.Size
        The number of hidden units.
    w_init: callable, ArrayLike
        The input weight initializer.
    b_init: callable, ArrayLike
        The bias weight initializer.
    state_init: callable, ArrayLike
        The state initializer.
    name: optional, str
        The name of the module.

    """
    __module__ = 'brainscale.nn'

    def __init__(
        self,
        in_size: bst.typing.Size,
        out_size: bst.typing.Size,
        w_init: Union[ArrayLike, Callable] = init.Orthogonal(),
        b_init: Union[ArrayLike, Callable] = init.ZeroInit(),
        state_init: Union[ArrayLike, Callable] = init.ZeroInit(),
        name: str = None,
    ):
        super().__init__(name=name)

        # parameters
        self._state_initializer = state_init
        self.out_size = out_size
        self.in_size = in_size

        # functions
        self.W_x = Linear(self.in_size[-1], self.out_size[-1], w_init=w_init, b_init=b_init)

        # weights
        self.W_f = Linear(self.in_size[-1] + self.out_size[-1], self.out_size[-1], w_init=w_init, b_init=b_init)
        self.W_i = Linear(self.in_size[-1] + self.out_size[-1], self.out_size[-1], w_init=w_init, b_init=b_init)

    def init_state(self, batch_size: int = None, **kwargs):
        self.h = ETraceState(init.param(self._state_initializer, self.out_size, batch_size))

    def reset_state(self, batch_size: int = None, **kwargs):
        self.h.value = init.param(self._state_initializer, self.out_size, batch_size)

    def update(self, x):
        xh = u.math.concatenate([x, self.h.value], axis=-1)
        f = functional.sigmoid(self.W_f(xh))
        i = functional.sigmoid(self.W_i(xh))
        self.h.value = f * self.h.value + i * self.W_x(x)
        return self.h.value


def glorot_init(s):
    return bst.random.randn(*s) / u.math.sqrt(s[0])


class LRUCell(bst.nn.Module):
    r"""
    `Linear Recurrent Unit <https://arxiv.org/abs/2303.06349>`_ (LRU) layer.

    .. math::

       h_{t+1} = \lambda * h_t + \exp(\gamma^{\mathrm{log}}) B x_{t+1} \\
       \lambda = \text{diag}(\exp(-\exp(\nu^{\mathrm{log}}) + i \exp(\theta^\mathrm{log}))) \\
       y_t = Re[C h_t + D x_t]

    Args:
        d_hidden: int
            Hidden state dimension.
        d_model: int
            Input and output dimensions.
        r_min: float, optional
            Smallest lambda norm.
        r_max: float, optional
            Largest lambda norm.
        max_phase: float, optional
            Max phase lambda.
    """

    def __init__(
        self,
        d_model: int,  # input and output dimensions
        d_hidden: int,  # hidden state dimension
        r_min: float = 0.0,  # smallest lambda norm
        r_max: float = 1.0,  # largest lambda norm
        max_phase: float = 6.28,  # max phase lambda
    ):
        super().__init__()

        self.in_size = d_model
        self.out_size = d_hidden

        self.d_hidden = d_hidden
        self.d_model = d_model
        self.r_min = r_min
        self.r_max = r_max
        self.max_phase = max_phase

        # -------- recurrent weight matrix --------

        # theta parameter
        theta_log = u.math.log(max_phase * bst.random.uniform(size=d_hidden))
        self.theta_log = ElementWiseParamOp(theta_log, op=u.math.exp)

        # nu parameter
        nu_log = u.math.log(
            -0.5 * u.math.log(
                bst.random.uniform(size=d_hidden) * (r_max ** 2 - r_min ** 2) + r_min ** 2
            )
        )
        self.nu_log = ElementWiseParamOp(nu_log, op=lambda v: u.math.exp(-u.math.exp(v)))

        # -------- input weight matrix --------

        # gamma parameter
        diag_lambda = u.math.exp(-u.math.exp(nu_log) + 1j * u.math.exp(theta_log))
        gamma_log = u.math.log(u.math.sqrt(1 - u.math.abs(diag_lambda) ** 2))
        self.gamma_log = ElementWiseParamOp(gamma_log, op=u.math.exp)

        # Glorot initialized Input/Output projection matrices
        self.B_re = Linear(d_model, d_hidden, w_init=glorot_init, b_init=None)
        self.B_im = Linear(d_model, d_hidden, w_init=glorot_init, b_init=None)

        # -------- output weight matrix --------

        self.C_re = Linear(d_hidden, d_model, w_init=glorot_init, b_init=None)
        self.C_im = Linear(d_hidden, d_model, w_init=glorot_init, b_init=None)

        # Parameter for skip connection
        D = bst.random.randn(d_model)
        self.D = ETraceParamOp(D, lambda x, p: x * p, is_diagonal=True)

    def init_state(self, batch_size: int = None, **kwargs):
        self.h_re = ETraceState(bst.init.param(bst.init.ZeroInit(), self.d_hidden, batch_size))
        self.h_im = ETraceState(bst.init.param(bst.init.ZeroInit(), self.d_hidden, batch_size))

    def reset_state(self, batch_size: int = None, **kwargs):
        self.h_re.value = bst.init.param(bst.init.ZeroInit(), self.d_hidden, batch_size)
        self.h_im.value = bst.init.param(bst.init.ZeroInit(), self.d_hidden, batch_size)

    def update(self, inputs):
        a = self.nu_log.execute()
        b = self.theta_log.execute()
        c = self.gamma_log.execute()
        a_cos_b = a * u.math.cos(b)
        a_sin_b = a * u.math.sin(b)
        self.h_re.value = a_cos_b * self.h_re.value - a_sin_b * self.h_im.value + c * self.B_re(inputs)
        self.h_im.value = a_sin_b * self.h_re.value + a_cos_b * self.h_im.value + c * self.B_im(inputs)
        r = self.C_re(self.h_re.value) - self.C_im(self.h_im.value) + self.D.execute(inputs)
        return r
