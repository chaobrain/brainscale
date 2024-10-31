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

from brainscale._etrace_concepts import ETraceState, ETraceParamOp
from brainscale._typing import ArrayLike
from ._linear import Linear

__all__ = [
    'ValinaRNNCell', 'GRUCell', 'MGUCell', 'LSTMCell', 'URLSTMCell',
    'RHNCell', 'MinimalRNNCell',
]


class ValinaRNNCell(nn.RNNCell):
    """
    Vanilla RNN cell.
  
    Args:
      num_in: int. The number of input units.
      num_out: int. The number of hidden units.
      state_init: callable, ArrayLike. The state initializer.
      w_init: callable, ArrayLike. The input weight initializer.
      b_init: optional, callable, ArrayLike. The bias weight initializer.
      activation: str, callable. The activation function. It can be a string or a callable function.
      name: optional, str. The name of the module.
    """
    __module__ = 'brainscale.nn'

    def __init__(
        self,
        num_in: int,
        num_out: int,
        state_init: Union[ArrayLike, Callable] = init.ZeroInit(),
        w_init: Union[ArrayLike, Callable] = init.XavierNormal(),
        b_init: Union[ArrayLike, Callable] = init.ZeroInit(),
        activation: str | Callable = 'relu',
        name: str = None,
    ):
        super().__init__(name=name)

        # parameters
        self._state_initializer = state_init
        self.num_out = num_out
        self.num_in = num_in
        self.in_size = (num_in,)
        self.out_size = (num_out,)

        # activation function
        if isinstance(activation, str):
            self.activation = getattr(functional, activation)
        else:
            assert callable(activation), "The activation function should be a string or a callable function. "
            self.activation = activation

        # weights
        self.W = Linear(num_in + num_out, num_out, w_init=w_init, b_init=b_init)

    def init_state(self, batch_size: int = None, **kwargs):
        self.h = ETraceState(init.param(self._state_initializer, self.num_out, batch_size))

    def reset_state(self, batch_size: int = None, **kwargs):
        self.h.value = init.param(self._state_initializer, self.num_out, batch_size)

    def update(self, x):
        xh = u.math.concatenate([x, self.h.value], axis=-1)
        self.h.value = self.activation(self.W(xh))
        return self.h.value


class GRUCell(nn.RNNCell):
    """
    Gated Recurrent Unit (GRU) cell.
  
    Args:
      num_in: int. The number of input units.
      num_out: int. The number of hidden units.
      state_init: callable, ArrayLike. The state initializer.
      w_init: callable, ArrayLike. The input weight initializer.
      b_init: optional, callable, ArrayLike. The bias weight initializer.
      activation: str, callable. The activation function. It can be a string or a callable function.
      name: optional, str. The name of the module.
    """
    __module__ = 'brainscale.nn'

    def __init__(
        self,
        num_in: int,
        num_out: int,
        w_init: Union[ArrayLike, Callable] = init.Orthogonal(),
        b_init: Union[ArrayLike, Callable] = init.ZeroInit(),
        state_init: Union[ArrayLike, Callable] = init.ZeroInit(),
        activation: str | Callable = 'tanh',
        name: str = None,
    ):
        super().__init__(name=name)

        # parameters
        self._state_initializer = state_init
        self.num_out = num_out
        self.num_in = num_in
        self.in_size = (num_in,)
        self.out_size = (num_out,)

        # activation function
        if isinstance(activation, str):
            self.activation = getattr(functional, activation)
        else:
            assert callable(activation), "The activation function should be a string or a callable function. "
            self.activation = activation

        # weights
        self.Wz = Linear(num_in + num_out, num_out, w_init=w_init, b_init=b_init)
        self.Wr = Linear(num_in + num_out, num_out, w_init=w_init, b_init=b_init)
        self.Wh = Linear(num_in + num_out, num_out, w_init=w_init, b_init=b_init)

    def init_state(self, batch_size: int = None, **kwargs):
        self.h = ETraceState(init.param(self._state_initializer, [self.num_out], batch_size))

    def reset_state(self, batch_size: int = None, **kwargs):
        self.h.value = init.param(self._state_initializer, [self.num_out], batch_size)

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


class MGUCell(nn.RNNCell):
    r"""
    Minimal Gated Recurrent Unit (MGU) cell.
  
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
      num_in: int. The number of input units.
      num_out: int. The number of hidden units.
      state_init: callable, ArrayLike. The state initializer.
      w_init: callable, ArrayLike. The input weight initializer.
      b_init: optional, callable, ArrayLike. The bias weight initializer.
      activation: str, callable. The activation function. It can be a string or a callable function.
      name: optional, str. The name of the module.
    """
    __module__ = 'brainscale.nn'

    def __init__(
        self,
        num_in: int,
        num_out: int,
        w_init: Union[ArrayLike, Callable] = init.Orthogonal(),
        b_init: Union[ArrayLike, Callable] = init.ZeroInit(),
        state_init: Union[ArrayLike, Callable] = init.ZeroInit(),
        activation: str | Callable = 'tanh',
        name: str = None,
    ):
        super().__init__(name=name)

        # parameters
        self._state_initializer = state_init
        self.num_out = num_out
        self.num_in = num_in
        self.in_size = (num_in,)
        self.out_size = (num_out,)

        # activation function
        if isinstance(activation, str):
            self.activation = getattr(functional, activation)
        else:
            assert callable(activation), "The activation function should be a string or a callable function. "
            self.activation = activation

        # weights
        self.Wf = Linear(num_in + num_out, num_out, w_init=w_init, b_init=b_init)
        self.Wh = Linear(num_in + num_out, num_out, w_init=w_init, b_init=b_init)

    def init_state(self, batch_size: int = None, **kwargs):
        self.h = ETraceState(init.param(self._state_initializer, [self.num_out], batch_size))

    def reset_state(self, batch_size: int = None, **kwargs):
        self.h.value = init.param(self._state_initializer, [self.num_out], batch_size)

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
    num_in: int
      The dimension of the input vector
    num_out: int
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
        num_in: int,
        num_out: int,
        w_init: Union[ArrayLike, Callable] = init.XavierNormal(),
        b_init: Union[ArrayLike, Callable] = init.ZeroInit(),
        state_init: Union[ArrayLike, Callable] = init.ZeroInit(),
        activation: str | Callable = 'tanh',
        name: str = None,
    ):
        super().__init__(name=name)

        # parameters
        self.num_out = num_out
        self.num_in = num_in
        self.in_size = (num_in,)
        self.out_size = (num_out,)

        # initializers
        self._state_initializer = state_init

        # activation function
        if isinstance(activation, str):
            self.activation = getattr(functional, activation)
        else:
            assert callable(activation), "The activation function should be a string or a callable function. "
            self.activation = activation

        # weights
        self.Wi = Linear(num_in + num_out, num_out, w_init=w_init, b_init=b_init)
        self.Wg = Linear(num_in + num_out, num_out, w_init=w_init, b_init=b_init)
        self.Wf = Linear(num_in + num_out, num_out, w_init=w_init, b_init=b_init)
        self.Wo = Linear(num_in + num_out, num_out, w_init=w_init, b_init=b_init)

    def init_state(self, batch_size: int = None, **kwargs):
        self.c = ETraceState(init.param(self._state_initializer, [self.num_out], batch_size))
        self.h = ETraceState(init.param(self._state_initializer, [self.num_out], batch_size))

    def reset_state(self, batch_size: int = None, **kwargs):
        self.c.value = init.param(self._state_initializer, [self.num_out], batch_size)
        self.h.value = init.param(self._state_initializer, [self.num_out], batch_size)

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
        num_in: int,
        num_out: int,
        w_init: Union[ArrayLike, Callable] = init.XavierNormal(),
        state_init: Union[ArrayLike, Callable] = init.ZeroInit(),
        activation: str | Callable = 'tanh',
        name: str = None,
    ):
        super().__init__(name=name)

        # parameters
        self.num_out = num_out
        self.num_in = num_in
        self.in_size = (num_in,)
        self.out_size = (num_out,)

        # initializers
        self._state_initializer = state_init

        # activation function
        if isinstance(activation, str):
            self.activation = getattr(functional, activation)
        else:
            assert callable(activation), "The activation function should be a string or a callable function. "
            self.activation = activation

        # weights
        self.Wu = Linear(num_in + num_out, num_out, w_init=w_init, b_init=None)
        self.Wf = Linear(num_in + num_out, num_out, w_init=w_init, b_init=None)
        self.Wr = Linear(num_in + num_out, num_out, w_init=w_init, b_init=None)
        self.Wo = Linear(num_in + num_out, num_out, w_init=w_init, b_init=None)
        self.bias = ETraceParamOp(self._forget_bias(), op=u.math.add, grad='full')

    def _forget_bias(self):
        u = bst.random.uniform(1 / self.num_out, 1 - 1 / self.num_out, (self.num_out,))
        return -u.math.log(1 / u - 1)

    def init_state(self, batch_size: int = None, **kwargs):
        self.c = ETraceState(init.param(self._state_initializer, [self.num_out], batch_size))
        self.h = ETraceState(init.param(self._state_initializer, [self.num_out], batch_size))

    def reset_state(self, batch_size: int = None, **kwargs):
        self.c.value = init.param(self._state_initializer, [self.num_out], batch_size)
        self.h.value = init.param(self._state_initializer, [self.num_out], batch_size)

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


class _RHNBlock(bst.nn.Module):
    def __init__(
        self,
        num_in: int,
        num_out: int,
        w_init: Union[ArrayLike, Callable] = init.Orthogonal(),
        b_init: Union[ArrayLike, Callable] = None,
        name: str = None,
        first_layer: bool = False,
        couple: bool = False,
        dropout_prob: float = 1.0,
    ):
        super().__init__(name=name)
        self.num_in = num_in
        self.num_out = num_out
        self.first_layer = first_layer
        self.couple = couple
        if first_layer:
            self.W_H = Linear(num_in + num_out, num_out, w_init=w_init, b_init=b_init)
            self.W_T = Linear(num_in + num_out, num_out, w_init=w_init, b_init=b_init)
            if not couple:
                self.W_C = nn.Linear(num_in + num_out, num_out, w_init=w_init, b_init=b_init)
        else:
            self.W_H = Linear(num_out, num_out, w_init=w_init, b_init=b_init)
            self.W_T = Linear(num_out, num_out, w_init=w_init, b_init=b_init)
            if not couple:
                self.W_C = nn.Linear(num_out, num_out, w_init=w_init, b_init=b_init)
        self.dropout = nn.Dropout(dropout_prob)

    def update(self, x, hidden):
        if self.first_layer:
            x = u.math.concatenate([x, hidden], axis=-1)
        else:
            x = hidden  # ignore input
        h = u.math.tanh(self.W_H(x))
        t = bst.functional.sigmoid(self.W_T(x))
        if self.couple:
            c = 1 - t
        else:
            c = bst.functional.sigmoid(self.W_C(x))
        t = self.dropout(t)
        h = h * t + hidden * c
        return h


class RHNCell(nn.RNNCell):
    r"""
    The recurrent highway cell.
  
    Residual Layers
    ---------------
  
    A **Residual connection** (He et al., 2015) in a neural network is a mechanism that mitigates vanishing gradients
    via "skip connections" that allow smooth gradient flow. Using residual connections aid in training of deep neural
    networks and this has been shown over time with empirical results. Given an input vector $x \in \mathbb{R}^n$,
    the output for a certain layer in a residual neural network is given by:
  
    $$y = f(W x + b) + x$$
  
    where $W$ and $b$ are the weight matrix and the bias vector of the layer and $f$ is a nonlinear activation function.
    Residual layers have success stories in many applications and have found homes in state-of-the-art architectures,
    such as the ResNet (He et al., 2015) series in computer vision, and BERT (Devlin et al., 2019) in natural language
    processing.
  
  
    Highway Layers
    --------------
  
    A modification to the residual layer is the **Highway Layer** (Srivastava et al.,  2015a). Inspired by gating
    mechanisms in the Long-Short Term Memory (Hochreiter & Schmidhuber, 1997), the highway layer uses gates to control
    how much information to pass and how much information to retain from the skip connection via learned weights.
  
    Given $h = H(x, W_H)$, $t = T(x, W_T)$, and $c = C(x, W_C)$ where $h$, $t$, and $c$ are the results of nonlinear
    transforms $H$, $T$, and $C$ with associated weight matrices $W_{H, T, C}$ and biases $b_{H, T, C}$, the output
    $y$ of a highway layer is computed as:
  
    $$y = h \odot t + x \odot c$$
  
    Where $\odot$ is the hadamard (elementwise) product operation. In practice, $H$ often uses the $tanh$ nonlinearity,
    and the $T$ and $C$ use the sigmoid ($\sigma$) nonlinearity.
  
    $H$ can often be thought of as the "main non-linear transform" of the input $x$. $T$ and $C$ act as gates,
    controling from a range of $[0, 1]$ how much of the transformed input and the original input are to be carried over.
    In practice, a suggestion from the Highway networks paper is to couple the $C$ gate to the output of the $T$ gate
    by setting $C(\cdot) = 1 - T(\cdot)$. This reduces the parameters to optimize and could prevent an unbounded
    blow-up of states, which makes optimization smoother. However, this imposes a modeling bias, which could prove
    suboptimal for certain tasks (Greff et al., 2015; Jozefowicz et al., 2015).
  
    Recurrrent Highways
    -------------------
  
    A **Recurrent Highway** (Zilly et al., 2017) adapts the idea of a highway layer to include a recurrence mechanism, 
    acting as a drop-in replacement for LSTMs or other gated-RNNs for a variety of sequence modeling tasks. 
    An improvement that recurrent highways have is a timestep-to-timestep transition larger than one, as opposed 
    to common gated-RNNs.
  
    Recall that a general RNN transition given $s$ timesteps is in the form:
  
    $$y^{[s]} = f(Wx^{[s]} + Ry^{[s - 1]} + b)$$
  
    A one-depth Recurrent Highway Network transition is given by:
  
    $$h^{[s]} = tanh(W_{H}x^{[s]} + R_{H}y^{[s - 1]} + b)$$
    $$t^{[s]} = \sigma(W_{T}x^{[s]} + R_{T}y^{[s - 1]} + b)$$
    $$c^{[s]} = 1 - t^{[s]}$$
    $$y^{[s]} = h^{[s]} \odot t^{[s]} + y^{[s - 1]} \odot c^{[s]}$$
  
    By stacking multiple recurrent highways on top of each other, we could achieve a larger timestep-to-timestep 
    transition. Given $L$ layers,  $l = \{1, 2, ..., L\}$ "ticks" in each timestep, and $s_l$ as the intermediate 
    output between stacked layers, the recurrence can be expanded to:
  
    $$h^{[s]}_l = tanh(W_Hx^{[s]}\mathcal{I}_{\{l=1\}} + R_{H_l}s^{[s]}_{l-1} + b_{H_l})$$
    $$t^{[s]}_l = \sigma(W_Tx^{[s]}\mathcal{I}_{\{l=1\}} + R_{T_l}s^{[s]}_{l-1} + b_{T_l})$$
    $$c^{[s]}_l = \sigma(W_Cx^{[s]}\mathcal{I}_{\{l=1\}} + R_{C_l}s^{[s]}_{l-1} + b_{C_l})$$
    $$s^{[t]}_0 = y^{[t-1]}$$
    $$s^{[t]}_l = h^{[s]}_l \odot t^{[s]}_l + s^{[t]}_{l-1} \odot c^{[s]}_l$$
  
    where $\mathcal{I}$ is the indicator function. Like in the standard highway layer and the one-depth recurrent 
    highway, the $C$ and $T$ gates can be coupled setting $c^{[s]}_l = 1 - t^{[s]}_l$, reducing the numer of parameters 
    to optimize. We can introduce recurrent dropout (Semeniuta et al., 2014) on $t$ as a one hyperparameter for all 
    layers. The recurrent highway can be used as a drop-in replacement to any gated-RNN cell in any sequence-modeling 
    architecture.
  
    We construct our highway layer block such that it can be used as a standard highway layer or can be stacked in a 
    recurrent highway layer. The ```first_layer``` argument should set to ```True``` when stacking (see indicator function
    when $l = 1$ in the equations.) We can set the option to ```couple``` $C$ and $T$ as well. We also use recurrent 
    dropout on $t$ as needed.
  
    Parameters
    ----------
    num_in: int
      The number of input units.
    num_out: int
      The number of hidden units.
    w_init: callable, ArrayLike
      The input weight initializer.
    b_init: callable, ArrayLike
      The bias weight initializer.
    state_init: callable, ArrayLike
      The state initializer.
    name: optional, str
      The name of the module.
    couple: bool
      Whether to couple the $C$ and $T$ gates.
    dropout_prob: float
      The dropout probability for the $t$ gate.
    depth: int
      The number of recurrence depth in the recurrent highway.
    """
    __module__ = 'brainscale.nn'

    def __init__(
        self,
        num_in: int,
        num_out: int,
        w_init: Union[ArrayLike, Callable] = init.Orthogonal(),
        b_init: Union[ArrayLike, Callable] = None,
        state_init: Union[ArrayLike, Callable] = init.ZeroInit(),
        name: str = None,
        couple: bool = False,
        dropout_prob: float = 1.0,
        depth: int = 3,
    ):
        super().__init__(name=name)
        self.num_in = num_in
        self.num_out = num_out
        self.depth = depth
        self.couple = couple
        self._state_init = state_init

        self.highways = [
            _RHNBlock(num_in, num_out, first_layer=l == 0, couple=couple, dropout_prob=dropout_prob,
                      w_init=w_init, b_init=b_init, name=f'highway_{l}')
            for l in range(depth)
        ]

    def init_state(self, batch_size: int = None, **kwargs):
        self.h = ETraceState(init.param(self._state_init, [self.num_out], batch_size))

    def reset_state(self, batch_size: int = None, **kwargs):
        self.h.value = init.param(self._state_init, [self.num_out], batch_size)

    def forward(self, x):
        # expects input (x) dimensions [bs, inp_dim]
        hidden = self.h.value
        for block in self.highways:
            # TODO: multiple recurrence
            hidden = block(x, hidden)
        self.h.value = hidden
        return hidden


class MinimalRNNCell(nn.RNNCell):
    r"""
    Minimal RNN Cell.
  
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
    num_in: int
      The number of input units.
    num_out: int
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
        num_in: int,
        num_out: int,
        w_init: Union[ArrayLike, Callable] = init.Orthogonal(),
        b_init: Union[ArrayLike, Callable] = init.ZeroInit(),
        state_init: Union[ArrayLike, Callable] = init.ZeroInit(),
        phi: Callable = None,
        name: str = None,
    ):
        super().__init__(name=name)

        # parameters
        self._state_initializer = state_init
        self.num_out = num_out
        self.num_in = num_in
        self.in_size = (num_in,)
        self.out_size = (num_out,)

        # functions
        if phi is None:
            phi = Linear(num_in, num_out, w_init=w_init, b_init=b_init)
        assert callable(phi), f"The phi function should be a callable function. But got {phi}"
        self.phi = phi

        # weights
        self.W_u = Linear(num_out * 2, num_out, w_init=w_init, b_init=b_init)

    def init_state(self, batch_size: int = None, **kwargs):
        self.h = ETraceState(init.param(self._state_initializer, [self.num_out], batch_size))

    def reset_state(self, batch_size: int = None, **kwargs):
        self.h.value = init.param(self._state_initializer, [self.num_out], batch_size)

    def update(self, x):
        z = self.phi(x)
        u_ = functional.sigmoid(self.W_u(u.math.concatenate([z, self.h.value], axis=-1)))
        self.h.value = u_ * self.h.value + (1 - u_) * z
        return self.h.value
