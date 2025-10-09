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

import numbers
from typing import Callable, Optional

import brainpy
import brainstate
import braintools
import brainunit as u
import jax
import jax.numpy as jnp

from brainscale._etrace_concepts import ETraceParam, ETraceState
from brainscale._etrace_operators import MatMulOp
from brainscale._typing import Size, ArrayLike, Spike

__all__ = [
    'LeakyRateReadout',
    'LeakySpikeReadout',
]


class LeakyRateReadout(brainstate.nn.Module):
    """
    Leaky dynamics for the read-out module used in the Real-Time Recurrent Learning.

    The LeakyRateReadout class implements a leaky integration mechanism
    for processing continuous input signals in neural networks. It is
    designed to simulate the dynamics of rate-based neurons, applying
    leaky integration to the input and producing a continuous output
    signal.

    This class is part of the BrainScale project and integrates with
    the Brain Dynamics Programming ecosystem, providing a biologically
    inspired approach to neural computation.

    Attributes
    ----------
    in_size : Size
        The size of the input to the readout module.
    out_size : Size
        The size of the output from the readout module.
    tau : ArrayLike
        The time constant for the leaky integration dynamics.
    w_init : Callable
        A callable for initializing the weights of the readout module.
    r_initializer : Callable
        A callable for initializing the state of the readout module.
    name : Optional[str]
        An optional name for the module.
    """
    __module__ = 'brainscale.nn'

    def __init__(
        self,
        in_size: Size,
        out_size: Size,
        tau: ArrayLike = 5. * u.ms,
        w_init: Callable = braintools.init.KaimingNormal(),
        r_initializer: Callable = braintools.init.ZeroInit(),
        name: Optional[str] = None,
    ):
        """
        Initializes the LeakyRateReadout module with specified parameters.

        This constructor sets up the initial configuration for the LeakyRateReadout
        module, including input and output sizes, time constant, weight initialization,
        and state initialization.

        Parameters
        ----------
        in_size : Size
            The size of the input to the readout module. It can be an integer or a tuple
            representing the dimensions of the input.
        out_size : Size
            The size of the output from the readout module. It can be an integer or a tuple
            representing the dimensions of the output.
        tau : ArrayLike, optional
            The time constant for the leaky integration dynamics. Default is 5 milliseconds.
        w_init : Callable, optional
            A callable for initializing the weights of the readout module. Default is KaimingNormal.
        r_initializer : Callable, optional
            A callable for initializing the state of the readout module. Default is ZeroInit.
        name : Optional[str], optional
            An optional name for the module. Default is None.
        """
        super().__init__(name=name)

        # parameters
        self.in_size = (in_size,) if isinstance(in_size, numbers.Integral) else tuple(in_size)
        self.out_size = (out_size,) if isinstance(out_size, numbers.Integral) else tuple(out_size)
        self.tau = braintools.init.param(tau, self.in_size)
        self.decay = jnp.exp(-brainstate.environ.get_dt() / self.tau)
        self.r_initializer = r_initializer

        # weights
        weight = braintools.init.param(w_init, (self.in_size[0], self.out_size[0]))
        self.weight_op = ETraceParam({'weight': weight}, op=MatMulOp())

    def init_state(self, batch_size=None, **kwargs):
        """
        Initializes the state of the readout module.

        This function sets up the initial state for the readout module
        using the specified batch size and any additional keyword arguments.

        Parameters
        ----------
        batch_size : int, optional
            The size of the batch for which the state is initialized. If not provided,
            the default behavior is used.
        **kwargs
            Additional keyword arguments that may be used for state initialization.
        """
        self.r = ETraceState(braintools.init.param(self.r_initializer, self.out_size, batch_size))

    def reset_state(self, batch_size=None, **kwargs):
        """
        Resets the state of the readout module.

        This function resets the internal state of the readout module to its initial
        configuration using the specified batch size and any additional keyword arguments.

        Parameters
        ----------
        batch_size : int, optional
            The size of the batch for which the state is reset. If not provided,
            the default behavior is used.
        **kwargs
            Additional keyword arguments that may be used for state resetting.

        """
        self.r.value = braintools.init.param(self.r_initializer, self.out_size, batch_size)

    def update(self, x):
        """
        Updates the state of the readout module with the given input.

        This function applies the leaky integration dynamics to the input
        and updates the internal state accordingly.

        Parameters
        ----------
        x : ArrayLike
            The input to the readout module. It should be compatible with
            the dimensions specified during the initialization of the module.

        Returns
        -------
        ArrayLike
            The updated state of the readout module after applying the
            leaky integration dynamics.
        """
        r = self.decay * self.r.value + self.weight_op.execute(x)
        self.r.value = r
        return r


class LeakySpikeReadout(brainpy.Neuron):
    """
    Integrate-and-fire neuron model for spike-based readout.

    The LeakySpikeReadout class implements a leaky integrate-and-fire
    neuron model used for spike-based readout in neural networks. It
    simulates the dynamics of membrane potential and spike generation
    based on input spikes, using specified parameters such as time
    constant, threshold voltage, and spike function.

    This class is part of the BrainScale project and is designed to
    integrate with the Brain Dynamics Programming ecosystem, providing
    a biologically inspired approach to neural computation.

    Attributes
    ----------
    in_size : Size
        The size of the input to the readout module.
    keep_size : bool
        A flag indicating whether to keep the input size unchanged.
    tau : ArrayLike
        The time constant for the leaky integration dynamics.
    V_th : ArrayLike
        The threshold voltage for spike generation.
    w_init : Callable
        A callable for initializing the weights of the readout module.
    V_initializer : Callable
        A callable for initializing the membrane potential.
    spk_fun : Callable
        A callable representing the spike function.
    spk_reset : str
        The method for resetting spikes.
    name : str
        An optional name for the module.
    """

    __module__ = 'brainscale.nn'

    def __init__(
        self,
        in_size: Size,
        out_size: Size,
        tau: ArrayLike = 5. * u.ms,
        V_th: ArrayLike = 1. * u.mV,
        w_init: Callable = braintools.init.KaimingNormal(unit=u.mV),
        V_initializer: Callable = braintools.init.ZeroInit(unit=u.mV),
        spk_fun: Callable = braintools.surrogate.ReluGrad(),
        spk_reset: str = 'soft',
        name: str = None,
    ):
        """
        Initializes the LeakySpikeReadout module with specified parameters.

        This constructor sets up the initial configuration for the LeakySpikeReadout
        module, including input size, time constant, threshold voltage, weight initialization,
        and spike function.
        
        Parameters
        ----------
        in_size : Size
            The size of the input to the readout module. It can be an integer or a tuple
            representing the dimensions of the input.
        tau : ArrayLike, optional
            The time constant for the leaky integration dynamics. Default is 5 milliseconds.
        V_th : ArrayLike, optional
            The threshold voltage for spike generation. Default is 1 millivolt.
        w_init : Callable, optional
            A callable for initializing the weights of the readout module. Default is KaimingNormal.
        V_initializer : Callable, optional
            A callable for initializing the membrane potential. Default is ZeroInit.
        spk_fun : Callable, optional
            A callable representing the spike function. Default is ReluGrad.
        spk_reset : str, optional
            The method for resetting spikes. Default is 'soft'.
        name : str, optional
            An optional name for the module. Default is None.
        """
        super().__init__(in_size, name=name, spk_fun=spk_fun, spk_reset=spk_reset)
        self.out_size = out_size

        # parameters
        self.tau = braintools.init.param(tau, self.varshape)
        self.V_th = braintools.init.param(V_th, self.varshape)
        self.V_initializer = V_initializer

        # weights
        weight = braintools.init.param(w_init, (self.in_size[0], self.out_size[0]))
        self.weight_op = ETraceParam({'weight': weight}, op=MatMulOp())

    def init_state(self, batch_size, **kwargs):
        """
        Initializes the state of the LeakySpikeReadout module.

        This function sets up the initial membrane potential state for the
        LeakySpikeReadout module using the specified batch size and any
        additional keyword arguments.

        Parameters
        ----------
        batch_size : int
            The size of the batch for which the state is initialized.
        **kwargs
            Additional keyword arguments that may be used for state initialization.
        """
        self.V = ETraceState(braintools.init.param(self.V_initializer, self.varshape, batch_size))

    def reset_state(self, batch_size, **kwargs):
        """
        Resets the state of the LeakySpikeReadout module.

        This function resets the internal membrane potential state of the
        LeakySpikeReadout module to its initial configuration using the
        specified batch size and any additional keyword arguments.

        Parameters
        ----------
        batch_size : int
            The size of the batch for which the state is reset.
        **kwargs
            Additional keyword arguments that may be used for state resetting.

        """
        self.V.value = braintools.init.param(self.V_initializer, self.varshape, batch_size)

    @property
    def spike(self):
        """
        Computes the spike output of the LeakySpikeReadout module.

        This property calculates the spike output based on the current
        membrane potential value.

        Returns
        -------
        Spike
            The spike output of the module.
        """
        return self.get_spike(self.V.value)

    def get_spike(self, V):
        """
        Computes the spike output based on the membrane potential.

        This function calculates the spike output using the specified
        spike function and the current membrane potential.

        Parameters
        ----------
        V : ArrayLike
            The current membrane potential.

        Returns
        -------
        Spike
            The computed spike output.
        """
        v_scaled = (V - self.V_th) / self.V_th
        return self.spk_fun(v_scaled)

    def update(self, spike: Spike) -> Spike:
        """
        Updates the membrane potential and computes the spike output.

        This function applies the leaky integrate-and-fire dynamics to
        update the membrane potential and compute the resulting spike
        output based on the input spike.

        Parameters
        ----------
        spike : Spike
            The input spike to the readout module.

        Returns
        -------
        Spike
            The updated spike output after applying the dynamics.
        """
        # reset
        last_V = self.V.value
        last_spike = self.get_spike(last_V)
        V_th = self.V_th if self.spk_reset == 'soft' else jax.lax.stop_gradient(last_V)
        V = last_V - V_th * last_spike
        # membrane potential
        dv = lambda v, x: (-v + self.sum_current_inputs(x, v)) / self.tau
        V = brainstate.nn.exp_euler_step(dv, V, self.weight_op.execute(spike))
        V = self.sum_delta_inputs(V)
        self.V.value = V
        return self.get_spike(V)
