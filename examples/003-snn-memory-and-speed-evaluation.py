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

import argparse
import os.path
import time
from functools import reduce
from typing import Any, Callable, Union

import matplotlib.pyplot as plt

import numpy as np

from utils import MyArgumentParser

parser = MyArgumentParser()
parser.add_argument('--memory_eval', type=int, default=0, help='1: True, 0: False')

# Learning parameters
parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
parser.add_argument("--batch_size", type=int, default=128, help="Batch size.")
parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs.")
parser.add_argument("--dt", type=float, default=1., help="The simulation time step.")
parser.add_argument("--loss", type=str, default='cel', choices=['cel', 'mse'], help="Loss function.")

# dataset
parser.add_argument("--mode", type=str, default="train", choices=['train', 'sim'], help="The computing mode.")
parser.add_argument("--n_data_worker", type=int, default=0, help="Number of data loading workers")
parser.add_argument("--data_length", type=int, default=200, help="")
parser.add_argument("--drop_last", type=int, default=0, help="")

# training parameters
parser.add_argument("--exp_name", type=str, default='', help="")
parser.add_argument("--spk_fun", type=str, default='s2nn', help="spike surrogate gradient function.")
parser.add_argument("--warmup_ratio", type=float, default=0.0, help="The ratio for network simulation.")
parser.add_argument("--optimizer", type=str, default='adam', help="")
parser.add_argument("--filepath", type=str, default='', help="The name for the current experiment.")
parser.add_argument("--lr_milestones", type=str, default='', help="The name for the current experiment.")

# regularization parameters
parser.add_argument("--spk_reg_factor", type=float, default=0.0, help="Spike regularization factor.")
parser.add_argument("--spk_reg_rate", type=float, default=10., help="Target firing rate.")
parser.add_argument("--v_reg_factor", type=float, default=0.0, help="Voltage regularization factor.")
parser.add_argument("--v_reg_low", type=float, default=-20., help="The lowest voltage for regularization.")
parser.add_argument("--v_reg_high", type=float, default=1.4, help="The highest voltage for regularization.")
parser.add_argument("--weight_L1", type=float, default=0.0, help="The weight L1 regularization.")
parser.add_argument("--weight_L2", type=float, default=0.0, help="The weight L2 regularization.")

# model parameters
parser.add_argument("--model", type=str, default='lif-delta', help="The model architecture.")
parser.add_argument("--detach_spk", type=int, default=0, help="Number of recurrent neurons.")
parser.add_argument("--n_rec", type=int, default=200, help="Number of recurrent neurons.")
parser.add_argument("--n_layer", type=int, default=2, help="Number of recurrent layers.")
parser.add_argument("--V_th", type=float, default=1.)
parser.add_argument("--tau_mem_sigma", type=float, default=1.)
parser.add_argument("--tau_mem", type=float, default=10.)
parser.add_argument("--tau_syn", type=float, default=10.)
parser.add_argument("--tau_o", type=float, default=10.)
parser.add_argument("--ff_scale", type=float, default=10.)
parser.add_argument("--rec_scale", type=float, default=2.)
parser.add_argument("--spk_reset", type=str, default='soft')

global_args = parser.parse_args()

import brainscale
import brainstate as bst
import braintools
import brainunit as u
import jax
import jax.numpy as jnp
from torch.utils.data import DataLoader

PyTree = Any


def _format_sim_epoch(sim: Union[int, float], length: int):
    if 0. <= sim < 1.:
        return int(length * sim)
    else:
        return int(sim)


def _raster_plot(sp_matrix, times):
    """Get spike raster plot which displays the spiking activity
    of a group of neurons over time.
  
    Parameters
    ----------
    sp_matrix : bnp.ndarray
        The matrix which record spiking activities.
    times : bnp.ndarray
        The time steps.
  
    Returns
    -------
    raster_plot : tuple
        Include (neuron index, spike time).
    """
    sp_matrix = np.asarray(sp_matrix)
    times = np.asarray(times)
    elements = np.where(sp_matrix > 0.)
    index = elements[1]
    times = times[elements[0]]
    return index, times


class _LIF_Delta_Dense_Layer(bst.nn.Module):
    """
    LIF neurons and dense connected delta synapses.
    """

    def __init__(
        self,
        n_in, n_rec,
        tau_mem=5., V_th=1.,
        spk_fun: Callable = bst.surrogate.ReluGrad(),
        spk_reset: str = 'soft',
        rec_scale: float = 1.,
        ff_scale: float = 1.,
    ):
        super().__init__()
        self.neu = brainscale.LIF(n_rec, tau=tau_mem, spk_fun=spk_fun, spk_reset=spk_reset, V_th=V_th)
        rec_init: Callable = bst.init.KaimingNormal(rec_scale)
        ff_init: Callable = bst.init.KaimingNormal(ff_scale)
        w_init = jnp.concat([ff_init([n_in, n_rec]), rec_init([n_rec, n_rec])], axis=0)
        self.syn = bst.nn.DeltaProj(
            comm=brainscale.Linear(n_in + n_rec, n_rec, w_init=w_init),
            post=self.neu
        )

    def update(self, spk):
        inp = jnp.concat([spk, self.neu.get_spike()], axis=-1)
        self.syn(inp)
        self.neu()
        return self.neu.get_spike()


class _LIF_ExpCu_Dense_Layer(bst.nn.Module):
    """
    LIF neurons and dense connected exponential current synapses.
    """

    def __init__(
        self, n_in, n_rec,
        tau_mem=5., tau_syn=10., V_th=1.,
        spk_fun: Callable = bst.surrogate.ReluGrad(),
        spk_reset: str = 'soft',
        rec_scale: float = 1.,
        ff_scale: float = 1.,
    ):
        super().__init__()
        self.neu = brainscale.LIF(n_rec, tau=tau_mem, spk_fun=spk_fun, spk_reset=spk_reset, V_th=V_th)
        rec_init: Callable = bst.init.KaimingNormal(rec_scale)
        ff_init: Callable = bst.init.KaimingNormal(ff_scale)
        w_init = jnp.concat([ff_init([n_in, n_rec]), rec_init([n_rec, n_rec])], axis=0)
        self.syn = bst.nn.AlignPostProj(
            comm=brainscale.Linear(n_in + n_rec, n_rec, w_init),
            syn=brainscale.Expon.delayed(size=n_rec, tau=tau_syn),
            out=bst.nn.CUBA.desc(),
            post=self.neu
        )

    def update(self, spk):
        self.syn(jnp.concat([spk, self.neu.get_spike()], axis=-1))
        self.neu()
        return self.neu.get_spike()


class ETraceNet(bst.nn.Module):
    def __init__(
        self, n_in, n_rec, n_out, n_layer, args,
    ):
        super().__init__()

        # arguments
        self.n_in = n_in
        self.n_rec = n_rec
        self.n_out = n_out
        self.n_layer = args.n_layer

        if args.spk_fun == 's2nn':
            spk_fun = bst.surrogate.S2NN()
        elif args.spk_fun == 'relu':
            spk_fun = bst.surrogate.ReluGrad()
        elif args.spk_fun == 'multi_gaussian':
            spk_fun = bst.surrogate.MultiGaussianGrad()
        else:
            raise ValueError('Unknown spiking surrogate gradient function.')

        # recurrent layers
        self.rec_layers = []
        for layer_idx in range(n_layer):
            tau_mem = (bst.random.normal(args.tau_mem, args.tau_mem_sigma, [n_rec])
                       if args.tau_mem_sigma > 0. else
                       args.tau_mem)
            if args.model == 'lif-exp-cu':
                rec = _LIF_ExpCu_Dense_Layer(
                    n_rec=n_rec, n_in=n_in, tau_mem=tau_mem, tau_syn=args.tau_syn, V_th=args.V_th,
                    spk_fun=spk_fun, spk_reset=args.spk_reset,
                    rec_scale=args.rec_scale, ff_scale=args.ff_scale,
                )
                n_in = n_rec
            elif args.model == 'lif-delta':
                rec = _LIF_Delta_Dense_Layer(
                    n_rec=n_rec, n_in=n_in, tau_mem=tau_mem, V_th=args.V_th,
                    spk_fun=spk_fun, spk_reset=args.spk_reset,
                    rec_scale=args.rec_scale, ff_scale=args.ff_scale,
                )
                n_in = n_rec
            else:
                raise ValueError('Unknown neuron model.')

            self.rec_layers.append(rec)

        # output layer
        self.out = brainscale.LeakyRateReadout(
            in_size=n_rec,
            out_size=n_out,
            tau=args.tau_o,
            w_init=bst.init.KaimingNormal()
        )

    def update(self, x):
        for i in range(self.n_layer):
            x = self.rec_layers[i](x)
        return self.out(x)

    def membrane_reg(self, mem_low: float, mem_high: float, factor: float = 0.):
        loss = 0.
        if factor > 0.:
            # extract all Neuron models
            neurons = self.nodes().subset(bst.nn.Neuron).unique().values()
            # evaluate the membrane potential
            for l in neurons:
                loss += jnp.square(jnp.mean(jax.nn.relu(l.V.value - mem_high) ** 2 +
                                            jax.nn.relu(mem_low - l.V.value) ** 2))
            loss = loss * factor
        return loss

    def spike_reg(self, target_fr: float, factor: float = 0.):
        # target_fr: Hz
        loss = 0.
        if factor > 0.:
            # extract all Neuron models
            neurons = self.nodes().subset(bst.nn.Neuron).unique().values()
            # evaluate the spiking dynamics
            for l in neurons:
                loss += (jnp.mean(l.get_spike()) - target_fr / 1e3 * bst.environ.get_dt()) ** 2
            loss = loss * factor
        return loss

    def save(self, step, **kwargs):
        if self.checkpointer is not None:
            states = self.states().subset(bst.ParamState).to_dict_values()
            self.checkpointer.save_data(states, step, **kwargs)

    def restore(self, step=None):
        if self.checkpointer is not None:
            states = self.states().subset(bst.ParamState)
            values = self.checkpointer.load_data(states.to_dict_values(), step=step)
            for k, v in values.items():
                states[k].value = v

    def verify(self, dataloader, x_func, num_show=5, filepath=None):
        def _step(index, x):
            with bst.environ.context(i=index, t=index * bst.environ.get_dt()):
                out = self.update(x)
            return out, [r.neu.get_spike() for r in self.rec_layers], [r.neu.V.value for r in self.rec_layers]

        dataloader = iter(dataloader)
        xs, ys = next(dataloader)  # xs: [n_samples, n_steps, n_in]
        xs = jnp.asarray(x_func(xs))
        print(xs.shape, ys.shape)
        bst.nn.init_all_states(self, xs.shape[1])

        time_indices = np.arange(0, xs.shape[0])
        outs, sps, vs = bst.compile.for_loop(_step, time_indices, xs)
        outs = u.math.as_numpy(outs)
        sps = [u.math.as_numpy(out) for out in sps]
        vs = [u.math.as_numpy(out) for out in vs]
        # vs = [np.where(sp, v + sps_inc, v) for sp, v in zip(sps, vs)]

        ts = time_indices * bst.environ.get_dt()
        max_t = xs.shape[0] * bst.environ.get_dt()

        for i in range(min(num_show, xs.shape[1])):
            fig, gs = braintools.visualize.get_figure(2, len(self.rec_layers) + 1, 3., 3.)

            # input spiking
            ax_inp = fig.add_subplot(gs[0, 0])
            indices, times = _raster_plot(xs[:, i], ts)
            ax_inp.plot(times, indices, 'k,')
            ax_inp.set_xlim(0., max_t)
            ax_inp.set_ylabel('Input Spiking')

            # recurrent spiking
            for j in range(len(self.rec_layers)):
                ax_rec = fig.add_subplot(gs[0, j + 1])
                indices, times = _raster_plot(sps[j][:, i], ts)
                ax_rec.plot(times, indices, 'k,')
                ax_rec.set_xlim(0., max_t)
                ax_rec.set_ylabel(f'Recurrent Spiking L{j}')

            # decision activity
            ax_out = fig.add_subplot(gs[1, 0])
            ax_out.plot(ts, outs[:, i], alpha=0.7)
            ax_out.set_ylabel('Output Activity')
            ax_out.set_xlabel('Time [ms]')
            ax_out.set_xlim(0., max_t)

            # recurrent potential
            for j in range(len(self.rec_layers)):
                ax = fig.add_subplot(gs[1, j + 1])
                plt.plot(ts, vs[j][:, i])
                ax.set_xlim(0., max_t)
                ax.set_ylabel(f'Recurrent Potential L{j}')

            if filepath:
                plt.savefig(f'{filepath}/{i}.png')

            if filepath is None:
                plt.show()
        plt.close()


class Trainer(object):
    """
    The training class with only loss.
    """

    def __init__(
        self,
        target: ETraceNet,
        opt: bst.optim.Optimizer,
        arguments: argparse.Namespace,
    ):
        super().__init__()

        # target network
        self.target = target

        # parameters
        self.args = arguments

        # loss function
        if self.args.loss == 'mse':
            self.loss_fn = braintools.metric.squared_error
        elif self.args.loss == 'cel':
            self.loss_fn = braintools.metric.softmax_cross_entropy_with_integer_labels
        else:
            raise ValueError

        # optimizer
        self.opt = opt
        opt.register_trainable_weights(self.target.states().subset(bst.ParamState))

    def _acc(self, out, target):
        return jnp.mean(jnp.equal(target, jnp.argmax(jnp.mean(out, axis=0), axis=1)))

    def _loss(self, out, target):
        loss = self.loss_fn(out, target).mean()

        # L1 regularization loss
        if self.args.weight_L1 != 0.:
            leaves = self.target.states().subset(bst.ParamState).to_dict_values()
            loss += self.args.weight_L1 * reduce(jnp.add, jax.tree.map(lambda x: jnp.sum(jnp.abs(x)), leaves))

        # membrane potential regularization loss
        if self.args.v_reg_factor != 0.:
            mem_low = self.args.v_reg_low
            mem_high = self.args.v_reg_high
            loss += self.target.membrane_reg(mem_low, mem_high, self.args.v_reg_factor)

        # spike regularization loss
        if self.args.spk_reg_factor != 0.:
            fr = self.args.spk_reg_rate
            loss += self.target.spike_reg(fr, self.args.spk_reg_factor)

        return loss

    @bst.compile.jit(static_argnums=0)
    def predict(self, inputs, targets):
        def _step(i, inp):
            # call the model
            with bst.environ.context(i=i, t=i * bst.environ.get_dt()):
                out = self.target(inp)
                # calculate the loss
                loss = self._loss(out, targets)
                return loss, out

        # initialize the states
        bst.nn.init_all_states(self.target, inputs.shape[1])
        indices = np.arange(inputs.shape[0])
        losses, outs = bst.compile.for_loop(_step, indices, inputs)
        acc = self._acc(outs, targets)
        return losses.mean(), acc

    @bst.compile.jit(static_argnums=(0,))
    def etrace_train(self, inputs, targets):
        # initialize the states
        bst.nn.init_all_states(self.target, inputs.shape[1])
        # weights
        weights = self.target.states().subset(bst.ParamState)

        # the model for a single step
        def _single_step(i, inp, fit: bool = True):
            with bst.environ.context(i=i, t=i * bst.environ.get_dt(), fit=fit):
                out = self.target(inp)
            return out

        # initialize the online learning model
        if self.args.method == 'expsm_diag':
            model = brainscale.DiagIODimAlgorithm(
                _single_step,
                self.args.etrace_decay,
                vjp_time=self.args.vjp_time,
            )
            model.compile_graph(0, jax.ShapeDtypeStruct(inputs.shape[1:], inputs.dtype))
        elif self.args.method == 'diag':
            model = brainscale.DiagParamDimAlgorithm(
                _single_step,
                vjp_time=self.args.vjp_time,
            )
            model.compile_graph(0, jax.ShapeDtypeStruct(inputs.shape[1:], inputs.dtype))
        elif self.args.method == 'hybrid':
            model = brainscale.DiagHybridDimAlgorithm(
                _single_step,
                self.args.etrace_decay,
                vjp_time=self.args.vjp_time,
            )
            model.compile_graph(0, jax.ShapeDtypeStruct(inputs.shape[1:], inputs.dtype))
        else:
            raise ValueError(f'Unknown online learning methods: {self.args.method}.')

        # model.graph.show_graph(start_frame=0, n_frame=5)

        def _etrace_grad(i, inp):
            # call the model
            out = model(i, inp, running_index=i)
            # calculate the loss
            loss = self._loss(out, targets)
            return loss, out

        def _etrace_step(prev_grads, x):
            # no need to return weights and states, since they are generated then no longer needed
            i, inp = x
            f_grad = bst.augment.grad(_etrace_grad, grad_vars=weights, has_aux=True, return_value=True)
            cur_grads, local_loss, out = f_grad(i, inp)
            next_grads = jax.tree.map(lambda a, b: a + b, prev_grads, cur_grads)
            return next_grads, (out, local_loss)

        def _etrace_train(indices_, inputs_):
            # forward propagation
            grads = jax.tree.map(lambda a: jnp.zeros_like(a), weights.to_dict_values())
            grads, (outs, losses) = bst.compile.scan(_etrace_step, grads, (indices_, inputs_))
            # gradient updates
            # jax.debug.print('grads = {g}', g=jax.tree.map(lambda a: jnp.max(jnp.abs(a)), grads))
            grads = bst.functional.clip_grad_norm(grads, 1.)
            self.opt.update(grads)
            # accuracy
            return losses.mean(), outs

        # running indices
        indices = np.arange(inputs.shape[0])
        if self.args.warmup_ratio > 0:
            n_sim = _format_sim_epoch(self.args.warmup_ratio, inputs.shape[0])
            bst.compile.for_loop(lambda i, inp: model(i, inp, running_index=i), indices[:n_sim], inputs[:n_sim])
            loss, outs = _etrace_train(indices[n_sim:], inputs[n_sim:])
        else:
            loss, outs = _etrace_train(indices, inputs)

        acc = self._acc(outs, targets)

        mem = jax.pure_callback(
            lambda: jax.devices()[0].memory_stats()['bytes_in_use'] / 1024 / 1024 / 1024,
            jax.ShapeDtypeStruct((), bst.environ.dftype())
        )
        # returns
        return loss, acc, mem

    def memory_efficient_etrace_functions(self, input_info):
        # initialize the states
        bst.nn.init_all_states(self.target, self.args.batch_size)
        # weights
        weights = self.target.states().subset(bst.ParamState)

        # the model for a single step
        def _single_step(i, inp, fit: bool = True):
            with bst.environ.context(i=i, t=i * bst.environ.get_dt(), fit=fit):
                out = self.target(inp)
            return out

        # initialize the online learning model
        if self.args.method == 'expsm_diag':
            model = brainscale.DiagIODimAlgorithm(_single_step, self.args.etrace_decay, )
            model.compile_graph(input_info)
        elif self.args.method == 'diag':
            model = brainscale.DiagParamDimAlgorithm(_single_step, )
            model.compile_graph(input_info)
        elif self.args.method == 'hybrid':
            model = brainscale.DiagHybridDimAlgorithm(_single_step, self.args.etrace_decay, )
            model.compile_graph(input_info)
        else:
            raise ValueError(f'Unknown online learning methods: {self.args.method}.')

        def reset_state(batch_size):
            bst.nn.reset_all_states(self.target, batch_size)
            model.reset_state(batch_size)

        @bst.compile.jit
        def _etrace_single_run(i, inp):
            model(inp)

        def _etrace_grad(i, inp, targets):
            # call the model
            out = model(inp)
            # calculate the loss
            loss = self._loss(out, targets)
            return loss, out

        @bst.compile.jit
        def _etrace_step(prev_grads, inp, targets):
            # no need to return weights and states, since they are generated then no longer needed
            f_grad = bst.augment.grad(_etrace_grad, grad_vars=weights, has_aux=True, return_value=True)
            cur_grads, local_loss, out = f_grad(inp, targets)
            next_grads = jax.tree.map(lambda a, b: a + b, prev_grads, cur_grads)
            return next_grads, out, local_loss

        return reset_state, _etrace_single_run, _etrace_step

    def memory_efficient_etrace_train(self, reset_fun, predict_step, train_step, inputs, targets):
        reset_fun(self.args.batch_size)
        # running indices
        indices = np.arange(inputs.shape[0])
        n_sim = _format_sim_epoch(self.args.warmup_ratio, inputs.shape[0])
        # initial gradients
        grads = jax.tree.map(lambda a: jnp.zeros_like(a), self.opt.weight_states.to_dict_values())
        # training
        outs, losses = [], []
        for i in indices:
            if i < n_sim:
                predict_step(i, inputs[i])
            else:
                grads, out, loss = train_step(grads, i, inputs[i], targets)
                outs.append(out)
                losses.append(loss)

        # gradient updates
        grads = bst.functional.clip_grad_norm(grads, 1.)
        self.opt.update(grads)
        # accuracy
        acc = self._acc(jnp.asarray(outs), targets)
        # memory
        mem = jax.devices()[0].memory_stats()['bytes_in_use'] / 1024 / 1024 / 1024
        return jnp.asarray(losses).mean(), acc, mem

    @bst.compile.jit(static_argnums=(0,))
    def bptt_train(self, inputs, targets):
        # running indices
        indices = np.arange(inputs.shape[0])
        # initialize the states
        bst.nn.init_all_states(self.target, inputs.shape[1])

        # the model for a single step
        def _single_step(i, inp, fit: bool = True):
            with bst.environ.context(i=i, t=i * bst.environ.get_dt(), fit=fit):
                out = self.target(inp)
            return out

        def _run_step_train(i, inp):
            with bst.environ.context(i=i, t=i * bst.environ.get_dt()):
                out = self.target(inp)
                loss = self._loss(out, targets)
            return out, loss

        def _bptt_grad_step():
            if self.args.warmup_ratio > 0:
                n_sim = _format_sim_epoch(self.args.warmup_ratio, inputs.shape[0])
                _ = bst.compile.for_loop(_single_step, indices[:n_sim], inputs[:n_sim])
                outs, losses = bst.compile.for_loop(_run_step_train, indices[n_sim:], inputs[n_sim:])
            else:
                outs, losses = bst.compile.for_loop(_run_step_train, indices, inputs)
            return losses.mean(), outs

        # gradients
        weights = self.target.states().subset(bst.ParamState)
        grads, loss, outs = bst.augment.grad(_bptt_grad_step, grad_vars=weights, has_aux=True, return_value=True)()

        # optimization
        grads = bst.functional.clip_grad_norm(grads, 1.)
        self.opt.update(grads)

        # accuracy
        acc = self._acc(outs, targets)

        mem = jax.pure_callback(
            lambda: jax.devices()[0].memory_stats()['bytes_in_use'] / 1024 / 1024 / 1024,
            jax.ShapeDtypeStruct((), bst.environ.dftype())
        )

        return loss, acc, mem

    def f_train(self, train_loader, test_loader, x_func, y_func):
        print(self.args)

        if self.args.memory_eval and self.args.method != 'bptt':
            reset_fun, pred_fun, train_fun = self.memory_efficient_etrace_functions(
                jax.ShapeDtypeStruct((self.args.batch_size, 32768), jnp.float32)
            )

        max_acc = 0.
        for epoch in range(self.args.epochs):
            epoch_acc, epoch_loss, epoch_time, epoch_mem = [], [], [], []
            for batch, (x_local, y_local) in enumerate(train_loader):
                # inputs and targets
                x_local = x_func(x_local)
                y_local = y_func(y_local)

                t0 = time.time()
                # training
                if self.args.method == 'bptt':
                    loss, acc, mem = self.bptt_train(x_local, y_local)
                elif self.args.memory_eval:
                    loss, acc, mem = self.memory_efficient_etrace_train(
                        reset_fun, pred_fun, train_fun, x_local, y_local
                    )
                else:
                    loss, acc, mem = self.etrace_train(x_local, y_local)
                t = time.time() - t0
                print(f'Epoch {epoch:4d}, training batch {batch:4d}, training loss = {float(loss):.8f}, '
                      f'training acc = {float(acc):.6f}, time = {t:.5f} s, memory = {mem:.2f} GB')
                epoch_acc.append(acc)
                epoch_loss.append(loss)
                epoch_time.append(t)
                epoch_mem.append(mem)

            mean_loss = np.mean(epoch_loss)
            mean_acc = np.mean(epoch_acc)
            mean_time = np.mean(epoch_time[1:-1])
            mean_mem = np.mean(epoch_mem[1:-1])
            print(f'Epoch {epoch:4d}, training loss = {mean_loss:.8f}, '
                  f'training acc = {mean_acc:.6f}, time = {mean_time:.5f} s, '
                  f'memory = {mean_mem:.2f} GB')
            self.opt.lr.step_epoch()

            # training accuracy
            if mean_acc > max_acc:
                max_acc = mean_acc
                # if platform.platform().startswith('Linux'):
                #   self.target.save(epoch)
                #   self.print(f'Save the model at epoch {epoch} with accuracy {max_acc:.6f}')

            # testing accuracy
            epoch_acc, epoch_loss, epoch_time, epoch_mem = [], [], [], []
            for batch, (x_local, y_local) in enumerate(test_loader):
                x_local = x_func(x_local)
                y_local = y_func(y_local)
                loss, acc = self.predict(x_local, y_local)
                epoch_acc.append(acc)
                epoch_loss.append(loss)
            mean_loss = np.mean(epoch_loss)
            mean_acc = np.mean(epoch_acc)
            print(f'Epoch {epoch:4d}, testing loss = {mean_loss:.8f}, '
                  f'testing acc = {mean_acc:.6f}')
            print('')


# We need to stack the batch elements
def numpy_collate(batch):
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple, list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    else:
        return np.array(batch)


class FormattedDVSGesture:
    def __init__(self, filepath: str):
        self.filepath = filepath
        data = np.load(filepath, allow_pickle=True)
        self.xs = data['xs']
        self.ys = data['ys']
        self.img_size = data['img_size']

    def __getitem__(self, idx):
        arr = np.zeros(tuple(self.img_size), dtype=bst.environ.dftype())
        indices = self.xs[idx]
        time_indices = indices[:, 0]
        neuron_indices = indices[:, 1]
        arr[time_indices, neuron_indices] = 1.
        y = self.ys[idx]
        return arr, y

    def __len__(self):
        return len(self.ys)


def _get_gesture_data(args, cache_dir=os.path.expanduser("./data")):
    # The Dynamic Vision Sensor (DVS) Gesture (DVSGesture) dataset consists of 11 classes of hand gestures recorded
    # by a DVS sensor. The DVSGesture dataset is a spiking version of the MNIST dataset. The dataset consists of
    # 60k training and 10k test samples.

    in_shape = (128, 128, 2)
    out_shape = 11
    n_step = args.data_length
    train_path = os.path.join(cache_dir, f"DVSGesture/DVSGesture-mlp-train-step={n_step}.npz")
    test_path = os.path.join(cache_dir, f"DVSGesture/DVSGesture-mlp-test-step={n_step}.npz")
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise ValueError(f'Cache files {train_path} and {test_path} do not exist. '
                         f'please run "dvs-gesture-preprocessing.py" first.')
    else:
        print(f'Used cache files {train_path} and {test_path}.')
    train_set = FormattedDVSGesture(train_path)
    test_set = FormattedDVSGesture(test_path)
    train_loader = DataLoader(
        train_set,
        shuffle=True,
        batch_size=args.batch_size,
        collate_fn=numpy_collate,
        num_workers=args.n_data_worker,
        drop_last=args.drop_last == 1,
    )
    test_loader = DataLoader(
        test_set,
        shuffle=False,
        batch_size=args.batch_size,
        collate_fn=numpy_collate,
        num_workers=args.n_data_worker
    )

    return bst.util.DotDict(
        {
            'train_loader': train_loader,
            'test_loader': test_loader,
            'in_shape': in_shape,
            'out_shape': out_shape,
            'label_process': lambda x: x,
            'input_process': lambda x: np.transpose(x, (1, 0, 2)),
        }
    )


def network_training():
    # environment setting
    bst.environ.set(
        mode=bst.mixin.JointMode(bst.mixin.Batching(), bst.mixin.Training()),
        dt=global_args.dt
    )

    # loading the data
    dataset = _get_gesture_data(global_args)

    # creating the network and optimizer
    net = ETraceNet(
        np.prod(dataset.in_shape),
        global_args.n_rec,
        dataset.out_shape,
        global_args.n_layer,
        args=global_args,
    )

    if global_args.mode == 'sim':
        if global_args.filepath:
            net.restore()
        net.verify(dataset.train_loader, dataset.input_process, num_show=5)

    elif global_args.mode == 'train':
        if global_args.optimizer == 'adam':
            opt_cls = bst.optim.Adam
        elif global_args.optimizer == 'momentum':
            opt_cls = bst.optim.Momentum
        elif global_args.optimizer == 'sgd':
            opt_cls = bst.optim.SGD
        else:
            raise ValueError(f'Unknown optimizer: {global_args.optimizer}')
        if global_args.lr_milestones == '':
            lr = global_args.lr
        else:
            lr_milestones = [int(m) for m in global_args.lr_milestones.split(',')]
            lr = bst.optim.MultiStepLR(global_args.lr, milestones=lr_milestones, gamma=0.2)
        opt = opt_cls(lr=lr, weight_decay=global_args.weight_L2)

        # creating the trainer
        trainer = Trainer(net, opt, global_args)
        trainer.f_train(
            dataset.train_loader,
            dataset.test_loader,
            x_func=dataset.input_process,
            y_func=dataset.label_process,
        )

    else:
        raise ValueError(f'Unknown mode: {global_args.mode}')


if __name__ == '__main__':
    pass
    network_training()
