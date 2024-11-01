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


from typing import Callable
from typing import Iterable, Union

import brainstate as bst
import braintools as bts
import brainunit as u
import jax
import matplotlib.pyplot as plt
import numba
import numpy as np
from tqdm import tqdm

import brainscale

LOSS = float
ACCURACY = float


class GIF(bst.nn.Neuron):
    def __init__(
        self, size,
        V_rest=0. * u.mV,
        V_th_inf=1. * u.mV,
        R=1. * u.ohm,
        tau=20. * u.ms,
        tau_I2=50. * u.ms,
        A2=0. * u.mA,
        V_initializer: Callable = bst.init.ZeroInit(unit=u.mV),
        I2_initializer: Callable = bst.init.ZeroInit(unit=u.mA),
        spike_fun: Callable = bst.surrogate.ReluGrad(),
        spk_reset: str = 'soft',
        keep_size: bool = False,
        name: str = None,
    ):
        super().__init__(size, keep_size=keep_size, name=name, spk_fun=spike_fun, spk_reset=spk_reset)

        # parameters
        self.V_rest = bst.init.param(V_rest, self.varshape, allow_none=False)
        self.V_th_inf = bst.init.param(V_th_inf, self.varshape, allow_none=False)
        self.R = bst.init.param(R, self.varshape, allow_none=False)
        self.tau = bst.init.param(tau, self.varshape, allow_none=False)
        self.tau_I2 = bst.init.param(tau_I2, self.varshape, allow_none=False)
        self.A2 = bst.init.param(A2, self.varshape, allow_none=False)

        # initializers
        self._V_initializer = V_initializer
        self._I2_initializer = I2_initializer

    def init_state(self, batch_size=None):
        # 将模型用于在线学习，需要初始化状态变量
        self.V = brainscale.ETraceState(bst.init.param(self._V_initializer, self.varshape, batch_size))
        self.I2 = brainscale.ETraceState(bst.init.param(self._I2_initializer, self.varshape, batch_size))

    def update(self, x=0.):
        # 如果前一时刻发放了脉冲，则将膜电位和适应性电流进行重置
        last_spk = self.get_spike()
        last_spk = jax.lax.stop_gradient(last_spk)
        last_V = self.V.value - self.V_th_inf * last_spk
        last_I2 = self.I2.value - self.A2 * last_spk
        # 更新状态
        I2 = bst.nn.exp_euler_step(lambda i2: - i2 / self.tau_I2, last_I2)
        V = bst.nn.exp_euler_step(lambda v, Iext: (- v + self.V_rest + self.R * Iext) / self.tau,
                                  last_V, x + I2)
        self.I2.value = I2
        self.V.value = V
        # 输出
        inp = self.V.value - self.V_th_inf
        inp = jax.nn.standardize(u.get_magnitude(inp))
        return inp

    def get_spike(self, V=None):
        V = self.V.value if V is None else V
        spk = self.spk_fun((V - self.V_th_inf) / self.V_th_inf)
        return spk


class GifNet(bst.nn.Module):
    def __init__(
        self,
        n_in: int,
        n_rec: int,
        n_out: int,
        ff_scale: float = 1.,
        rec_scale: float = 1.,
        tau_neu: float = 5. * u.ms,
        tau_syn: float = 5. * u.ms,
        tau_I2: float = 5. * u.ms,
        A2=-1. * u.mA,
        tau_o: float = 5. * u.ms,
    ):
        super().__init__()

        # 初始化权重
        ff_init = bst.init.KaimingNormal(ff_scale, unit=u.mA)
        rec_init = bst.init.KaimingNormal(rec_scale, unit=u.mA)
        w = u.math.concatenate([ff_init((n_in, n_rec)), rec_init((n_rec, n_rec))], axis=0)

        # 参数
        self.n_in = n_in
        self.n_rec = n_rec
        self.n_out = n_out

        # 模型层
        self.ir2r = brainscale.nn.Linear(n_in + n_rec, n_rec, w_init=w, b_init=bst.init.ZeroInit(unit=u.mA))
        self.exp = brainscale.nn.Expon(n_rec, tau=tau_syn, g_initializer=bst.init.ZeroInit(unit=u.mA))
        self.r = GIF(
            n_rec,
            V_rest=0. * u.mV,
            V_th_inf=1. * u.mV,
            A2=A2,
            tau=tau_neu,
            tau_I2=bst.random.uniform(100. * u.ms, tau_I2 * 1.5, n_rec),
        )
        self.out = brainscale.nn.LeakyRateReadout(n_rec, n_out, tau=tau_o, w_init=bst.init.KaimingNormal())

    def update(self, spikes):
        cond = self.ir2r(u.math.concatenate([spikes, self.r.get_spike()], axis=-1))
        out = self.r(self.exp(cond))
        return self.out(out)

    def verify(self, input_spikes, num_show=5, sps_inc=10.):
        def _step(x):
            out = self.update(x)
            return out, self.r.get_spike(), self.r.V.value

        # 输入脉冲
        xs = np.transpose(input_spikes, (1, 0, 2))  # [n_steps, n_samples, n_in]

        # 运行仿真模型
        bst.nn.init_all_states(self, xs.shape[1])
        outs, sps, vs = bst.compile.for_loop(_step, xs)
        outs = u.math.as_numpy(outs)
        sps = u.math.as_numpy(sps)
        vs = u.math.as_numpy(vs.to_decimal(u.mV))
        vs = np.where(sps, vs + sps_inc, vs)
        max_t = xs.shape[0]

        for i in range(num_show):
            fig, gs = bts.visualize.get_figure(4, 1, 2., 10.)

            # 输入活动可视化
            ax_inp = fig.add_subplot(gs[0, 0])
            t_indices, n_indices = np.where(xs[:, i] > 0)
            ax_inp.plot(t_indices, n_indices, '.')
            ax_inp.set_xlim(0., max_t)
            ax_inp.set_ylabel('Input Activity')

            # 神经元活动可视化
            ax = fig.add_subplot(gs[1, 0])
            plt.plot(vs[:, i])
            ax.set_xlim(0., max_t)
            ax.set_ylabel('Recurrent Potential')

            # 脉冲活动可视化
            ax_rec = fig.add_subplot(gs[2, 0])
            t_indices, n_indices = np.where(sps[:, i] > 0)
            ax_rec.plot(t_indices, n_indices, '.')
            ax_rec.set_xlim(0., max_t)
            ax_rec.set_ylabel('Recurrent Spiking')

            # 输出活动可视化
            ax_out = fig.add_subplot(gs[3, 0])
            for j in range(outs.shape[-1]):
                ax_out.plot(outs[:, i, j], label=f'Readout {j}', alpha=0.7)
            ax_out.set_ylabel('Output Activity')
            ax_out.set_xlabel('Time [ms]')
            ax_out.set_xlim(0., max_t)
            plt.legend()

        plt.show()
        plt.close()


@numba.njit
def _dms(num_steps, num_inputs, n_motion_choice, motion_tuning,
         sample_time, test_time, fr, bg_fr, rotate_dir):
    # data
    X = np.zeros((num_steps, num_inputs))

    # sample
    match = np.random.randint(2)
    sample_dir = np.random.randint(n_motion_choice)

    # Generate the sample and test stimuli based on the rule
    if match == 1:  # match trial
        test_dir = (sample_dir + rotate_dir) % n_motion_choice
    else:
        test_dir = np.random.randint(n_motion_choice)
        while test_dir == ((sample_dir + rotate_dir) % n_motion_choice):
            test_dir = np.random.randint(n_motion_choice)

    # SAMPLE stimulus
    X[sample_time] += motion_tuning[sample_dir] * fr
    # TEST stimulus
    X[test_time] += motion_tuning[test_dir] * fr
    X += bg_fr

    # to spiking
    X = np.random.random(X.shape) < X
    X = X.astype(np.float32)

    # can use a greater weight for test period if needed
    return X, match


class DMSDataset:
    """
    Delayed match-to-sample task.
    """
    times = ('dead', 'fixation', 'sample', 'delay', 'test')
    output_features = ('non-match', 'match')

    _rotate_choice = {
        '0': 0,
        '45': 1,
        '90': 2,
        '135': 3,
        '180': 4,
        '225': 5,
        '270': 6,
        '315': 7,
        '360': 8,
    }

    def __init__(
        self,
        t_fixation=500. * u.ms,
        t_sample=500. * u.ms,
        t_delay=1000. * u.ms,
        t_test=500. * u.ms,
        limits=(0., np.pi * 2),
        rotation_match='0',
        kappa=3.,
        bg_fr=1. * u.Hz,
        n_input=100,
        firing_rate=100. * u.Hz,
        batch_size: int = 128,
        num_batch: int = 1000,
    ):
        super().__init__()

        # parameters
        self.num_batch = num_batch
        self.batch_size = batch_size
        self.num_inputs = n_input
        self.num_outputs = 2
        self.firing_rate = firing_rate
        dt = bst.environ.get_dt()

        # time
        self.t_fixation = int(t_fixation / dt)
        self.t_sample = int(t_sample / dt)
        self.t_delay = int(t_delay / dt)
        self.t_test = int(t_test / dt)
        self.num_steps = self.t_fixation + self.t_sample + self.t_delay + self.t_test
        test_onset = self.t_fixation + self.t_sample + self.t_delay
        self._test_onset = test_onset
        self.test_time = slice(test_onset, test_onset + self.t_test)
        self.fix_time = slice(0, test_onset)
        self.sample_time = slice(self.t_fixation, self.t_fixation + self.t_sample)

        # input shape
        self.rotation_match = rotation_match
        self._rotate = self._rotate_choice[rotation_match]
        self.bg_fr = bg_fr  # background firing rate
        self.v_min = limits[0]
        self.v_max = limits[1]
        self.v_range = limits[1] - limits[0]

        # Tuning function data
        self.n_motion_choice = 8
        self.kappa = kappa  # concentration scaling factor for von Mises

        # Generate list of preferred directions
        # dividing neurons by 2 since two equal
        # groups representing two modalities
        pref_dirs = np.arange(self.v_min, self.v_max, self.v_range / self.num_inputs)

        # Generate list of possible stimulus directions
        stim_dirs = np.arange(self.v_min, self.v_max, self.v_range / self.n_motion_choice)

        d = np.cos(np.expand_dims(stim_dirs, 1) - pref_dirs)
        self.motion_tuning = np.exp(self.kappa * d) / np.exp(self.kappa)

    @property
    def n_sim(self):
        return self._test_onset

    def __len__(self):
        return self.num_batch

    def __iter__(self):
        # firing rate
        fr = np.asarray(self.firing_rate * bst.environ.get_dt())
        bg_fr = np.asarray(self.bg_fr * bst.environ.get_dt())

        # generate data
        for _ in range(self.num_batch):
            xs, ys = [], []
            for _ in range(self.batch_size):
                x, y = _dms(self.num_steps,
                            self.num_inputs,
                            self.n_motion_choice,
                            self.motion_tuning,
                            self.sample_time,
                            self.test_time,
                            fr,
                            bg_fr,
                            self._rotate)
                xs.append(x)
                ys.append(y)
            yield np.asarray(xs), np.asarray(ys)


class Trainer(object):
    def __init__(
        self,
        target: GifNet,
        opt: bst.optim.Optimizer,
        dataset: Iterable,
        x_fun: Callable,
        n_sim: int = 0,
        batch_size: int = 128,
        acc_th: float = 0.90,
    ):
        super().__init__()

        # dataset
        self.dataset = dataset
        self.x_fun = x_fun

        # target network
        self.target = target

        # optimizer
        self.opt = opt
        weights = self.target.states().subset(bst.ParamState)
        opt.register_trainable_weights(weights)

        # training parameters
        self.n_sim = n_sim
        self.batch_size = batch_size
        self.acc_th = acc_th

    def _acc(self, out, target):
        return jax.numpy.mean(jax.numpy.equal(target, jax.numpy.argmax(jax.numpy.mean(out, axis=0), axis=1)))

    def batch_train(self, xs, ys) -> Union[LOSS, ACCURACY]:
        raise NotImplementedError

    def f_train(self):
        losses, accs = [], []
        n_batch = len(self.dataset)
        i_epoch = 0
        acc_ = 0.
        while acc_ < self.acc_th:
            i_epoch += 1
            bar = tqdm(enumerate(self.dataset))
            for i, (x_local, y_local) in bar:
                # training
                x_local = self.x_fun(x_local)  # [n_steps, n_samples, n_in]
                y_local = y_local  # [n_samples]
                loss, acc = self.batch_train(x_local, y_local)
                bar.set_description(f'loss = {loss:.5f}, acc={acc:.5f}', refresh=True)
                losses.append(loss)
                accs.append(acc)
            acc_ = np.mean(accs[-n_batch:])
            print(f'Epoch {i_epoch}, acc={acc_:.5f}, loss={np.mean(losses[-n_batch:]):.5f}')
        return np.asarray(losses), np.asarray(accs)


class OnlineTrainer(Trainer):
    def __init__(self, *args, decay_or_rank=0.99, **kwargs):
        super().__init__(*args, **kwargs)
        self.decay_or_rank = decay_or_rank

    @bst.compile.jit(static_argnums=(0,))
    def batch_train(self, inputs, targets):
        # initialize the states
        bst.nn.init_all_states(self.target, inputs.shape[1])

        # weights
        weights = self.target.states().subset(bst.ParamState)

        # initialize the online learning model
        # model = brainscale.DiagParamDimAlgorithm(self.target, mode=bst.mixin.Batching())
        model = brainscale.DiagIODimAlgorithm(self.target, self.decay_or_rank)
        model.compile_graph(inputs[0])

        def _etrace_grad(i, inp):
            # call the model
            out = model(inp, running_index=i)
            # calculate the loss
            loss = bts.metric.softmax_cross_entropy_with_integer_labels(out, targets).mean()
            return loss, out

        def _etrace_step(prev_grads, x):
            # no need to return weights and states, since they are generated then no longer needed
            i, inp = x
            f_grad = bst.augment.grad(_etrace_grad, weights, has_aux=True, return_value=True)
            cur_grads, local_loss, out = f_grad(i, inp)
            next_grads = jax.tree.map(lambda a, b: a + b, prev_grads, cur_grads)
            return next_grads, (out, local_loss)

        def _etrace_train(indices_, inputs_):
            # forward propagation
            grads = jax.tree.map(u.math.zeros_like, weights.to_dict_values())
            grads, (outs, losses) = bst.compile.scan(_etrace_step, grads, (indices_, inputs_))
            # gradient updates
            grads = bst.functional.clip_grad_norm(grads, 1.)
            self.opt.update(grads)
            # accuracy
            return losses.mean(), outs

        # running indices
        indices = np.arange(inputs.shape[0])
        if self.n_sim > 0:
            bst.compile.for_loop(lambda i, inp: model(inp, running_index=i), indices[:self.n_sim], inputs[:self.n_sim])
        loss, outs = _etrace_train(indices[self.n_sim:], inputs[self.n_sim:])

        # returns
        return loss, self._acc(outs, targets)


class BPTTTrainer(Trainer):
    @bst.compile.jit(static_argnums=(0,))
    def bptt_train(self, inputs, targets):
        # initialize the states
        bst.nn.init_all_states(self.target, inputs.shape[1])

        # the model for a single step
        def _run_step_train(inp):
            out = self.target(inp)
            loss = bts.metric.softmax_cross_entropy_with_integer_labels(out, targets).mean()
            return out, loss

        def _bptt_grad_step():
            if self.n_sim > 0:
                _ = bst.compile.for_loop(self.target, inputs[:self.n_sim])
            outs, losses = bst.compile.for_loop(_run_step_train, inputs[self.n_sim:])
            return losses.mean(), outs

        # gradients
        weights = self.target.states().subset(bst.ParamState)
        grads, loss, outs = bst.augment.grad(_bptt_grad_step, weights, has_aux=True, return_value=True)()

        # optimization
        grads = bst.functional.clip_grad_norm(grads, 1.)
        self.opt.update(grads)

        return loss, self._acc(outs, targets)


class LIF_Delta_Net(bst.nn.Module):
    """
    LIF neurons and dense connected delta synapses.
    """

    def __init__(
        self,
        n_in, n_rec, n_out,
        tau_mem=5. * u.ms,
        tau_o=5. * u.ms,
        V_th=1. * u.mV,
        spk_fun: Callable = bst.surrogate.ReluGrad(),
        spk_reset: str = 'soft',
        rec_scale: float = 1.,
        ff_scale: float = 1.,
    ):
        super().__init__()
        self.neu = brainscale.nn.LIF(n_rec, tau=tau_mem, spk_fun=spk_fun, spk_reset=spk_reset, V_th=V_th)
        rec_init: Callable = bst.init.KaimingNormal(rec_scale, unit=u.mV)
        ff_init: Callable = bst.init.KaimingNormal(ff_scale, unit=u.mV)
        w_init = u.math.concatenate([ff_init([n_in, n_rec]), rec_init([n_rec, n_rec])], axis=0)
        self.syn = bst.nn.DeltaProj(
            comm=brainscale.nn.Linear(n_in + n_rec, n_rec, w_init=w_init, b_init=bst.init.ZeroInit(unit=u.mV)),
            post=self.neu
        )
        self.out = brainscale.nn.LeakyRateReadout(in_size=n_rec, out_size=n_out, tau=tau_o,
                                                  w_init=bst.init.KaimingNormal())

    def update(self, spk):
        self.syn(u.math.concatenate([spk, self.neu.get_spike()], axis=-1))
        return self.out(self.neu())

    def verify(self, input_spikes, num_show=5, sps_inc=10.):
        def _step(x):
            out = self.update(x)
            return out, self.neu.get_spike(), self.neu.V.value

        # 输入脉冲
        xs = np.transpose(input_spikes.reshape(*input_spikes.shape[:2], -1), (1, 0, 2))  # [n_steps, n_samples, n_in]

        # 运行仿真模型
        bst.nn.init_all_states(self, xs.shape[1])
        outs, sps, vs = bst.compile.for_loop(_step, xs)
        outs = u.math.as_numpy(outs)
        sps = u.math.as_numpy(sps)
        vs = u.math.as_numpy(vs.to_decimal(u.mV))
        vs = np.where(sps, vs + sps_inc, vs)
        max_t = xs.shape[0]

        for i in range(num_show):
            fig, gs = bts.visualize.get_figure(4, 1, 2., 10.)

            # 输入活动可视化
            ax_inp = fig.add_subplot(gs[0, 0])
            t_indices, n_indices = np.where(xs[:, i] > 0)
            ax_inp.plot(t_indices, n_indices, '.')
            ax_inp.set_xlim(0., max_t)
            ax_inp.set_ylabel('Input Activity')

            # 神经元活动可视化
            ax = fig.add_subplot(gs[1, 0])
            plt.plot(vs[:, i])
            ax.set_xlim(0., max_t)
            ax.set_ylabel('Recurrent Potential')

            # 脉冲活动可视化
            ax_rec = fig.add_subplot(gs[2, 0])
            t_indices, n_indices = np.where(sps[:, i] > 0)
            ax_rec.plot(t_indices, n_indices, '.')
            ax_rec.set_xlim(0., max_t)
            ax_rec.set_ylabel('Recurrent Spiking')

            # 输出活动可视化
            ax_out = fig.add_subplot(gs[3, 0])
            for j in range(outs.shape[-1]):
                ax_out.plot(outs[:, i, j], label=f'Readout {j}', alpha=0.7)
            ax_out.set_ylabel('Output Activity')
            ax_out.set_xlabel('Time [ms]')
            ax_out.set_xlim(0., max_t)
            plt.legend()

        plt.show()
        plt.close()
