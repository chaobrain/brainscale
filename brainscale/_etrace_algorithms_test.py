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

import os

os.environ['JAX_TRACEBACK_FILTERING'] = 'off'

import unittest
from functools import partial, reduce

import brainstate as bst
import brainunit as u
import jax
import jax.numpy as jnp

from pprint import pprint
import brainscale
from brainscale._etrace_model_test import (
    IF_Delta_Dense_Layer,
    LIF_ExpCo_Dense_Layer,
    ALIF_ExpCo_Dense_Layer,
    LIF_ExpCu_Dense_Layer,
    LIF_STDExpCu_Dense_Layer,
    LIF_STPExpCu_Dense_Layer,
    ALIF_ExpCu_Dense_Layer,
    ALIF_Delta_Dense_Layer,
    ALIF_STDExpCu_Dense_Layer,
    ALIF_STPExpCu_Dense_Layer,
)


class TestDiagOn(unittest.TestCase):

    def test_rnn_no_bptt(self):
        for cls in [
            brainscale.nn.GRUCell,
            brainscale.nn.LSTMCell,
            brainscale.nn.LRUCell,
            brainscale.nn.MGUCell,
            brainscale.nn.MinimalRNNCell,
        ]:
            n_in = 4
            n_rec = 5
            n_seq = 10
            model = cls(n_in, n_rec)
            model = bst.nn.init_all_states(model)

            inputs = bst.random.randn(n_seq, n_in)
            algorithm = brainscale.DiagIODimAlgorithm(model, decay_or_rank=0.9)
            algorithm.compile_graph(inputs[0])

            outs = bst.compile.for_loop(algorithm, inputs)
            print(outs.shape)

            @bst.compile.jit
            def grad_no_bptt(inp):
                return bst.augment.grad(
                    lambda inp: algorithm(inp).sum(),
                    model.states(bst.ParamState)
                )(inp)

            grads = grad_no_bptt(inputs[0])
            grads = grad_no_bptt(inputs[1])
            pprint(grads)

    def test_rnn_has_bptt(self):
        for cls in [
            brainscale.nn.GRUCell,
            brainscale.nn.LSTMCell,
            brainscale.nn.LRUCell,
            brainscale.nn.MGUCell,
            brainscale.nn.MinimalRNNCell,
        ]:
            n_in = 4
            n_rec = 5
            n_seq = 10
            model = cls(n_in, n_rec)
            model = bst.nn.init_all_states(model)

            inputs = bst.random.randn(n_seq, n_in)
            algorithm = brainscale.DiagIODimAlgorithm(model, decay_or_rank=0.9)
            algorithm.compile_graph(inputs[0], multi_step=True)

            outs = algorithm(inputs, multi_step=True)
            print(outs.shape)

            @bst.compile.jit
            def grad_no_bptt(inp):
                return bst.augment.grad(
                    lambda inp: algorithm(inp, multi_step=True).sum(),
                    model.states(bst.ParamState)
                )(inp)

            grads = grad_no_bptt(inputs[:1])
            pprint(grads)
            print()
            grads = grad_no_bptt(inputs[1:2])
            pprint(grads)

    def test_snn_no_bptt(self):
        bst.environ.set(dt=0.1 * u.ms)

        for cls in [
            IF_Delta_Dense_Layer,
            LIF_ExpCo_Dense_Layer,
            ALIF_ExpCo_Dense_Layer,
            LIF_ExpCu_Dense_Layer,
            LIF_STDExpCu_Dense_Layer,
            LIF_STPExpCu_Dense_Layer,
            ALIF_ExpCu_Dense_Layer,
            ALIF_Delta_Dense_Layer,
            ALIF_STDExpCu_Dense_Layer,
            ALIF_STPExpCu_Dense_Layer,
        ]:
            print(cls)

            n_in = 4
            n_rec = 5
            n_seq = 10
            model = cls(n_in, n_rec)
            model = bst.nn.init_all_states(model)

            inputs = bst.random.randn(n_seq, n_in)
            algorithm = brainscale.DiagIODimAlgorithm(model, decay_or_rank=0.9)
            algorithm.compile_graph(inputs[0])

            outs = bst.compile.for_loop(algorithm, inputs)
            print(outs.shape)

            @bst.compile.jit
            def grad_no_bptt(inp):
                return bst.augment.grad(
                    lambda inp: algorithm(inp).sum(),
                    model.states(bst.ParamState)
                )(inp)

            grads = grad_no_bptt(inputs[0])
            grads = grad_no_bptt(inputs[1])
            pprint(grads)

    def test_snn_has_bptt(self):
        bst.environ.set(dt=0.1 * u.ms)

        for cls in [
            IF_Delta_Dense_Layer,
            LIF_ExpCo_Dense_Layer,
            ALIF_ExpCo_Dense_Layer,
            LIF_ExpCu_Dense_Layer,
            LIF_STDExpCu_Dense_Layer,
            LIF_STPExpCu_Dense_Layer,
            ALIF_ExpCu_Dense_Layer,
            ALIF_Delta_Dense_Layer,
            ALIF_STDExpCu_Dense_Layer,
            ALIF_STPExpCu_Dense_Layer,
        ]:
            print(cls)

            n_in = 4
            n_rec = 5
            n_seq = 10
            model = cls(n_in, n_rec)
            model = bst.nn.init_all_states(model)

            inputs = bst.random.randn(n_seq, n_in)
            algorithm = brainscale.DiagIODimAlgorithm(model, decay_or_rank=0.9)
            algorithm.compile_graph(inputs[0], multi_step=True)

            outs = algorithm(inputs, multi_step=True)
            print(outs.shape)

            @bst.compile.jit
            def grad_no_bptt(inp):
                return bst.augment.grad(
                    lambda inp: algorithm(inp, multi_step=True).sum(),
                    model.states(bst.ParamState)
                )(inp)

            grads = grad_no_bptt(inputs[:1])
            pprint(grads)
            print()
            grads = grad_no_bptt(inputs[1:2])
            pprint(grads)


class TestDiagOn2(unittest.TestCase):

    def test_rnn_no_bptt(self):
        for cls in [
            brainscale.nn.GRUCell,
            brainscale.nn.LSTMCell,
            brainscale.nn.LRUCell,
            brainscale.nn.MGUCell,
            brainscale.nn.MinimalRNNCell,
        ]:
            n_in = 4
            n_rec = 5
            n_seq = 10
            model = cls(n_in, n_rec)
            model = bst.nn.init_all_states(model)

            inputs = bst.random.randn(n_seq, n_in)
            algorithm = brainscale.DiagParamDimAlgorithm(model)
            algorithm.compile_graph(inputs[0])

            outs = bst.compile.for_loop(algorithm, inputs)
            print(outs.shape)

            @bst.compile.jit
            def grad_no_bptt(inp):
                return bst.augment.grad(
                    lambda inp: algorithm(inp).sum(),
                    model.states(bst.ParamState)
                )(inp)

            grads = grad_no_bptt(inputs[0])
            grads = grad_no_bptt(inputs[1])
            pprint(grads)

    def test_rnn_has_bptt(self):
        for cls in [
            brainscale.nn.GRUCell,
            brainscale.nn.LSTMCell,
            brainscale.nn.LRUCell,
            brainscale.nn.MGUCell,
            brainscale.nn.MinimalRNNCell,
        ]:
            n_in = 4
            n_rec = 5
            n_seq = 10
            model = cls(n_in, n_rec)
            model = bst.nn.init_all_states(model)

            inputs = bst.random.randn(n_seq, n_in)
            algorithm = brainscale.DiagParamDimAlgorithm(model)
            algorithm.compile_graph(inputs[0], multi_step=True)

            outs = algorithm(inputs, multi_step=True)
            print(outs.shape)

            @bst.compile.jit
            def grad_no_bptt(inp):
                return bst.augment.grad(
                    lambda inp: algorithm(inp, multi_step=True).sum(),
                    model.states(bst.ParamState)
                )(inp)

            grads = grad_no_bptt(inputs[:1])
            pprint(grads)
            print()
            grads = grad_no_bptt(inputs[1:2])
            pprint(grads)

    def test_snn_no_bptt(self):
        bst.environ.set(dt=0.1 * u.ms)

        for cls in [
            IF_Delta_Dense_Layer,
            LIF_ExpCo_Dense_Layer,
            ALIF_ExpCo_Dense_Layer,
            LIF_ExpCu_Dense_Layer,
            LIF_STDExpCu_Dense_Layer,
            LIF_STPExpCu_Dense_Layer,
            ALIF_ExpCu_Dense_Layer,
            ALIF_Delta_Dense_Layer,
            ALIF_STDExpCu_Dense_Layer,
            ALIF_STPExpCu_Dense_Layer,
        ]:
            print(cls)

            n_in = 4
            n_rec = 5
            n_seq = 10
            model = cls(n_in, n_rec)
            model = bst.nn.init_all_states(model)

            param_states = model.states(bst.ParamState).to_dict_values()

            inputs = bst.random.randn(n_seq, n_in)
            algorithm = brainscale.DiagParamDimAlgorithm(model)
            algorithm.compile_graph(inputs[0])

            outs = bst.compile.for_loop(algorithm, inputs)
            print(outs.shape)

            @bst.compile.jit
            def grad_no_bptt(inp):
                return bst.augment.grad(
                    lambda inp: algorithm(inp).sum(),
                    model.states(bst.ParamState)
                )(inp)

            grads = grad_no_bptt(inputs[0])
            grads = grad_no_bptt(inputs[1])
            pprint(grads)

            for k in grads:
                assert u.get_unit(param_states[k]) == u.get_unit(grads[k])

    def test_snn_has_bptt(self):
        bst.environ.set(dt=0.1 * u.ms)

        for cls in [
            IF_Delta_Dense_Layer,
            LIF_ExpCo_Dense_Layer,
            ALIF_ExpCo_Dense_Layer,
            LIF_ExpCu_Dense_Layer,
            LIF_STDExpCu_Dense_Layer,
            LIF_STPExpCu_Dense_Layer,
            ALIF_ExpCu_Dense_Layer,
            ALIF_Delta_Dense_Layer,
            ALIF_STDExpCu_Dense_Layer,
            ALIF_STPExpCu_Dense_Layer,
        ]:
            print(cls)

            n_in = 4
            n_rec = 5
            n_seq = 10
            model = cls(n_in, n_rec)
            model = bst.nn.init_all_states(model)

            param_states = model.states(bst.ParamState).to_dict_values()

            inputs = bst.random.randn(n_seq, n_in)
            algorithm = brainscale.DiagParamDimAlgorithm(model)
            algorithm.compile_graph(inputs[0], multi_step=True)

            outs = algorithm(inputs, multi_step=True)
            print(outs.shape)

            @bst.compile.jit
            def grad_no_bptt(inp):
                return bst.augment.grad(
                    lambda inp: algorithm(inp, multi_step=True).sum(),
                    model.states(bst.ParamState)
                )(inp)

            grads = grad_no_bptt(inputs[:1])
            pprint(grads)
            print()
            grads = grad_no_bptt(inputs[1:2])
            pprint(grads)

            for k in grads:
                assert u.get_unit(param_states[k]) == u.get_unit(grads[k])



class TestDiagGrad(unittest.TestCase):
    def _hidden_to_hidden(self, model, other_vals, pre_hidden_vals, inputs, out_hidden: str = None):
        etrace_vars, other_states = model.states().split(brainscale.ETraceState)
        etrace_vars.assign_values(pre_hidden_vals)
        other_states.assign_values(other_vals)
        with brainscale.stop_param_gradients():
            model(inputs)
        if out_hidden is not None:
            return etrace_vars[out_hidden].value
        else:
            return etrace_vars.to_dict_values()

    def _whether_collective_and_independent_are_same(self, model: bst.nn.Module, inputs):
        states = model.states()
        state_vals = states.to_dict_values()
        etrace_vars, other_states = states.split(brainscale.ETraceState)
        etrace_grads = jax.tree.map(bst.random.randn_like, etrace_vars.to_dict_values())

        # collective solve
        fun = partial(self._hidden_to_hidden, model, other_states.to_dict_values(), inputs=inputs)
        _, new_grads1 = jax.jvp(fun, (etrace_vars.to_dict_values(),), (etrace_grads,))
        states.assign_values(state_vals)

        # independent solve
        new_grads2 = dict()
        for key in etrace_vars:
            fun = partial(self._hidden_to_hidden, model, other_states.to_dict_values(), inputs=inputs, out_hidden=key)
            jac = bst.augment.vector_grad(fun, argnums=0)(etrace_vars.to_dict_values())
            states.assign_values(state_vals)

            leaves = jax.tree.leaves(jax.tree.map(jnp.multiply, jac, etrace_grads))
            grad = reduce(jnp.add, leaves)
            new_grads2[key] = grad
            self.assertTrue(jnp.allclose(grad, new_grads1[key], atol=1e-5), f"key: {key}")

    def test1(self):
        bst.environ.set(mode=bst.mixin.JointMode(bst.mixin.Training(), bst.mixin.Batching()))
        bst.environ.set(i=0, t=0.)

        n_batch, n_in, n_rec = 16, 4, 10
        std_model = LIF_STDExpCu_Dense_Layer(n_in, n_rec, inp_std=False)
        stp_model = LIF_STPExpCu_Dense_Layer(n_in, n_rec, inp_stp=False)

        bst.nn.init_all_states(std_model, n_batch)
        std_model.neu.V.value = bst.random.uniform(-1., 1.5, std_model.neu.V.value.shape)
        bst.nn.init_all_states(stp_model, n_batch)
        stp_model.neu.V.value = bst.random.uniform(-1., 1.5, stp_model.neu.V.value.shape)

        inputs = jnp.asarray(bst.random.rand(n_batch, n_in) < 0.2, dtype=bst.environ.dftype())

        self._whether_collective_and_independent_are_same(std_model, inputs)
        self._whether_collective_and_independent_are_same(stp_model, inputs)
