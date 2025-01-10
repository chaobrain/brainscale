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
            algorithm = brainscale.DiagIODimAlgorithm(model, decay_or_rank=0.9, vjp_method='multi-step')
            algorithm.compile_graph(inputs[0])

            outs = algorithm(brainscale.MultiStepData(inputs))
            print(outs.shape)

            @bst.compile.jit
            def grad_no_bptt(inp):
                return bst.augment.grad(
                    lambda inp: algorithm(brainscale.MultiStepData(inp)).sum(),
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
            algorithm = brainscale.DiagIODimAlgorithm(model, decay_or_rank=0.9, vjp_method='multi-step')
            algorithm.compile_graph(inputs[0])

            outs = algorithm(brainscale.MultiStepData(inputs))
            print(outs.shape)

            @bst.compile.jit
            def grad_no_bptt(inp):
                return bst.augment.grad(
                    lambda inp: algorithm(brainscale.MultiStepData(inp)).sum(),
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
            algorithm = brainscale.DiagParamDimAlgorithm(model, vjp_method='multi-step')
            algorithm.compile_graph(inputs[0])

            outs = algorithm(brainscale.MultiStepData(inputs))
            print(outs.shape)

            @bst.compile.jit
            def grad_no_bptt(inp):
                return bst.augment.grad(
                    lambda inp: algorithm(brainscale.MultiStepData(inp)).sum(),
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
            algorithm = brainscale.DiagParamDimAlgorithm(model, vjp_method='multi-step')
            algorithm.compile_graph(inputs[0])

            outs = algorithm(brainscale.MultiStepData(inputs))
            print(outs.shape)

            @bst.compile.jit
            def grad_no_bptt(inp):
                return bst.augment.grad(
                    lambda inp: algorithm(brainscale.MultiStepData(inp)).sum(),
                    model.states(bst.ParamState)
                )(inp)

            grads = grad_no_bptt(inputs[:1])
            pprint(grads)
            print()
            grads = grad_no_bptt(inputs[1:2])
            pprint(grads)

            for k in grads:
                assert u.get_unit(param_states[k]) == u.get_unit(grads[k])

