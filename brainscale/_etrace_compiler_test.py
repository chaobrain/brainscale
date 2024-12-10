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


import unittest
from pprint import pprint

import brainstate as bst
import brainunit as u

import brainscale
from brainscale._etrace_model_test import (
    _IF_Delta_Dense_Layer,
    _LIF_ExpCo_Dense_Layer,
    _ALIF_ExpCo_Dense_Layer,
    _LIF_ExpCu_Dense_Layer,
    _LIF_STDExpCu_Dense_Layer,
    _LIF_STPExpCu_Dense_Layer,
    _ALIF_ExpCu_Dense_Layer,
    _ALIF_Delta_Dense_Layer,
    _ALIF_STDExpCu_Dense_Layer,
    _ALIF_STPExpCu_Dense_Layer,
)

pprint


class TestCompileGraphRNN(unittest.TestCase):
    def test_gru_one_layer(self):
        n_in = 3
        n_out = 4

        gru = brainscale.nn.GRUCell(n_in, n_out)
        bst.nn.init_all_states(gru)

        input = bst.random.rand(n_in)
        graph = brainscale.compile_graph(gru, False, input)

        self.assertTrue(isinstance(graph, brainscale.CompiledGraph))
        self.assertTrue(graph.num_out == 1)
        self.assertTrue(len(graph.stateful_fn_states) == 4)
        self.assertTrue(len(graph.hidden_groups) == 1)

        param_states = gru.states(bst.ParamState)
        self.assertTrue(len(param_states) == 3)
        self.assertTrue(len(graph.hidden_param_op_relations) == 2)

        # pprint(graph)

    def test_lru_one_layer(self):
        n_in = 3
        n_out = 4

        lru = brainscale.nn.LRUCell(n_in, n_out)
        bst.nn.init_all_states(lru)

        input = bst.random.rand(n_in)
        graph = brainscale.compile_graph(lru, False, input)

        self.assertTrue(len(graph.hidden_groups) == 1)
        self.assertTrue(len(graph.hidden_groups[0].hidden_paths) == 2)

        for relation in graph.hidden_param_op_relations:
            if relation.path[0] in ['C_re', 'C_im', 'D']:
                self.assertTrue(len(relation.hidden_paths) == 0)

        # pprint(graph)

    def test_lstm_one_layer(self):
        n_in = 3
        n_out = 4

        lstm = brainscale.nn.LSTMCell(n_in, n_out)
        bst.nn.init_all_states(lstm)

        input = bst.random.rand(n_in)
        graph = brainscale.compile_graph(lstm, False, input)

        self.assertTrue(isinstance(graph, brainscale.CompiledGraph))
        self.assertTrue(graph.num_out == 1)
        self.assertTrue(len(graph.hidden_groups) == 1)
        self.assertTrue(len(graph.hidden_groups[0].hidden_paths) == 2)
        self.assertTrue(len(graph.stateful_fn_states) == 6)

        hid_states = lstm.states(brainscale.ETraceState)
        self.assertTrue(len(hid_states) == len(graph.hid_path_to_transition))

        param_states = lstm.states(bst.ParamState)
        self.assertTrue(len(param_states) == len(graph.hidden_param_op_relations))

        hidden_paths = set(graph.hidden_groups[0].hidden_paths)
        for relation in graph.hidden_param_op_relations:
            if relation.path[0] == 'Wo':
                self.assertTrue(set(relation.hidden_paths) == set([('h',)]))
            else:
                self.assertTrue(set(relation.hidden_paths) == hidden_paths)

        # pprint(graph)

    def test_lstm_two_layers(self):
        n_in = 3
        n_out = 4

        net = bst.nn.Sequential(
            brainscale.nn.LSTMCell(n_in, n_out),
            bst.nn.ReLU(),
            brainscale.nn.LSTMCell(n_out, n_in),
        )
        bst.nn.init_all_states(net)

        input = bst.random.rand(n_in)
        graph = brainscale.compile_graph(net, False, input)

        self.assertTrue(isinstance(graph, brainscale.CompiledGraph))
        self.assertTrue(graph.num_out == 1)
        self.assertTrue(len(graph.hidden_groups) == 2)
        self.assertTrue(len(graph.hidden_groups[0].hidden_paths) == 2)
        self.assertTrue(len(graph.hidden_groups[1].hidden_paths) == 2)

        hidden_group1_path = {('layers', 0, 'c'), ('layers', 0, 'h')}
        hidden_group2_path = {('layers', 2, 'c'), ('layers', 2, 'h')}

        for relation in graph.hidden_param_op_relations:
            if relation.path[1] == 0:
                if relation.path[2] != 'Wo':
                    self.assertTrue(set(relation.hidden_paths) == hidden_group1_path)
            if relation.path[1] == 2:
                if relation.path[2] != 'Wo':
                    self.assertTrue(set(relation.hidden_paths) == hidden_group2_path)

        # pprint(graph)

    def test_lru_two_layers(self):
        n_in = 3
        n_out = 4

        net = bst.nn.Sequential(
            brainscale.nn.LRUCell(n_in, n_out),
            bst.nn.ReLU(),
            brainscale.nn.LRUCell(n_in, n_out),
        )
        bst.nn.init_all_states(net)

        input = bst.random.rand(n_in)
        graph = brainscale.compile_graph(net, False, input)

        self.assertTrue(len(graph.hidden_groups) == 2)
        self.assertTrue(len(graph.hidden_groups[0].hidden_paths) == 2)
        self.assertTrue(len(graph.hidden_groups[1].hidden_paths) == 2)
        self.assertTrue(len(graph.hidden_param_op_relations) == 10)

        layer1_hiddens = {('layers', 0, 'h_im'), ('layers', 0, 'h_re')}
        layer2_hiddens = {('layers', 2, 'h_im'), ('layers', 2, 'h_re')}

        for relation in graph.hidden_param_op_relations:
            if relation.path[1] == 0 and relation.path[2] not in ['B_im', 'B_re']:
                self.assertTrue(set(relation.hidden_paths) == layer1_hiddens)
            if relation.path[1] == 2 and relation.path[2] not in ['B_im', 'B_re']:
                self.assertTrue(set(relation.hidden_paths) == layer2_hiddens)

    def test_lru_two_layers_v2(self):
        n_in = 4
        n_out = 4

        net = bst.nn.Sequential(
            brainscale.nn.LRUCell(n_in, n_out),
            bst.nn.ReLU(),
            brainscale.nn.LRUCell(n_in, n_out),
        )
        bst.nn.init_all_states(net)

        input = bst.random.rand(n_in)
        graph = brainscale.compile_graph(net, False, input)

        self.assertTrue(len(graph.hidden_groups) == 2)
        self.assertTrue(len(graph.hidden_groups[0].hidden_paths) == 2)
        self.assertTrue(len(graph.hidden_groups[1].hidden_paths) == 2)
        self.assertTrue(len(graph.hidden_param_op_relations) == 10)

        layer1_hiddens = {('layers', 0, 'h_im'), ('layers', 0, 'h_re')}
        layer2_hiddens = {('layers', 2, 'h_im'), ('layers', 2, 'h_re')}

        for relation in graph.hidden_param_op_relations:
            if relation.path[1] == 0 and relation.path[2] not in ['B_im', 'B_re']:
                self.assertTrue(set(relation.hidden_paths) == layer1_hiddens)
            if relation.path[1] == 2 and relation.path[2] not in ['B_im', 'B_re']:
                self.assertTrue(set(relation.hidden_paths) == layer2_hiddens)


class TestCompileGraphSNN(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        bst.environ.set(dt=0.1 * u.ms)

    def test_if_delta_dense(self):
        n_in = 3
        n_rec = 4

        net = _IF_Delta_Dense_Layer(n_in, n_rec)
        bst.nn.init_all_states(net)

        input = bst.random.rand(n_in)
        graph = brainscale.compile_graph(net, False, input)

        pprint(graph)
        pass

    def test_lif_expco_dense_layer(self):
        n_in = 3
        n_rec = 4

        net = _LIF_ExpCo_Dense_Layer(n_in, n_rec)
        bst.nn.init_all_states(net)

        input = bst.random.rand(n_in)
        graph = brainscale.compile_graph(net, False, input)

        pprint(graph)

    def test_alif_expco_dense_layer(self):
        n_in = 3
        n_rec = 4

        net = _ALIF_ExpCo_Dense_Layer(n_in, n_rec)
        bst.nn.init_all_states(net)

        input = bst.random.rand(n_in)
        graph = brainscale.compile_graph(net, False, input)

        pprint(graph)

    def test_lif_expcu_dense_layer(self):
        n_in = 3
        n_rec = 4

        net = _LIF_ExpCu_Dense_Layer(n_in, n_rec)
        bst.nn.init_all_states(net)

        input = bst.random.rand(n_in)
        graph = brainscale.compile_graph(net, False, input)

        pprint(graph)

    def test_lif_std_expcu_dense_layer(self):
        n_in = 3
        n_rec = 4

        net = _LIF_STDExpCu_Dense_Layer(n_in, n_rec)
        bst.nn.init_all_states(net)

        input = bst.random.rand(n_in)
        graph = brainscale.compile_graph(net, False, input)

        pprint(graph)

    def test_lif_stp_expcu_dense_layer(self):
        n_in = 3
        n_rec = 4

        net = _LIF_STPExpCu_Dense_Layer(n_in, n_rec)
        bst.nn.init_all_states(net)

        input = bst.random.rand(n_in)
        graph = brainscale.compile_graph(net, False, input)

        pprint(graph)

    def test_alif_expcu_dense_layer(self):
        n_in = 3
        n_rec = 4

        net = _ALIF_ExpCu_Dense_Layer(n_in, n_rec)
        bst.nn.init_all_states(net)

        input = bst.random.rand(n_in)
        graph = brainscale.compile_graph(net, False, input)

        pprint(graph)

    def test_alif_delta_dense_layer(self):
        n_in = 3
        n_rec = 4

        net = _ALIF_Delta_Dense_Layer(n_in, n_rec)
        bst.nn.init_all_states(net)

        input = bst.random.rand(n_in)
        graph = brainscale.compile_graph(net, False, input)

        pprint(graph)

    def test_alif_std_expcu_dense_layer(self):
        n_in = 3
        n_rec = 4

        net = _ALIF_STDExpCu_Dense_Layer(n_in, n_rec)
        bst.nn.init_all_states(net)

        input = bst.random.rand(n_in)
        graph = brainscale.compile_graph(net, False, input)

        pprint(graph)

    def test_alif_stp_expcu_dense_layer(self):
        n_in = 3
        n_rec = 4

        net = _ALIF_STPExpCu_Dense_Layer(n_in, n_rec)
        bst.nn.init_all_states(net)

        input = bst.random.rand(n_in)
        graph = brainscale.compile_graph(net, False, input)

        pprint(graph)
