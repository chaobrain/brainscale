# Copyright 2025 BDP Ecosystem Limited. All Rights Reserved.
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

import brainunit as u
import jax
import pytest

import brainscale
import brainstate as bst
from brainscale._etrace_compiler_hidden_group import find_hidden_groups_from_module
from brainscale._etrace_compiler_hidden_group import group_merging
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


class TestGroupMerging(unittest.TestCase):
    def test_no_intersection(self):
        groups = [[1, 2], [3, 4], [5, 6]]
        expected = [frozenset([1, 2]),
                    frozenset([3, 4]),
                    frozenset([5, 6])]
        result = group_merging(groups, version=1)
        self.assertEqual(set(result), set(expected))
        result = group_merging(groups, version=0)
        self.assertEqual(set(result), set(expected))

    def test_single_intersection(self):
        groups = [[1, 2], [2, 3], [4, 5]]
        expected = [frozenset([1, 2, 3]),
                    frozenset([4, 5])]
        result = group_merging(groups, version=1)
        self.assertEqual(set(result), set(expected))
        result = group_merging(groups, version=0)
        self.assertEqual(set(result), set(expected))

    def test_single_intersection3(self):
        groups = [[1, 2], [1, 2], [2, 3], [4, 5]]
        expected = [frozenset([1, 2, 3]),
                    frozenset([4, 5])]
        result = group_merging(groups, version=1)
        self.assertEqual(set(result), set(expected))
        result = group_merging(groups, version=0)
        self.assertEqual(set(result), set(expected))

    def test_single_intersection2(self):
        groups = [
            [('neu', 'a'), ('neu', 'V')],
            [('neu', 'V'), ('neu', '_before_updates', 'syn', 'g')]
        ]

        expected = [frozenset({('neu', 'a'), ('neu', '_before_updates', 'syn', 'g'), ('neu', 'V')})]
        result = group_merging(groups, version=1)
        print(result)
        self.assertEqual(set(result), set(expected))
        result = group_merging(groups, version=0)
        print(result)
        self.assertEqual(set(result), set(expected))

    def test_multiple_intersections(self):
        groups = [[1, 2], [2, 3], [3, 4], [5, 6]]
        expected = [frozenset([1, 2, 3, 4]),
                    frozenset([5, 6])]
        result = group_merging(groups, version=1)
        self.assertEqual(set(result), set(expected))
        result = group_merging(groups, version=0)
        self.assertEqual(set(result), set(expected))

    def test_all_intersect(self):
        groups = [[1, 2], [2, 3], [3, 4], [4, 1]]
        expected = [frozenset([1, 2, 3, 4])]
        result = group_merging(groups, version=0)
        self.assertEqual(set(result), set(expected))
        result = group_merging(groups, version=1)
        self.assertEqual(set(result), set(expected))

    def test_empty_groups(self):
        groups = []
        expected = []
        result = group_merging(groups, version=0)
        self.assertEqual(result, expected)
        result = group_merging(groups, version=1)
        self.assertEqual(result, expected)

    def test_single_group(self):
        groups = [[1, 2, 3]]
        expected = [frozenset([1, 2, 3])]
        result = group_merging(groups, version=0)
        self.assertEqual(set(result), set(expected))
        result = group_merging(groups, version=1)
        self.assertEqual(set(result), set(expected))


class TestFindHiddenGroupsFromModule:
    def test_gru_one_layer(self):
        n_in = 3
        n_out = 4

        gru = brainscale.nn.GRUCell(n_in, n_out)
        bst.nn.init_all_states(gru)

        input = bst.random.rand(n_in)
        hidden_groups, hid_path_to_group = find_hidden_groups_from_module(gru, input)

        print()
        pprint(hidden_groups)
        print()
        pprint(hid_path_to_group)
        print()

    @classmethod
    @pytest.mark.parametrize(
        'cls,',
        [
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
        ]
    )
    def test_snn_single_layer(self, cls):
        n_in = 3
        n_out = 4
        input = bst.random.rand(n_in)

        print(cls)

        with bst.environ.context(dt=0.1 * u.ms):
            layer = cls(n_in, n_out)
            bst.nn.init_all_states(layer)
            hidden_groups, hid_path_to_group = find_hidden_groups_from_module(layer, input)

        print()
        for group in hidden_groups:
            pprint(group.hidden_paths)

        assert (len(hidden_groups) == 1)
        print()

    @classmethod
    @pytest.mark.parametrize(
        'cls,',
        [
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
        ]
    )
    def test_snn_two_layers(self, cls):
        n_in = 3
        n_out = 4
        input = bst.random.rand(n_in)

        print()
        print(cls)

        with bst.environ.context(dt=0.1 * u.ms):
            layer = bst.nn.Sequential(cls(n_in, n_out), cls(n_out, n_out))
            bst.nn.init_all_states(layer)
            hidden_groups, hid_path_to_group = find_hidden_groups_from_module(layer, input)

        for group in hidden_groups:
            pprint(group.hidden_paths)

        assert (len(hidden_groups) == 2)
        # print()


class TestHiddenGroup_state_transition:
    def test_gru(self):
        n_in = 3
        n_out = 4

        gru = brainscale.nn.GRUCell(n_in, n_out)
        bst.nn.init_all_states(gru)

        input = bst.random.rand(n_in)
        hidden_groups, hid_path_to_group = find_hidden_groups_from_module(gru, input)

        print()
        for group in hidden_groups:
            print(group)
            hidden_vals = [bst.random.rand_like(invar.aval) for invar in group.hidden_invars]
            input_vals = [bst.random.rand_like(invar.aval) for invar in group.transition_jaxpr_constvars]
            out_vals = group.transition(hidden_vals, input_vals)
            print(out_vals)

    @classmethod
    @pytest.mark.parametrize(
        'cls,',
        [
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
        ]
    )
    def test_snn_single_layer(self, cls):
        n_in = 3
        n_out = 4
        input = bst.random.rand(n_in)

        print()
        print(cls)

        with bst.environ.context(dt=0.1 * u.ms):
            layer = cls(n_in, n_out)
            bst.nn.init_all_states(layer)
            hidden_groups, hid_path_to_group = find_hidden_groups_from_module(layer, input)

        for group in hidden_groups:
            pprint(group.hidden_paths)
            hidden_vals = [bst.random.rand_like(invar.aval) for invar in group.hidden_invars]
            input_vals = [bst.random.rand_like(invar.aval) for invar in group.transition_jaxpr_constvars]
            out_vals = group.transition(hidden_vals, input_vals)
            print(out_vals)

    @classmethod
    @pytest.mark.parametrize(
        'cls,',
        [
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
        ]
    )
    def test_snn_two_layers(self, cls):
        n_in = 3
        n_out = 4
        input = bst.random.rand(n_in)

        print()
        print(cls)
        with bst.environ.context(dt=0.1 * u.ms):
            layer = bst.nn.Sequential(cls(n_in, n_out), cls(n_out, n_out))
            bst.nn.init_all_states(layer)
            hidden_groups, hid_path_to_group = find_hidden_groups_from_module(layer, input)

        for group in hidden_groups:
            pprint(group.hidden_paths)
            hidden_vals = [bst.random.rand_like(invar.aval) for invar in group.hidden_invars]
            input_vals = [bst.random.rand_like(invar.aval) for invar in group.transition_jaxpr_constvars]
            out_vals = group.transition(hidden_vals, input_vals)
            print(out_vals)


class TestHiddenGroup_diagonal_jacobian:
    def test_gru(self):
        n_in = 3
        n_out = 4

        gru = brainscale.nn.GRUCell(n_in, n_out)
        bst.nn.init_all_states(gru)

        input = bst.random.rand(n_in)
        hidden_groups, hid_path_to_group = find_hidden_groups_from_module(gru, input)

        print()
        for group in hidden_groups:
            hidden_vals = [bst.random.rand_like(invar.aval) for invar in group.hidden_invars]
            input_vals = [bst.random.rand_like(invar.aval) for invar in group.transition_jaxpr_constvars]
            diag_jac = group.diagonal_jacobian(hidden_vals, input_vals)
            print(diag_jac)

    def test_gru_accuracy(self):
        n_in = 1
        n_out = 1

        gru = brainscale.nn.GRUCell(n_in, n_out)
        bst.nn.init_all_states(gru)

        input = bst.random.rand(n_in)
        hidden_groups, hid_path_to_group = find_hidden_groups_from_module(gru, input)

        print()
        for group in hidden_groups:
            hidden_vals = [bst.random.rand_like(invar.aval) for invar in group.hidden_invars]
            input_vals = [bst.random.rand_like(invar.aval) for invar in group.transition_jaxpr_constvars]
            diag_jac = group.diagonal_jacobian(hidden_vals, input_vals)
            print(diag_jac)

            fn = lambda hid: group._concat_hidden(group.transition(group._split_hidden(hid), input_vals))
            jax_jac = jax.jacrev(fn)(group._concat_hidden(hidden_vals))
            print(jax_jac)

            assert (u.math.allclose(u.math.squeeze(diag_jac), u.math.squeeze(jax_jac), atol=1e-5))

    @classmethod
    @pytest.mark.parametrize(
        'cls,',
        [
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
        ]
    )
    def test_snn_single_layer(self, cls):
        n_in = 3
        n_out = 4
        input = bst.random.rand(n_in)

        print()
        print(cls)

        with bst.environ.context(dt=0.1 * u.ms):
            layer = cls(n_in, n_out)
            bst.nn.init_all_states(layer)
            hidden_groups, hid_path_to_group = find_hidden_groups_from_module(layer, input)

        for group in hidden_groups:
            pprint(group.hidden_paths)
            hidden_vals = [bst.random.rand_like(invar.aval) for invar in group.hidden_invars]
            input_vals = [bst.random.rand_like(invar.aval) for invar in group.transition_jaxpr_constvars]
            diag_jac = group.diagonal_jacobian(hidden_vals, input_vals)
            print(diag_jac)

    @classmethod
    @pytest.mark.parametrize(
        'cls,',
        [
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
        ]
    )
    def test_snn_single_layer_accuracy(self, cls):
        n_in = 1
        n_out = 1
        input = bst.random.rand(n_in)

        print()
        print(cls)

        with bst.environ.context(dt=0.1 * u.ms):
            layer = cls(n_in, n_out)
            bst.nn.init_all_states(layer)
            hidden_groups, hid_path_to_group = find_hidden_groups_from_module(layer, input)

        for group in hidden_groups:
            pprint(group.hidden_paths)
            hidden_vals = [bst.random.rand_like(invar.aval) for invar in group.hidden_invars]
            input_vals = [bst.random.rand_like(invar.aval) for invar in group.transition_jaxpr_constvars]
            diag_jac = group.diagonal_jacobian(hidden_vals, input_vals)
            diag_jac = u.math.squeeze(diag_jac)
            print(diag_jac)

            fn = lambda hid: group._concat_hidden(group.transition(group._split_hidden(hid), input_vals))
            jax_jac = jax.jacrev(fn)(group._concat_hidden(hidden_vals))
            jax_jac = u.math.squeeze(jax_jac)
            print(jax_jac)

            assert (u.math.allclose(diag_jac, jax_jac, atol=1e-5))

    @classmethod
    @pytest.mark.parametrize(
        'cls,',
        [
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
        ]
    )
    def test_snn_two_layers(self, cls):
        n_in = 3
        n_out = 4
        input = bst.random.rand(n_in)

        print()
        print(cls)
        with bst.environ.context(dt=0.1 * u.ms):
            layer = bst.nn.Sequential(cls(n_in, n_out), cls(n_out, n_out))
            bst.nn.init_all_states(layer)
            hidden_groups, hid_path_to_group = find_hidden_groups_from_module(layer, input)

        for group in hidden_groups:
            pprint(group.hidden_paths)
            hidden_vals = [bst.random.rand_like(invar.aval) for invar in group.hidden_invars]
            input_vals = [bst.random.rand_like(invar.aval) for invar in group.transition_jaxpr_constvars]
            diag_jac = group.diagonal_jacobian(hidden_vals, input_vals)
            print(diag_jac)

    @classmethod
    @pytest.mark.parametrize(
        'cls,',
        [
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
        ]
    )
    def test_snn_two_layers_accuracy(a, cls, ):
        n_in = 1
        n_out = 1
        input = bst.random.rand(n_in)

        print()
        print(cls)
        with bst.environ.context(dt=0.1 * u.ms):
            layer = bst.nn.Sequential(cls(n_in, n_out), cls(n_out, n_out))
            bst.nn.init_all_states(layer)
            hidden_groups, hid_path_to_group = find_hidden_groups_from_module(layer, input)

        for group in hidden_groups:
            pprint(group.hidden_paths)
            hidden_vals = [bst.random.rand_like(invar.aval) for invar in group.hidden_invars]
            input_vals = [bst.random.rand_like(invar.aval) for invar in group.transition_jaxpr_constvars]
            diag_jac = group.diagonal_jacobian(hidden_vals, input_vals)
            diag_jac = u.math.squeeze(diag_jac)
            print(diag_jac)

            fn = lambda hid: group._concat_hidden(group.transition(group._split_hidden(hid), input_vals))
            jax_jac = jax.jacrev(fn)(group._concat_hidden(hidden_vals))
            jax_jac = u.math.squeeze(jax_jac)
            print(jax_jac)

            assert (u.math.allclose(diag_jac, jax_jac, atol=1e-5))


class TestMath:
    @classmethod
    @pytest.mark.parametrize("x, expected", [
        (3, 9),
        (4, 16)
    ])
    def test_square(cls, x, expected):
        assert x * x == expected
