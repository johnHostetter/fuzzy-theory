"""
Test the ZeroOrderTSK is working as intended, such as its output is correctly calculated.
"""

import unittest
from typing import List, Tuple

import torch
import numpy as np

from fuzzy.sets.impl import Gaussian
from fuzzy.sets.abstract import FuzzySet
from fuzzy.sets.membership import Membership
from fuzzy.logic.rule import Rule
from fuzzy.logic.variables import LinguisticVariables
from fuzzy.logic.knowledge_base import KnowledgeBase
from fuzzy.logic.control.defuzzification import ZeroOrder, Mamdani
from fuzzy.logic.control.controller import FuzzyLogicController as FLC
from fuzzy.relations.n_ary import NAryRelation
from fuzzy.relations.t_norm import Product
from fuzzy.utils.classes import TimeDistributed

from examples.demo_flcs import toy_mamdani


AVAILABLE_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TestTSK(unittest.TestCase):
    """
    Test the zero-order TSK neuro-fuzzy network.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        (
            self.fuzzy_logic_controller,
            self.input_data,
            self.rules,
        ) = self.test_create_tsk()

    def test_gradient_1(self) -> None:
        """
        First test that the gradient of PyTorch is working as intended.

        Returns:
            None
        """
        input_data = torch.tensor(
            [[1.2, 0.2], [1.1, 0.3], [2.1, 0.1], [2.7, 0.15], [1.7, 0.25]],
            device=AVAILABLE_DEVICE,
        )
        # the first variable has fuzzy sets with centers 0, 1, 2 (the column)
        centers = torch.nn.Parameter(
            torch.tensor([[0, 1], [1, 2], [2, 3]], device=AVAILABLE_DEVICE).double()
        )
        actual_result = input_data.unsqueeze(dim=-1) - centers.T
        expected_result = torch.tensor(
            [
                [[1.2000, 0.2000, -0.8000], [-0.8000, -1.8000, -2.8000]],
                [[1.1000, 0.1000, -0.9000], [-0.7000, -1.7000, -2.7000]],
                [[2.1000, 1.1000, 0.1000], [-0.9000, -1.9000, -2.9000]],
                [[2.7000, 1.7000, 0.7000], [-0.8500, -1.8500, -2.8500]],
                [[1.7000, 0.7000, -0.3000], [-0.7500, -1.7500, -2.7500]],
            ],
            device=AVAILABLE_DEVICE,
        )

        assert torch.allclose(actual_result.float(), expected_result)

    def test_gradient_2(self) -> None:
        """
        Second test that the gradient of PyTorch is working as intended.

        Returns:
            None
        """
        value_1 = torch.nn.Parameter(
            torch.tensor([0, 1], device=AVAILABLE_DEVICE).double()
        )
        value_3 = 2**value_1
        assert value_3.grad_fn is not None

    def test_to_device(self) -> None:
        """
        Check that we can move the FLC to another device.

        Returns:
            None
        """
        self.fuzzy_logic_controller.to(torch.device("cpu"))
        self.assertEqual(torch.device("cpu"), self.fuzzy_logic_controller.device)
        # check that this is reflected in each of its torch.nn.Modules
        for module in self.fuzzy_logic_controller.children():
            self.assertEqual(torch.device("cpu"), module.device)
        self.fuzzy_logic_controller.to(AVAILABLE_DEVICE)
        self.assertEqual(AVAILABLE_DEVICE, self.fuzzy_logic_controller.device)
        # check that this is reflected in each of its torch.nn.Modules
        for module in self.fuzzy_logic_controller.children():
            self.assertEqual(AVAILABLE_DEVICE, module.device)

    def test_tsk(self) -> None:
        """
        Test the zero-order TSK neuro-fuzzy network.

        Returns:
            None
        """
        # check that the output is correct
        predicted_y = self.fuzzy_logic_controller(self.input_data)
        self.assertEqual(predicted_y.shape, torch.Size([5, 1]))

        # check that the input granulation was correctly created
        assert torch.allclose(
            self.fuzzy_logic_controller.input.centers,
            torch.tensor(
                [[1.2000, 3.0000, 5.0000, 7.0000], [0.2000, 0.6000, 0.9000, 1.2000]],
                device=AVAILABLE_DEVICE,
            ),
        )
        assert torch.allclose(
            self.fuzzy_logic_controller.input.widths,
            torch.tensor(
                [[0.1000, 0.4000, 0.6000, 0.8000], [0.4000, 0.4000, 0.5000, 0.4500]],
                device=AVAILABLE_DEVICE,
            ),
        )
        # check that the output granulation was correctly created
        self.assertEqual(
            self.fuzzy_logic_controller.defuzzification.consequences.shape,
            torch.Size([4, 1]),
        )

    def test_missing_data_handling(self) -> None:
        """
        Test that the FLC can handle missing data.

        Returns:
            None
        """
        # check that the output is correct
        data_with_missing = torch.tensor(
            [
                [1.2, 0.2],
                [1.1, 0.3],
                [2.1, 0.1],
                [2.7, 0.15],
                [1.7, 0.25],
                [np.nan, 0.25],
            ],
            device=AVAILABLE_DEVICE,
        )
        assert self.fuzzy_logic_controller.input(data_with_missing).degrees.shape == (
            6,
            2,
            4,
        )  # 6 samples, 2 features, 4 granules
        assert torch.allclose(
            self.fuzzy_logic_controller.input(data_with_missing).degrees.to_dense(),
            torch.tensor(
                [
                    [
                        [1.0000e00, 1.6052e-09, 3.8016e-18, 1.4873e-23],
                        [1.0000e00, 3.6788e-01, 1.4086e-01, 7.1670e-03],
                    ],
                    [
                        [3.6788e-01, 1.5894e-10, 4.4777e-19, 2.3903e-24],
                        [9.3941e-01, 5.6978e-01, 2.3693e-01, 1.8316e-02],
                    ],
                    [
                        [6.6403e-36, 6.3297e-03, 7.1515e-11, 5.0953e-17],
                        [9.3941e-01, 2.0961e-01, 7.7305e-02, 2.5407e-03],
                    ],
                    [
                        [0.0000e00, 5.6978e-01, 4.1523e-07, 2.8376e-13],
                        [9.8450e-01, 2.8206e-01, 1.0540e-01, 4.3202e-03],
                    ],
                    [
                        [1.3888e-11, 2.5868e-05, 7.2878e-14, 8.6804e-20],
                        [9.8450e-01, 4.6504e-01, 1.8452e-01, 1.1600e-02],
                    ],
                    [
                        [np.nan, np.nan, np.nan, np.nan],
                        [9.8450e-01, 4.6504e-01, 1.8452e-01, 1.1600e-02],
                    ],
                ],
                device=AVAILABLE_DEVICE,
            ).float(),
            rtol=3e-3,
            atol=3e-3,
            equal_nan=True,
        )
        self.fuzzy_logic_controller.engine.nan_replacement = 0.0
        lower_rule_activations: Membership = self.fuzzy_logic_controller.engine(
            self.fuzzy_logic_controller.input(data_with_missing)
        )
        assert torch.allclose(
            lower_rule_activations.degrees,
            torch.tensor(
                [
                    [1.0000e00, 1.6052e-09, 5.9053e-10, 3.6788e-01],
                    [3.4559e-01, 1.4931e-10, 9.0561e-11, 5.6978e-01],
                    [6.2376e-36, 5.9462e-03, 1.3268e-03, 2.0961e-01],
                    [0.0000e00, 5.6095e-01, 1.6071e-01, 2.8206e-01],
                    [1.3673e-11, 2.5467e-05, 1.2030e-05, 4.6504e-01],
                    [0.0000e00, 0.0000e00, 0.0000e00, 0.0000e00],
                ],
                device=AVAILABLE_DEVICE,
            ),
            rtol=3e-3,
            atol=3e-3,
            equal_nan=True,
        )
        self.fuzzy_logic_controller.engine.nan_replacement = 1.0
        temp_upper_rule_activations: torch.Tensor = (
            self.fuzzy_logic_controller.engine.apply_mask(
                self.fuzzy_logic_controller.input(data_with_missing)
            )
        )
        assert torch.allclose(
            temp_upper_rule_activations,
            torch.tensor(
                [
                    [
                        [1.0000e00, 1.6052e-09, 1.6052e-09, 1.0000e00],
                        [1.0000e00, 1.0000e00, 3.6788e-01, 3.6788e-01],
                    ],
                    [
                        [3.6788e-01, 1.5894e-10, 1.5894e-10, 1.0000e00],
                        [9.3941e-01, 9.3941e-01, 5.6978e-01, 5.6978e-01],
                    ],
                    [
                        [6.6399e-36, 6.3297e-03, 6.3297e-03, 1.0000e00],
                        [9.3941e-01, 9.3941e-01, 2.0961e-01, 2.0961e-01],
                    ],
                    [
                        [0.0000e00, 5.6978e-01, 5.6978e-01, 1.0000e00],
                        [9.8450e-01, 9.8450e-01, 2.8206e-01, 2.8206e-01],
                    ],
                    [
                        [1.3888e-11, 2.5868e-05, 2.5868e-05, 1.0000e00],
                        [9.8450e-01, 9.8450e-01, 4.6504e-01, 4.6504e-01],
                    ],
                    [
                        [1.0000e00, 1.0000e00, 1.0000e00, 1.0000e00],
                        [9.8450e-01, 9.8450e-01, 4.6504e-01, 4.6504e-01],
                    ],
                ],
                device=AVAILABLE_DEVICE,
            ),
            rtol=3e-3,
            atol=3e-3,
            equal_nan=True,
        )
        upper_rule_activations: Membership = self.fuzzy_logic_controller.engine(
            self.fuzzy_logic_controller.input(data_with_missing)
        )
        assert torch.allclose(
            upper_rule_activations.degrees,
            torch.tensor(
                [
                    [1.0000e00, 1.6052e-09, 5.9053e-10, 3.6788e-01],
                    [3.4559e-01, 1.4931e-10, 9.0561e-11, 5.6978e-01],
                    [6.2376e-36, 5.9462e-03, 1.3268e-03, 2.0961e-01],
                    [0.0000e00, 5.6095e-01, 1.6071e-01, 2.8206e-01],
                    [1.3673e-11, 2.5467e-05, 1.2030e-05, 4.6504e-01],
                    [9.8450e-01, 9.8450e-01, 4.6504e-01, 4.6504e-01],
                ],
                device=AVAILABLE_DEVICE,
            ),
            rtol=3e-3,
            atol=3e-3,
            equal_nan=True,
        )

    def test_create_tsk(self) -> Tuple[FLC, torch.Tensor, List[Rule]]:
        """
        Test the creation of a TSK model.

        Returns:
            The FLC, input data, and rules.
        """
        input_data = torch.tensor(
            [[1.2, 0.2], [1.1, 0.3], [2.1, 0.1], [2.7, 0.15], [1.7, 0.25]],
            device=AVAILABLE_DEVICE,
        )
        antecedents = [
            Gaussian(
                centers=np.array([1.2, 3.0, 5.0, 7.0]),
                widths=np.array([0.1, 0.4, 0.6, 0.8]),
                device=AVAILABLE_DEVICE,
            ),
            Gaussian(
                centers=np.array([0.2, 0.6, 0.9, 1.2]),
                widths=np.array([0.4, 0.4, 0.5, 0.45]),
                device=AVAILABLE_DEVICE,
            ),
        ]
        # check that antecedents were correctly created
        self.assertTrue(
            torch.equal(
                antecedents[0].get_centers(),
                torch.tensor([[1.2, 3.0, 5.0, 7.0]], device=AVAILABLE_DEVICE),
            )
        )
        self.assertTrue(
            torch.equal(
                antecedents[0].get_widths(),
                torch.tensor([[0.1, 0.4, 0.6, 0.8]], device=AVAILABLE_DEVICE),
            )
        )
        self.assertTrue(
            torch.equal(
                antecedents[1].get_centers(),
                torch.tensor([[0.2, 0.6, 0.9, 1.2]], device=AVAILABLE_DEVICE),
            )
        )
        self.assertTrue(
            torch.equal(
                antecedents[1].get_widths(),
                torch.tensor([[0.4, 0.4, 0.5, 0.45]], device=AVAILABLE_DEVICE),
            )
        )
        rules = (
            [  # could be a set of rules, but a list is used here for reproducibility
                Rule(
                    premise=Product((0, 0), (1, 0), device=AVAILABLE_DEVICE),
                    consequence=NAryRelation((0, 0), device=AVAILABLE_DEVICE),
                ),
                Rule(
                    premise=Product((0, 1), (1, 0), device=AVAILABLE_DEVICE),
                    consequence=NAryRelation((0, 1), device=AVAILABLE_DEVICE),
                ),
                Rule(
                    premise=Product((0, 1), (1, 1), device=AVAILABLE_DEVICE),
                    consequence=NAryRelation((0, 2), device=AVAILABLE_DEVICE),
                ),
                Rule(
                    premise=Product((1, 1), device=AVAILABLE_DEVICE),
                    consequence=NAryRelation((0, 3), device=AVAILABLE_DEVICE),
                ),
            ]
        )
        # check that rules were correctly created
        knowledge_base = KnowledgeBase.create(
            linguistic_variables=LinguisticVariables(inputs=antecedents, targets=[]),
            rules=rules,
        )
        rule_vertex = knowledge_base.graph.vs.find(item_eq=rules[0])
        self.assertEqual(
            rule_vertex["item"], rules[0]
        )  # it is the correct relation we wanted
        self.assertIn("item", rule_vertex.attributes())  # it has 'item' attribute
        rule_vertices = knowledge_base.select_by_tags("rule")
        self.assertEqual(
            len(rule_vertices), len(rules)
        )  # the number of rule vertices should equal len(rules)
        # there should be 2 rules that use (1, 1);
        # the last rule has been simplified (redundant mention of condition)
        self.assertIn(rules[2].premise, knowledge_base[(1, 1)].keys())
        self.assertEqual(
            set(knowledge_base[(1, 1)].values()),
            {
                frozenset({(0, 1), (1, 1)}),
                frozenset({(1, 1)}),
            },
        )
        # the recovered rules should be in the same order as the rules
        for expected_rule, actual_rule in zip(rules, knowledge_base.rules):
            self.assertEqual(expected_rule, actual_rule)

        # check a zero-order TSK cannot be created with an incorrect number of consequences
        # self.assertRaises(
        #     ValueError,
        #     ZeroOrderTSK,
        #     specifications=Specifications(
        #         type="tsk",
        #         t_norm="algebraic_product",
        #     ),
        #     knowledge_base=knowledge_base,
        #     consequences=torch.zeros(len(rules) - 1),
        # )

        # check that the zero-order TSK neuro-fuzzy network was correctly created
        flc = FLC(
            source=knowledge_base,
            inference=ZeroOrder,
            device=AVAILABLE_DEVICE,
        )

        # check that the number of input/output features are correct
        assert flc.shape.n_inputs == 2
        assert flc.shape.n_outputs == 1

        actual_variables: List[FuzzySet] = flc.linguistic_variables().inputs
        for actual_variable, expected_variable in zip(actual_variables, antecedents):
            assert torch.allclose(
                actual_variable.get_centers(), expected_variable.get_centers()
            )
            assert torch.allclose(
                actual_variable.get_widths(), expected_variable.get_widths()
            )

        return flc, input_data, rules


class TestMamdani(unittest.TestCase):
    """
    Test the Mamdani neuro-fuzzy network.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.antecedents, self.consequents, self.rules = toy_mamdani(
            t_norm=Product, device=AVAILABLE_DEVICE
        )
        self.knowledge_base = KnowledgeBase.create(
            linguistic_variables=LinguisticVariables(
                inputs=self.antecedents, targets=self.consequents
            ),
            rules=self.rules,
        )

        self.fuzzy_logic_controller = FLC(
            source=self.knowledge_base,
            inference=Mamdani,
            device=AVAILABLE_DEVICE,
        )

    def test_mamdani(self) -> None:
        """
        Test the Mamdani neuro-fuzzy network.

        Returns:
            None
        """
        input_data = torch.tensor(
            [[1.2, 0.2], [1.1, 0.3], [2.1, 0.1], [2.7, 0.15], [1.7, 0.25]],
            device=AVAILABLE_DEVICE,
        )

        self.test_rules_are_added_correctly()

        self.test_space_dimensionality()
        self.fuzzy_logic_controller = FLC(
            source=self.knowledge_base,
            inference=Mamdani,
            device=AVAILABLE_DEVICE,
        )

        self.test_links_and_offsets()

        self.test_granulation_layers()

        # testing the calculation of the intermediate output of the Mamdani Fuzzy Logic Controller
        # NOTE: this may no longer be needed
        # calc_intermediate_output = (
        #     self.fuzzy_logic_controller.engine.calc_intermediate_output(
        #         self.fuzzy_logic_controller.engine(
        #             self.fuzzy_logic_controller.dispersion(
        #                 self.fuzzy_logic_controller.input(
        #                     input_data
        #                 )
        #             )
        #         )
        #     ).float()
        # )
        #
        # expected_calc_intermediate_output = torch.tensor(
        #     [
        #         [
        #             [1.00000000e00, 1.60523417e-09, 0.00000000e00, 0.00000000e00],
        #             [5.90532567e-10, 1.00000000e00, 1.60523417e-09, 0.00000000e00],
        #         ],
        #         [
        #             [3.45590532e-01, 1.49309759e-10, 0.00000000e00, 0.00000000e00],
        #             [9.05609476e-11, 3.45590532e-01, 1.49309759e-10, 0.00000000e00],
        #         ],
        #         [
        #             [1.32677925e-03, 5.94621152e-03, 0.00000000e00, 0.00000000e00],
        #             [1.32677925e-03, 6.23749488e-36, 5.94621152e-03, 0.00000000e00],
        #         ],
        #         [
        #             [1.60714641e-01, 5.60949206e-01, 0.00000000e00, 0.00000000e00],
        #             [1.60714641e-01, 0.00000000e00, 5.60949206e-01, 0.00000000e00],
        #         ],
        #         [
        #             [1.20298064e-05, 2.54671013e-05, 0.00000000e00, 0.00000000e00],
        #             [1.20298064e-05, 1.36725787e-11, 2.54671013e-05, 0.00000000e00],
        #         ],
        #     ],
        #     device=AVAILABLE_DEVICE,
        # )
        #
        # assert torch.allclose(
        #     calc_intermediate_output, expected_calc_intermediate_output
        # )

        expected_y = torch.tensor(
            [
                [5.0000000e-01, -6.9999999e-01],
                [1.7279544e-01, -2.4191362e-01],
                [2.4472545e-03, -5.6169494e-03],
                [2.4864213e-01, -5.3699726e-01],
                [1.3655053e-05, -2.5326384e-05],
            ],
            device=AVAILABLE_DEVICE,
        )
        predicted_y = self.fuzzy_logic_controller(input_data)

        assert torch.allclose(predicted_y, expected_y)

        # check that the FLC can handle missing values

        self.test_missing_data_handling()

        # test the time distributed version

        time_distributed = TimeDistributed(
            self.fuzzy_logic_controller, batch_first=True
        )
        temporal_predictions = time_distributed(
            input_data[None, :].repeat_interleave(repeats=4, dim=0)
        )

        assert torch.allclose(
            temporal_predictions,
            expected_y[None, :].repeat_interleave(repeats=4, dim=0),
        )

    def test_missing_data_handling(self):
        """
        Test that the FLC can handle missing data.

        Returns:
            None
        """
        # check that the output is correct
        data_with_missing = torch.tensor(
            [
                [1.2, 0.2],
                [1.1, 0.3],
                [2.1, 0.1],
                [2.7, 0.15],
                [1.7, 0.25],
                [np.nan, 0.25],
            ],
            device=AVAILABLE_DEVICE,
        )
        assert self.fuzzy_logic_controller.input(data_with_missing).degrees.shape == (
            6,
            2,
            4,
        )  # 6 samples, 2 features, 4 granules
        assert torch.allclose(
            self.fuzzy_logic_controller.input(data_with_missing)
            .degrees.to_dense()
            .float(),
            torch.tensor(
                [
                    [
                        [1.0000e00, 1.6052e-09, 3.8016e-18, 1.4873e-23],
                        [1.0000e00, 3.6788e-01, 1.4086e-01, 7.1670e-03],
                    ],
                    [
                        [3.6788e-01, 1.5894e-10, 4.4777e-19, 2.3903e-24],
                        [9.3941e-01, 5.6978e-01, 2.3693e-01, 1.8316e-02],
                    ],
                    [
                        [6.6403e-36, 6.3297e-03, 7.1515e-11, 5.0953e-17],
                        [9.3941e-01, 2.0961e-01, 7.7305e-02, 2.5407e-03],
                    ],
                    [
                        [0.0000e00, 5.6978e-01, 4.1523e-07, 2.8376e-13],
                        [9.8450e-01, 2.8206e-01, 1.0540e-01, 4.3202e-03],
                    ],
                    [
                        [1.3888e-11, 2.5868e-05, 7.2878e-14, 8.6804e-20],
                        [9.8450e-01, 4.6504e-01, 1.8452e-01, 1.1600e-02],
                    ],
                    [
                        [np.nan, np.nan, np.nan, np.nan],
                        [9.8450e-01, 4.6504e-01, 1.8452e-01, 1.1600e-02],
                    ],
                ],
                device=AVAILABLE_DEVICE,
            ),
            rtol=3e-3,
            atol=3e-3,
            equal_nan=True,
        )
        self.fuzzy_logic_controller.engine.nan_replacement = 0.0
        lower_rule_activations: Membership = self.fuzzy_logic_controller.engine(
            self.fuzzy_logic_controller.input(data_with_missing)
        )
        assert torch.allclose(
            lower_rule_activations.degrees.sort().values,
            torch.tensor(
                [
                    [5.9053e-10, 1.6052e-09, 1.0000e00],
                    [9.0561e-11, 1.4931e-10, 3.4559e-01],
                    [6.2380e-36, 1.3268e-03, 5.9462e-03],
                    [0.0000e00, 1.6071e-01, 5.6095e-01],
                    [1.3673e-11, 1.2030e-05, 2.5467e-05],
                    [0.0000e00, 0.0000e00, 0.0000e00],
                ],
                device=AVAILABLE_DEVICE,
            ),
            rtol=3e-3,
            atol=3e-3,
            equal_nan=True,
        )
        self.fuzzy_logic_controller.engine.nan_replacement = 1.0
        temp_upper_rule_activations: torch.Tensor = (
            self.fuzzy_logic_controller.engine.apply_mask(
                self.fuzzy_logic_controller.input(data_with_missing)
            )
        )
        assert torch.allclose(
            temp_upper_rule_activations,
            torch.tensor(
                [
                    [
                        [1.0000e00, 1.6052e-09, 1.6052e-09],
                        [1.0000e00, 1.0000e00, 3.6788e-01],
                    ],
                    [
                        [3.6788e-01, 1.5894e-10, 1.5894e-10],
                        [9.3941e-01, 9.3941e-01, 5.6978e-01],
                    ],
                    [
                        [6.6399e-36, 6.3297e-03, 6.3297e-03],
                        [9.3941e-01, 9.3941e-01, 2.0961e-01],
                    ],
                    [
                        [0.0000e00, 5.6978e-01, 5.6978e-01],
                        [9.8450e-01, 9.8450e-01, 2.8206e-01],
                    ],
                    [
                        [1.3888e-11, 2.5868e-05, 2.5868e-05],
                        [9.8450e-01, 9.8450e-01, 4.6504e-01],
                    ],
                    [
                        [1.0000e00, 1.0000e00, 1.0000e00],
                        [9.8450e-01, 9.8450e-01, 4.6504e-01],
                    ],
                ],
                device=AVAILABLE_DEVICE,
            ),
            rtol=3e-3,
            atol=3e-3,
            equal_nan=True,
        )
        upper_rule_activations: Membership = self.fuzzy_logic_controller.engine(
            self.fuzzy_logic_controller.input(data_with_missing)
        )
        assert torch.allclose(
            upper_rule_activations.degrees,
            torch.tensor(
                [
                    [1.0000e00, 1.6052e-09, 5.9053e-10],
                    [3.4559e-01, 1.4931e-10, 9.0561e-11],
                    [6.2376e-36, 5.9462e-03, 1.3268e-03],
                    [0.0000e00, 5.6095e-01, 1.6071e-01],
                    [1.3673e-11, 2.5467e-05, 1.2030e-05],
                    [9.8450e-01, 9.8450e-01, 4.6504e-01],
                ],
                device=AVAILABLE_DEVICE,
            ),
            rtol=3e-3,
            atol=3e-3,
            equal_nan=True,
        )

    def test_granulation_layers(self) -> None:
        """
        Test the granulation layers of the Mamdani FLC.

        Returns:
            None
        """
        # check that the antecedents of the Mamdani FLC refer to the input
        # granulation layer (i.e., the fuzzy sets defined in the input space)
        assert torch.equal(
            self.fuzzy_logic_controller.input.centers,
            torch.tensor(
                [[1.2000, 3.0000, 5.0000, 7.0000], [0.2000, 0.6000, 0.9000, 1.2000]],
                device=AVAILABLE_DEVICE,
            ),
        )
        assert torch.equal(
            self.fuzzy_logic_controller.input.widths,
            torch.tensor(
                [[0.1000, 0.4000, 0.6000, 0.8000], [0.4000, 0.4000, 0.5000, 0.4500]],
                device=AVAILABLE_DEVICE,
            ),
        )
        # check that the consequence of the Mamdani FLC inference engine refers to the output
        # granulation layer (i.e., the fuzzy sets defined in the output space)
        # specifically, the centers are used in the Mamdani FLC inference prediction
        assert torch.equal(
            self.fuzzy_logic_controller.defuzzification.consequences.centers,
            torch.tensor(
                [[0.5000, 0.3000, 0.0000], [-0.2000, -0.7000, -0.9000]],
                device=AVAILABLE_DEVICE,
            ),
        )
        assert torch.equal(
            self.fuzzy_logic_controller.defuzzification.consequences.widths,
            torch.tensor(
                [[0.1000, 0.4000, -1.0000], [0.4000, 0.4000, 0.5000]],
                device=AVAILABLE_DEVICE,
            ),
        )

    def test_space_dimensionality(self) -> None:
        """
        Test that the dimensionality of the input & output spaces is correctly calculated. This
        is required to generate the correct links shape for fuzzy inference. The dimensionality
        of the input & output spaces is calculated as the number of unique input & output
        variables, respectively. The dimensionality of the input & output spaces is used to
        generate the correct links shape for fuzzy inference.

        Returns:
            None
        """
        # check the intra-dimensionality of the input & output spaces are correctly calculated
        assert np.allclose(
            self.knowledge_base.intra_dimensions(tags="premise"),
            np.array([4, 4]),  # number of terms in each antecedent variable
        )
        assert np.allclose(
            self.knowledge_base.intra_dimensions(tags="consequence"),
            np.array([2, 3]),  # number of terms in each consequent variable
        )
        # the above is required to generate the correct links shape for fuzzy inference
        # check the variable dimensionality of the input & output spaces is correctly calculated
        assert self.knowledge_base.shape.n_inputs == len(self.antecedents)
        assert self.knowledge_base.shape.n_outputs == len(self.consequents)
        # the above is required to generate the correct links shape for fuzzy inference

    def test_rules_are_added_correctly(self) -> None:
        """
        Test that the rules are correctly added to the knowledge base. This is done by checking
        that the rule vertices are correctly added to the graph and that the rules are correctly
        stored in the knowledge base. The rule vertices are identified by their item attribute
        being equal to the provided Rule instances.

        Returns:
            None
        """
        rule_vertex = self.knowledge_base.graph.vs.find(item_eq=self.rules[0])
        assert (
            rule_vertex["item"] == self.rules[0]
        )  # it is the correct relation we wanted
        assert "item" in rule_vertex.attributes()  # it has 'type' attribute
        rule_vertices = self.knowledge_base.select_by_tags("rule")
        # the number of rule vertices should equal len(rules)
        assert len(rule_vertices) == len(self.rules)
        # the recovered rules should be in the same order as the rules
        for expected_rule, actual_rule in zip(self.rules, self.knowledge_base.rules):
            self.assertEqual(expected_rule, actual_rule)

    def test_links_and_offsets(self) -> None:
        """
        Test that the links and offsets are correctly constructed and stored in the Mamdani FLC
        inference engine. This is done by comparing the links and offsets of the Mamdani FLC
        inference engine with the links and offsets of the knowledge base. The links and offsets
        of the knowledge base are calculated by the matrix method of the knowledge base.

        Returns:
            None
        """
        expected_input_links = torch.tensor(
            [[[1.0, 0.0, 0.0], [0.0, 1.0, 1.0]], [[1.0, 1.0, 0.0], [0.0, 0.0, 1.0]]],
            dtype=torch.int8,
            device=AVAILABLE_DEVICE,
        )
        expected_output_links = torch.tensor(
            [
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            ],
            dtype=torch.int8,
            device=AVAILABLE_DEVICE,
        )

        # the following checks that the links between antecedents' memberships (input_links)
        # and the links between rules' activations (output_links) to the consequence layer
        # is correctly constructed and stored in the Mamdani FLC inference engine
        assert torch.allclose(
            expected_input_links,
            self.fuzzy_logic_controller.engine.applied_mask,
        )
        assert torch.allclose(
            expected_output_links,
            self.fuzzy_logic_controller.defuzzification.output_links,
        )
