"""
Test Mamdani fuzzy systems are working as intended (e.g., their output is correctly calculated).
"""

import unittest

import numpy as np
import torch

from fuzzy.logic.control.controller import FuzzyLogicController as FLC
from fuzzy.logic.control.defuzzification import Mamdani
from fuzzy.logic.knowledge_base import KnowledgeBase
from fuzzy.logic.variables import LinguisticVariables
from fuzzy.relations.t_norm import Product
from fuzzy.utils.classes import TimeDistributed

from .common import MissingDataHandlingMixin, assert_compile_fullgraph_matches_eager
from .demo_flcs import AVAILABLE_DEVICE, toy_mamdani


class TestMamdani(MissingDataHandlingMixin, unittest.TestCase):
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

    @unittest.skipUnless(
        torch.cuda.is_available(), "requires a second device (CUDA) to move to"
    )
    def test_to_moves_output_links_to_new_device(self) -> None:
        """
        Regression guard: Mamdani.to() used to call self.output_links.to(device)
        without reassigning the result. output_links is a plain tensor (not a
        Parameter or a registered buffer), so nn.Module.to()'s automatic handling
        does not cover it, and Tensor.to() is not in-place - the device move used to
        silently never take effect.

        Returns:
            None
        """
        antecedents, consequents, rules = toy_mamdani(
            t_norm=Product, device=torch.device("cpu")
        )
        knowledge_base = KnowledgeBase.create(
            linguistic_variables=LinguisticVariables(
                inputs=antecedents, targets=consequents
            ),
            rules=rules,
        )
        flc = FLC(
            source=knowledge_base,
            inference=Mamdani,
            device=torch.device("cpu"),
        )
        self.assertEqual(torch.device("cpu"),
                         flc.defuzzification.output_links.device)

        flc.defuzzification.to(torch.device("cuda"))

        self.assertEqual("cuda", flc.defuzzification.output_links.device.type)

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
        #                 self.fuzzy_logic_controller.input_granulation(
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
        self.assertIsNotNone(predicted_y.grad_fn)
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

    def _missing_data_expected(self) -> dict:
        return {
            "expected_lower": [
                [5.9053e-10, 1.6052e-09, 1.0000e00],
                [9.0561e-11, 1.4931e-10, 3.4559e-01],
                [6.2380e-36, 1.3268e-03, 5.9462e-03],
                [0.0000e00, 1.6071e-01, 5.6095e-01],
                [1.3673e-11, 1.2030e-05, 2.5467e-05],
                [0.0000e00, 0.0000e00, 0.0000e00],
            ],
            "expected_temp_upper": [
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
            "expected_upper": [
                [1.0000e00, 1.6052e-09, 5.9053e-10],
                [3.4559e-01, 1.4931e-10, 9.0561e-11],
                [6.2376e-36, 5.9462e-03, 1.3268e-03],
                [0.0000e00, 5.6095e-01, 1.6071e-01],
                [1.3673e-11, 2.5467e-05, 1.2030e-05],
                [9.8450e-01, 9.8450e-01, 4.6504e-01],
            ],
            "sort_lower": True,
        }

    def test_compile_fullgraph_matches_eager(self) -> None:
        """
        Regression guard: torch.compile(fullgraph=True) must trace the Mamdani FLC's
        forward pass without any graph break (e.g. the membership cache's
        @torch.compiler.disable'd lookup, or FuzzySetGroup's former custom
        __getattribute__), and its output must match eager execution exactly.

        Returns:
            None
        """
        input_data = torch.tensor(
            [[1.2, 0.2], [1.1, 0.3], [2.1, 0.1], [2.7, 0.15], [1.7, 0.25]],
            device=AVAILABLE_DEVICE,
        )
        assert_compile_fullgraph_matches_eager(
            self.fuzzy_logic_controller, input_data)

    def test_granulation_layers(self) -> None:
        """
        Test the granulation layers of the Mamdani FLC.

        Returns:
            None
        """
        # check that the antecedents of the Mamdani FLC refer to the input
        # granulation layer (i.e., the fuzzy sets defined in the input space)
        assert torch.equal(
            self.fuzzy_logic_controller.input_granulation.centers,
            torch.tensor(
                [[1.2000, 3.0000, 5.0000, 7.0000], [0.2000, 0.6000, 0.9000, 1.2000]],
                device=AVAILABLE_DEVICE,
            ),
        )
        assert torch.equal(
            self.fuzzy_logic_controller.input_granulation.widths,
            torch.tensor(
                [[0.1000, 0.4000, 0.6000, 0.8000], [0.4000, 0.4000, 0.5000, 0.4500]],
                device=AVAILABLE_DEVICE,
            ),
        )
        # check that the consequence of the Mamdani FLC inference engine refers to the output
        # granulation layer (i.e., the fuzzy sets defined in the output space)
        # specifically, the centers are used in the Mamdani FLC inference
        # prediction
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
        # check the intra-dimensionality of the input & output spaces are
        # correctly calculated
        assert np.allclose(
            self.knowledge_base.intra_dimensions(tags="premise"),
            np.array([4, 4]),  # number of terms in each antecedent variable
        )
        assert np.allclose(
            self.knowledge_base.intra_dimensions(tags="consequence"),
            np.array([2, 3]),  # number of terms in each consequent variable
        )
        # the above is required to generate the correct links shape for fuzzy inference
        # check the variable dimensionality of the input & output spaces is
        # correctly calculated
        assert self.knowledge_base.shape.n_inputs == len(self.antecedents)
        assert self.knowledge_base.shape.n_outputs == len(self.consequents)
        # the above is required to generate the correct links shape for fuzzy
        # inference

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
        for expected_rule, actual_rule in zip(
                self.rules, self.knowledge_base.rules):
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
            [[[1, 0, 0], [0, 1, 1]], [[1, 1, 0], [0, 0, 1]]],
            dtype=torch.int8,
            device=AVAILABLE_DEVICE,
        )
        expected_output_links = torch.tensor(
            [
                [[1, 0, 0], [0, 1, 0]],
                [[0, 1, 0], [0, 0, 1]],
                [[1, 0, 0], [1, 0, 0]],
            ],
            dtype=torch.int8,
            device=AVAILABLE_DEVICE,
        )

        # the following checks that the links between antecedents' memberships (input_links)
        # and the links between rules' activations (output_links) to the consequence layer
        # is correctly constructed and stored in the Mamdani FLC inference
        # engine
        assert torch.allclose(
            expected_input_links,
            self.fuzzy_logic_controller.engine.get_mask(),
        )
        assert torch.allclose(
            expected_output_links,
            self.fuzzy_logic_controller.defuzzification.output_links,
        )
