"""
Tests the fuzzy logic inference engines.
"""

import unittest
from typing import Tuple, Type

import torch
import numpy as np

from fuzzy.sets.continuous.impl import Gaussian
from fuzzy.sets.continuous.membership import Membership
from fuzzy.relations.continuous.t_norm import Minimum, Product, TNorm
from fuzzy.logic.knowledge_base import KnowledgeBase
from fuzzy.logic.variables import LinguisticVariables
from fuzzy.logic.control.defuzzification import ZeroOrder
from fuzzy.logic.control.controller import FuzzyLogicController

from examples.continuous.demo_flcs import toy_tsk


AVAILABLE_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_test_scenario(t_norm: Type[TNorm]) -> Tuple[
    torch.Tensor,
    KnowledgeBase,
]:
    """
    Makes a test scenario, with sample data, antecedents, rules, etc.

    Returns:
        Number of output features, consequences (torch.nn.parameter.Parameter), links,
        offset, antecedents_memberships
    """
    input_data = torch.tensor(
        [
            [1.5409961, -0.2934289],
            [-2.1787894, 0.56843126],
            [-1.0845224, -1.3985955],
            [0.40334684, 0.83802634],
        ],
        device=AVAILABLE_DEVICE,
    )

    _, _, rules = toy_tsk(t_norm=t_norm, device=AVAILABLE_DEVICE)
    antecedents = [
        Gaussian(
            centers=np.array([-1, 0.0, 1.0]),
            widths=np.array([1.0, 1.0, 1.0]),
        ),
        Gaussian(
            centers=np.array([-1.0, 0.0, 1.0]),
            widths=np.array([1.0, 1.0, 1.0]),
        ),
    ]

    knowledge_base = KnowledgeBase.create(
        linguistic_variables=LinguisticVariables(inputs=antecedents, targets=[]),
        rules=rules,
    )

    # get the links and offsets from the knowledge base for fuzzy inference
    input_granulation = knowledge_base.select_by_tags(tags={"premise", "group"})[0][
        "item"
    ].to(AVAILABLE_DEVICE)

    antecedents_memberships = input_granulation(input_data)

    return antecedents_memberships, knowledge_base


class TestFuzzyInference(unittest.TestCase):
    """
    Test the various implementations of fuzzy logic inference.
    """

    def test_product_inference_output(self) -> None:
        """
        Test the soft.fuzzy.logic.inference.engines.ProductInference class.

        Returns:
            None
        """
        antecedents_memberships, knowledge_base = make_test_scenario(t_norm=Product)
        product_inference = FuzzyLogicController(
            source=knowledge_base,
            inference=ZeroOrder,
            device=AVAILABLE_DEVICE,
        )
        actual_output: Membership = product_inference.engine(antecedents_memberships)
        expected_output = torch.tensor(
            [
                [
                    9.52992122e-04,
                    1.44050468e-03,
                    5.64775779e-02,
                    8.53692423e-02,
                    1.74637646e-02,
                ],
                [
                    2.12899314e-02,
                    1.80385602e-01,
                    7.41303946e-04,
                    6.28092951e-03,
                    7.20215626e-03,
                ],
                [
                    8.47027257e-01,
                    1.40406535e-01,
                    2.63140523e-01,
                    4.36191975e-02,
                    9.78540533e-04,
                ],
                [
                    4.75897548e-03,
                    6.91366485e-02,
                    2.89834770e-02,
                    4.21061312e-01,
                    8.27849315e-01,
                ],
            ],
            device=AVAILABLE_DEVICE,
        )
        assert torch.allclose(actual_output.degrees, expected_output)

    def test_minimum_inference_output(self) -> None:
        """
        Test the soft.fuzzy.logic.inference.engines.MinimumInference class.

        Returns:
            None
        """
        antecedents_memberships, knowledge_base = make_test_scenario(t_norm=Minimum)
        minimum_inference = FuzzyLogicController(
            source=knowledge_base,
            inference=ZeroOrder,
            device=AVAILABLE_DEVICE,
        )
        actual_output: Membership = minimum_inference.engine(antecedents_memberships)
        expected_output = torch.tensor(
            [
                [0.00157003, 0.00157003, 0.09304529, 0.09304529, 0.09304529],
                [0.08543695, 0.24918881, 0.00867662, 0.00867662, 0.00867662],
                [0.8531001, 0.1414132, 0.3084521, 0.1414132, 0.00317242],
                [0.034104, 0.13954304, 0.034104, 0.49545035, 0.8498557],
            ],
            device=AVAILABLE_DEVICE,
        )
        assert torch.allclose(actual_output.degrees, expected_output)
