"""
Tests the fuzzy logic inference engines.
"""

import unittest
from typing import Tuple, Type

import numpy as np
import torch

from fuzzy.logic.control.controller import FuzzyLogicController
from fuzzy.logic.control.defuzzification import ZeroOrder
from fuzzy.logic.knowledge_base import KnowledgeBase
from fuzzy.logic.variables import LinguisticVariables
from fuzzy.relations.t_norm import Minimum, Product, TNorm
from fuzzy.sets.impl import Gaussian
from fuzzy.sets.membership import Membership

from .demo_flcs import toy_tsk

AVAILABLE_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_test_scenario(t_norm: Type[TNorm]) -> Tuple[
    Membership,
    KnowledgeBase,
]:
    """
    Makes a test scenario, with sample data, antecedents, rules, etc.

    Returns:
        Number of output features, consequences (torch.nn.parameter.Parameter), links,
        offset, antecedents_memberships
    """
    _, _, rules = toy_tsk(t_norm=t_norm, device=AVAILABLE_DEVICE)
    antecedents = [
        Gaussian(
            centers=np.array([-0.9, 0.2, 0.7]),
            widths=np.array([1.5, 0.5, 1.1]),
            device=AVAILABLE_DEVICE,
        ),
        Gaussian(
            centers=np.array([-0.4, 1.3, 2.4]),
            widths=np.array([0.85, 0.34, 0.68]),
            device=AVAILABLE_DEVICE,
        ),
    ]

    knowledge_base = KnowledgeBase.create(
        linguistic_variables=LinguisticVariables(
            inputs=antecedents, targets=[]), rules=rules, )

    # get the links and offsets from the knowledge base for fuzzy inference
    input_granulation = knowledge_base.select_by_tags(tags={"premise", "group"})[0][
        "item"
    ].to(AVAILABLE_DEVICE)

    input_data: torch.Tensor = torch.tensor(
        [
            [1.5409961, -0.2934289],
            [-2.1787894, 0.56843126],
            [-1.0845224, -1.3985955],
            [0.40334684, 0.83802634],
        ],
        device=AVAILABLE_DEVICE,
    )

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
        antecedents_memberships, knowledge_base = make_test_scenario(
            t_norm=Product)
        self.assertIsNotNone(antecedents_memberships.degrees.grad_fn)
        product_inference = FuzzyLogicController(
            source=knowledge_base,
            inference=ZeroOrder,
            device=AVAILABLE_DEVICE,
        )
        actual_output: Membership = product_inference.engine(
            antecedents_memberships)
        self.assertIsNotNone(actual_output.degrees.grad_fn)
        expected_output = torch.tensor(
            [
                [
                    6.96742013e-02,
                    2.04711196e-11,
                    7.40043994e-04,
                    2.17433845e-13,
                    1.15470390e-10,
                ],
                [
                    1.32010251e-01,
                    4.71740961e-03,
                    4.03822635e-11,
                    1.44306724e-12,
                    1.04518844e-13,
                ],
                [
                    2.47751176e-01,
                    4.30835969e-28,
                    3.42174753e-04,
                    5.95037309e-31,
                    3.81395014e-17,
                ],
                [
                    5.63383065e-02,
                    7.41864368e-02,
                    1.01591364e-01,
                    1.33775786e-01,
                    4.33210563e-03,
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
        antecedents_memberships, knowledge_base = make_test_scenario(
            t_norm=Minimum)
        self.assertIsNotNone(antecedents_memberships.degrees.grad_fn)
        minimum_inference = FuzzyLogicController(
            source=knowledge_base,
            inference=ZeroOrder,
            device=AVAILABLE_DEVICE,
        )
        actual_output: Membership = minimum_inference.engine(
            antecedents_memberships)
        self.assertIsNotNone(actual_output.degrees.grad_fn)
        expected_output = torch.tensor(
            [
                [
                    7.0778102e-02,
                    2.8922956e-10,
                    7.5176911e-04,
                    2.8922956e-10,
                    1.5359821e-07,
                ],
                [
                    2.7305701e-01,
                    9.7577404e-03,
                    1.4788949e-10,
                    1.4788949e-10,
                    1.4788949e-10,
                ],
                [
                    2.5152883e-01,
                    4.3740527e-28,
                    1.3603799e-03,
                    4.3740527e-28,
                    2.8035920e-14,
                ],
                [
                    1.1986406e-01,
                    1.5783733e-01,
                    1.1986406e-01,
                    1.5783733e-01,
                    5.1112985e-03,
                ],
            ],
            device=AVAILABLE_DEVICE,
        )
        assert torch.allclose(actual_output.degrees, expected_output)
