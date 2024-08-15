"""
Test the LogisticCurve class.
"""

import unittest

import torch

from fuzzy.sets.continuous.impl import LogisticCurve


AVAILABLE_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TestLogistic(unittest.TestCase):
    """
    Test the LogisticCurve class.
    """

    def test_logistic_curve(self) -> None:
        """
        Test the calculation of the logistic curve.

        Returns:
            None
        """
        elements = torch.tensor(
            [
                [-1.1258, -1.1524, -0.2506],
                [-0.4339, 0.8487, 0.6920],
                [-0.3160, -2.1152, 0.4681],
                [-0.1577, 1.4437, 0.2660],
                [0.1665, 0.8744, -0.1435],
                [-0.1116, 0.9318, 1.2590],
                [2.0050, 0.0537, 0.6181],
                [-0.4128, -0.8411, -2.3160],
            ],
            device=AVAILABLE_DEVICE,
        )
        logistic_curve = LogisticCurve(midpoint=0.5, growth=10, supremum=1)

        self.assertTrue(
            torch.allclose(
                logistic_curve(elements),
                torch.tensor(
                    [
                        [8.6944e-08, 6.6637e-08, 5.4947e-04],
                        [8.7920e-05, 9.7032e-01, 8.7214e-01],
                        [2.8578e-04, 4.3886e-12, 4.2092e-01],
                        [1.3901e-03, 9.9992e-01, 8.7864e-02],
                        [3.4390e-02, 9.7689e-01, 1.6018e-03],
                        [2.2024e-03, 9.8685e-01, 9.9949e-01],
                        [1.0000e00, 1.1396e-02, 7.6513e-01],
                        [1.0857e-04, 1.4986e-06, 5.8921e-13],
                    ],
                    device=AVAILABLE_DEVICE,
                ),
                atol=1e-4,
                rtol=1e-4,
            )
        )
