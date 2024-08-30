"""
Unit tests for temporal functionality of neuro-fuzzy networks.
"""

import unittest

import torch

from fuzzy.utils import TimeDistributed


AVAILABLE_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TestTimeDistributed(unittest.TestCase):
    """
    Test the TimeDistributed class.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_data = torch.tensor(
            [
                [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
                [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
            ],
            dtype=torch.float32,
            device=AVAILABLE_DEVICE,
        )

    def test_time_distributed_with_no_temporal_data(self) -> None:
        """
        Test the TimeDistributed class with no temporal data (e.g., no temporal dimension).

        Returns:
            None
        """
        time_distributed = TimeDistributed(
            module=torch.nn.LazyLinear(out_features=1, device=AVAILABLE_DEVICE),
            batch_first=True,
        )
        actual_output = time_distributed(self.input_data[0])
        expected_output = time_distributed.module(self.input_data[0])
        self.assertTrue(torch.allclose(actual_output, expected_output))

    def test_time_distributed_with_temporal_data(self) -> None:
        """
        Test the TimeDistributed class with temporal data (e.g., temporal dimension).

        Returns:
            None
        """
        time_distributed = TimeDistributed(
            module=torch.nn.LazyLinear(out_features=1, device=AVAILABLE_DEVICE),
            batch_first=True,
        )
        actual_output = time_distributed(self.input_data)
        expected_output = time_distributed.module(self.input_data)
        self.assertTrue(torch.allclose(actual_output, expected_output))

        # now change the batch_first to False
        time_distributed = TimeDistributed(
            module=torch.nn.LazyLinear(out_features=1, device=AVAILABLE_DEVICE),
            batch_first=False,
        )
        actual_output = time_distributed(self.input_data)
        # now first dimension is the temporal dimension, second is the batch dimension, and
        # third is the feature dimension
        self.assertEqual(actual_output.shape, torch.Size([3, 2, 1]))
