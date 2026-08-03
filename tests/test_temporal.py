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
        # the shape of the input data is unchanged by batch_first; only its
        # interpretation (which leading dimension is "batch" vs "timestep") differs,
        # so the output must still preserve the original (2, 3) leading
        # dimensions
        self.assertEqual(actual_output.shape, torch.Size([2, 3, 1]))

    def test_time_distributed_batch_first_false_preserves_values(self) -> None:
        """
        Regression test: TimeDistributed.forward() used to reshape the module's output
        back with view(input_data.size(1), input_data.size(0), output_dim) when
        batch_first=False, but the flattened output was produced by squashing the
        original leading dimensions in their original (dim0, dim1) order - so
        unflattening with the dimensions swapped scrambled which values ended up at
        which (timestep, batch) position, even though the shape (T, B, 1) looked
        correct. Using an Identity module with distinguishable per-position values
        catches this at the value level, not just the shape level.

        Returns:
            None
        """
        # input shaped (T=2, B=3, F=1) for batch_first=False; each value encodes its
        # own (timestep, batch) position so any scrambling during reshape is
        # detectable
        input_data = torch.tensor(
            [
                [[0.0], [1.0], [2.0]],
                [[100.0], [101.0], [102.0]],
            ],
            dtype=torch.float32,
            device=AVAILABLE_DEVICE,
        )
        time_distributed = TimeDistributed(
            module=torch.nn.Identity(), batch_first=False
        )
        actual_output = time_distributed(input_data)
        self.assertEqual(actual_output.shape, torch.Size([2, 3, 1]))
        self.assertTrue(torch.equal(actual_output, input_data))
