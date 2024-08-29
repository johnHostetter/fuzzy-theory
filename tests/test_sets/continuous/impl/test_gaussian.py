"""
Test the Gaussian fuzzy set (i.e., membership function).
"""

import unittest

import torch
import numpy as np

from fuzzy.sets.continuous.impl import Gaussian

from tests.test_sets.continuous.impl.common import get_test_elements


AVAILABLE_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def gaussian_numpy(element: np.ndarray, center: np.ndarray, sigma: np.ndarray):
    """
        Gaussian membership function that receives an 'element' value, and uses
        the 'center' and 'sigma' to determine a degree of membership for 'element'.
        Implemented in Numpy and used in testing.

    Args:
        element: The element which we want to retrieve its membership degree.
        center: The center of the Gaussian fuzzy set.
        sigma: The width of the Gaussian fuzzy set.

    Returns:
        The membership degree of 'element'.
    """
    return np.exp(-1.0 * (np.power(element - center, 2) / (1.0 * np.power(sigma, 2))))


class TestGaussian(unittest.TestCase):
    """
    Test the Gaussian fuzzy set (i.e., membership function).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.elements = get_test_elements(device=AVAILABLE_DEVICE)

    def test_single_input(self) -> None:
        """
        Test that single input works for the Gaussian membership function.

        Returns:
            None
        """
        element = torch.zeros(1, device=AVAILABLE_DEVICE)
        gaussian_mf = Gaussian(
            centers=np.array([1.5409961]),
            widths=np.array([0.30742282]),
            device=AVAILABLE_DEVICE,
        )
        sigma = gaussian_mf.get_widths().cpu().detach().numpy()
        center = gaussian_mf.get_centers().cpu().detach().numpy()
        mu_pytorch = gaussian_mf(element).degrees.to_dense()
        mu_numpy = gaussian_numpy(element.cpu().detach().numpy(), center, sigma)

        # make sure the Gaussian parameters are still identical afterward
        assert torch.allclose(
            gaussian_mf.get_widths(), torch.tensor(sigma, device=AVAILABLE_DEVICE)
        )
        assert torch.allclose(
            gaussian_mf.get_centers(),
            torch.tensor(center, device=AVAILABLE_DEVICE),
        )
        # the outputs of the PyTorch and Numpy versions should be approx. equal
        assert np.allclose(mu_pytorch.cpu().detach().numpy(), mu_numpy, rtol=1e-6)

        # test that this is compatible with torch.jit.script
        gaussian_mf_scripted = torch.jit.script(gaussian_mf)
        assert torch.allclose(
            gaussian_mf_scripted(element).degrees.to_dense(),
            mu_pytorch,
        )

    def test_multi_input(self) -> None:
        """
        Test that multiple input works for the Gaussian membership function.

        Returns:
            None
        """
        gaussian_mf = Gaussian(
            centers=np.array([1.5410]),
            widths=np.array([0.3074]),
            device=AVAILABLE_DEVICE,
        )
        centers, sigmas = (
            gaussian_mf.get_centers().cpu().detach().numpy(),
            gaussian_mf.get_widths().cpu().detach().numpy(),
        )
        mu_pytorch = gaussian_mf(self.elements).degrees.to_dense()
        mu_numpy = gaussian_numpy(self.elements.cpu().detach().numpy(), centers, sigmas)

        # make sure the Gaussian parameters are still identical afterward
        assert torch.allclose(
            gaussian_mf.get_widths(),
            torch.tensor(sigmas, device=AVAILABLE_DEVICE).float(),
        )
        assert torch.allclose(
            gaussian_mf.get_centers(),
            torch.tensor(centers, device=AVAILABLE_DEVICE).float(),
        )
        # the outputs of the PyTorch and Numpy versions should be approx. equal
        # note that the PyTorch version has an extra dimension (4, 1, 1) compared to Numpy's (4, 1)
        assert np.allclose(
            mu_pytorch.cpu().detach().numpy().flatten(), mu_numpy.flatten()
        )

        # test that this is compatible with torch.jit.script
        gaussian_mf_scripted = torch.jit.script(gaussian_mf)
        assert torch.allclose(
            gaussian_mf_scripted(self.elements).degrees.to_dense(), mu_pytorch
        )

    def test_multi_input_with_centers_given(self) -> None:
        """
        Test that multiple input works for the Gaussian membership function when centers are
        specified for the fuzzy sets.

        Returns:
            None
        """
        centers = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        sigmas = np.array([0.4962566, 0.7682218, 0.08847743, 0.13203049, 0.30742282])
        gaussian_mf = Gaussian(
            centers=centers,
            widths=sigmas,
            device=AVAILABLE_DEVICE,
        )
        mu_pytorch = gaussian_mf(self.elements).degrees.to_dense()
        mu_numpy = gaussian_numpy(self.elements.cpu().detach().numpy(), centers, sigmas)

        # make sure the Gaussian parameters are still identical afterward
        assert torch.allclose(
            gaussian_mf.get_widths(),
            torch.tensor(sigmas, device=AVAILABLE_DEVICE).float(),
        )
        assert torch.allclose(
            gaussian_mf.get_centers(),
            torch.tensor(centers, device=AVAILABLE_DEVICE).float(),
        )
        # the outputs of the PyTorch and Numpy versions should be approx. equal
        assert np.allclose(
            mu_pytorch.squeeze(dim=1).cpu().detach().numpy(), mu_numpy, rtol=1e-4
        )

        expected_areas = torch.tensor(
            [0.7412324, 1.1474512, 0.13215375, 0.1972067, 0.45918167],
            device=AVAILABLE_DEVICE,
        )
        assert torch.allclose(gaussian_mf.area(), expected_areas)

        # test that this is compatible with torch.jit.script
        gaussian_mf_scripted = torch.jit.script(gaussian_mf)
        assert torch.allclose(
            gaussian_mf_scripted(self.elements).degrees.to_dense(), mu_pytorch
        )

    def test_multi_input_with_sigmas_given(self) -> None:
        """
        Test that multiple input works for the Gaussian membership function when sigmas are
        specified for the fuzzy sets.

        Returns:
            None
        """
        sigmas = np.array(
            [0.1, 0.25, 0.5, 0.75, 1.0]
        )  # negative widths are missing sets
        gaussian_mf = Gaussian(
            centers=np.array([1.5410]),
            widths=sigmas,
            device=AVAILABLE_DEVICE,
        )
        mu_pytorch = gaussian_mf(self.elements).degrees.to_dense()
        mu_numpy = gaussian_numpy(
            self.elements.cpu().detach().numpy(),
            gaussian_mf.get_centers().cpu().detach().numpy(),
            sigmas,
        )

        # make sure the Gaussian parameters are still identical afterward
        assert np.allclose(gaussian_mf.get_widths().cpu().detach().numpy(), sigmas)
        # the outputs of the PyTorch and Numpy versions should be approx. equal
        assert np.allclose(
            mu_pytorch.squeeze(dim=1).cpu().detach().numpy(), mu_numpy, rtol=1e-6
        )

        # test that this is compatible with torch.jit.script
        gaussian_mf_scripted = torch.jit.script(gaussian_mf)
        assert torch.allclose(
            gaussian_mf_scripted(self.elements).degrees.to_dense(), mu_pytorch
        )

    def test_multi_input_with_both_given(self) -> None:
        """
        Test that multiple input works for the Gaussian membership function when centers and
        sigmas are specified for the fuzzy sets.

        Returns:
            None
        """
        centers = np.array([-0.5, -0.25, 0.25, 0.5, 0.75])
        sigmas = np.array(
            [0.1, 0.25, 0.5, 0.75, 1.0]
        )  # negative widths are missing sets
        gaussian_mf = Gaussian(
            centers=centers,
            widths=sigmas,
            device=AVAILABLE_DEVICE,
        )
        mu_pytorch = gaussian_mf(self.elements).degrees.to_dense()
        mu_numpy = gaussian_numpy(self.elements.cpu().detach().numpy(), centers, sigmas)

        # make sure the Gaussian parameters are still identical afterward
        assert np.allclose(gaussian_mf.get_centers().cpu().detach().numpy(), centers)
        assert np.allclose(gaussian_mf.get_widths().cpu().detach().numpy(), sigmas)
        # the outputs of the PyTorch and Numpy versions should be approx. equal
        assert np.allclose(
            mu_pytorch.squeeze(dim=1).cpu().detach().numpy(), mu_numpy, rtol=1e-6
        )

        # test that this is compatible with torch.jit.script
        gaussian_mf_scripted = torch.jit.script(gaussian_mf)
        assert torch.allclose(
            gaussian_mf_scripted(self.elements).degrees.to_dense(), mu_pytorch
        )

    def test_multi_centers(self) -> None:
        """
        Test that multidimensional centers work with the Gaussian membership function.

        Returns:
            None
        """
        elements = torch.tensor(
            [
                [
                    [0.6960, 0.8233, 0.8147],
                    [0.1024, 0.3122, 0.5160],
                    [0.8981, 0.6810, 0.2366],
                ],
                [
                    [0.2447, 0.4218, 0.6146],
                    [0.8887, 0.6273, 0.6697],
                    [0.1439, 0.9383, 0.8101],
                ],
            ],
            device=AVAILABLE_DEVICE,
        )
        centers = np.array(
            [
                [
                    [0.1236, 0.4893, 0.8372],
                    [0.8275, 0.2979, 0.7192],
                    [0.2328, 0.1418, 0.1036],
                ],
                [
                    [0.9651, 0.7622, 0.1544],
                    [0.1274, 0.5798, 0.6425],
                    [0.1518, 0.6554, 0.3799],
                ],
            ]
        )
        sigmas = np.array([[[0.1, 0.25, 0.5]]])  # negative widths are missing sets
        gaussian_mf = Gaussian(
            centers=centers,
            widths=sigmas,
            device=AVAILABLE_DEVICE,
        )
        mu_pytorch = gaussian_mf(elements.unsqueeze(dim=0)).degrees.to_dense()
        mu_numpy = gaussian_numpy(elements.cpu().detach().numpy(), centers, sigmas)

        # make sure the Gaussian parameters are still identical afterward
        assert np.allclose(gaussian_mf.get_centers().cpu().detach().numpy(), centers)
        assert np.allclose(gaussian_mf.get_widths().cpu().detach().numpy(), sigmas)
        # the outputs of the PyTorch and Numpy versions should be approx. equal
        assert np.allclose(
            mu_pytorch.squeeze(dim=0).cpu().detach().numpy(), mu_numpy, rtol=1e-6
        )

        # test that this is compatible with torch.jit.script
        gaussian_mf_scripted = torch.jit.script(gaussian_mf)
        assert torch.allclose(
            gaussian_mf_scripted(
                elements.unsqueeze(dim=0),  # add batch dimension (size is 1)
            ).degrees.to_dense(),
            mu_pytorch,
        )

    def test_consistency(self) -> None:
        """
        Test that the results are consistent with the expected membership degrees.

        Returns:
            None
        """
        element = np.array(
            [[0.0001712], [0.00393354], [-0.03641258], [-0.01936134]], dtype=np.float32
        )
        centers = np.array(
            [
                [0.01497397, -1.3607662, 1.0883657, 1.9339248],
                [-0.01367673, 2.3560243, -1.8339163, -3.3379893],
                [-4.489564, -0.01467094, -0.13278057, 0.08638719],
                [0.17008819, 0.01596639, -1.7408595, 2.797653],
            ]
        )
        sigmas = np.array(
            [
                [1.16553577, 1.48497267, 0.91602303, 0.91602303],
                [1.98733806, 2.53987592, 1.58646032, 1.24709336],
                [1.24709336, 0.10437003, 0.12908118, 0.08517358],
                [0.08517358, 1.54283158, 1.89779089, 1.27380911],
            ]
        )
        target_membership_degrees = torch.tensor(
            gaussian_numpy(
                element[:, :, None],
                center=centers,
                sigma=sigmas,
            ),
            device=AVAILABLE_DEVICE,
            dtype=torch.float32,
        )

        gaussian_mf = Gaussian(
            centers=centers,
            widths=sigmas,
            device=AVAILABLE_DEVICE,
        )
        mu_pytorch = gaussian_mf(
            torch.tensor(element, device=AVAILABLE_DEVICE)
        ).degrees.to_dense()

        # make sure the Gaussian parameters are still identical afterward
        assert np.allclose(
            gaussian_mf.get_centers().cpu().detach().numpy(),
            centers,
        )
        assert np.allclose(gaussian_mf.get_widths().cpu().detach().numpy(), sigmas)
        # the outputs of the PyTorch and Numpy versions should be approx. equal
        assert torch.allclose(mu_pytorch, target_membership_degrees, rtol=1e-1)

        # test that this is compatible with torch.jit.script
        gaussian_mf_scripted = torch.jit.script(gaussian_mf)
        assert torch.allclose(
            gaussian_mf_scripted(
                torch.tensor(element, device=AVAILABLE_DEVICE)
            ).degrees.to_dense(),
            mu_pytorch,
        )

    def test_create_random(self) -> None:
        """
        Test that a random fuzzy set of this type can be created and that the results are consistent
        with the expected membership degrees.

        Returns:
            None
        """
        gaussian_mf = Gaussian.create(n_variables=4, n_terms=4, device=AVAILABLE_DEVICE)
        element = np.array([[0.0001712, 0.00393354, -0.03641258, -0.01936134]])
        target_membership_degrees = gaussian_numpy(
            element.reshape(4, 1),  # column vector
            gaussian_mf.get_centers().cpu().detach().numpy(),
            gaussian_mf.get_widths().cpu().detach().numpy(),
        )
        mu_pytorch = gaussian_mf(
            torch.tensor(element, device=AVAILABLE_DEVICE)
        ).degrees.to_dense()
        assert np.allclose(
            mu_pytorch.cpu().detach().numpy(), target_membership_degrees, atol=1e-1
        )

        # test that this is compatible with torch.jit.script
        gaussian_mf_scripted = torch.jit.script(gaussian_mf)
        assert torch.allclose(
            gaussian_mf_scripted(
                torch.tensor(element, device=AVAILABLE_DEVICE)
            ).degrees.to_dense(),
            mu_pytorch,
        )
