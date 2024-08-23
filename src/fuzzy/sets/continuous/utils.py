"""
Utility functions, such as for getting all the subclasses of a given class.
"""

import inspect
from typing import Dict, Any

import torch


def get_object_attributes(obj_instance) -> Dict[str, Any]:
    """
    Get the attributes of an object instance.
    """
    # get the attributes that are local to the class, but may be inherited from the super class
    local_attributes = inspect.getmembers(
        obj_instance,
        lambda attr: not (inspect.ismethod(attr)) and not (inspect.isfunction(attr)),
    )
    # get the attributes that are inherited from (or found within) the super class
    super_attributes = inspect.getmembers(
        obj_instance.__class__.__bases__[0],
        lambda attr: not (inspect.ismethod(attr)) and not (inspect.isfunction(attr)),
    )
    # get the attributes that are local to the class, but not inherited from the super class
    return {
        attr: value
        for attr, value in local_attributes
        if (attr, value) not in super_attributes and not attr.startswith("_")
    }


def regulator(sigma_1: torch.Tensor, sigma_2: torch.Tensor) -> torch.Tensor:
    """
    Regulator function as defined in CLIP.

    Args:
        sigma_1: The left sigma/width.
        sigma_2: The right sigma/width.

    Returns:
        sigma (float): An adjusted sigma so that the produced
        Gaussian membership function is not warped.
    """
    return (1 / 2) * (sigma_1 + sigma_2)


def find_widths(data_point, minimums, maximums, alpha: float) -> torch.Tensor:
    """
    Find the centers and widths to be used for a newly created fuzzy set.

    Args:
        data_point (1D Numpy array): A single input_data observation where
            each column is a feature/attribute.
        minimums (iterable): The minimum value per feature in X.
        maximums (iterable): The maximum value per feature in X.
        alpha (float): A hyperparameter to adjust the generated widths' coverage.

    Returns:
        A list of dictionaries, where each dictionary contains the center and width
        for a newly created fuzzy set (that is to be created later).
    """
    # The variable 'theta' is added to accommodate for the instance in which an observation has
    # values that are the minimum/maximum. Otherwise, when determining the Gaussian membership,
    # a division by zero will occur; it essentially acts as an error tolerance.
    theta: float = 1e-8
    sigmas = torch.empty((0, 0))
    for dim, attribute_value in enumerate(data_point):
        left_width: torch.Tensor = torch.sqrt(
            -1.0
            * (
                torch.pow((minimums[dim] - attribute_value) + theta, 2)
                / torch.log(torch.as_tensor([alpha], device=attribute_value.device))
            )
        )
        right_width: torch.Tensor = torch.sqrt(
            -1.0
            * (
                torch.pow((maximums[dim] - attribute_value) + theta, 2)
                / torch.log(torch.as_tensor([alpha], device=attribute_value.device))
            )
        )
        aggregated_sigma: torch.Tensor = regulator(left_width, right_width)
        if sigmas.shape[0] == 0:
            sigmas = aggregated_sigma
        else:
            sigmas = torch.hstack((sigmas, aggregated_sigma))

    return sigmas
