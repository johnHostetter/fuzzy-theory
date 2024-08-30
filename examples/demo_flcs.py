"""
Demo of the ZeroOrderTSK or Mamdani FLC and how to use it when doing 'expert design'
(i.e., manually writing fuzzy sets and fuzzy logic rules).
"""

from copy import deepcopy
from typing import Tuple, List, Any, Type

import torch
import numpy as np
from fuzzy.logic.rule import Rule
from fuzzy.logic.knowledge_base import KnowledgeBase
from fuzzy.logic.variables import LinguisticVariables
from fuzzy.logic.control.defuzzification import ZeroOrder, Mamdani
from fuzzy.logic.control.controller import FuzzyLogicController as FLC
from fuzzy.relations.t_norm import TNorm, Product
from fuzzy.relations.n_ary import NAryRelation
from fuzzy.sets.impl import Gaussian


AVAILABLE_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_premises(device: torch.device) -> List[Gaussian]:
    """
    Get the premises that should be used in both the TSK and Mamdani FLC examples.

    Args:
        device: The device to use for the Gaussian objects.

    Returns:
        The linguistic terms that are defined over the input dimensions.
    """
    premises = [
        Gaussian(
            centers=np.array([1.2, 3.0, 5.0, 7.0]),
            widths=np.array([0.1, 0.4, 0.6, 0.8]),
            device=device,
        ),
        Gaussian(
            centers=np.array([0.2, 0.6, 0.9, 1.2]),
            widths=np.array([0.4, 0.4, 0.5, 0.45]),
            device=device,
        ),
    ]
    return premises


def toy_tsk(
    t_norm: Type[TNorm], device: torch.device
) -> Tuple[List[Gaussian], List[Any], List[Rule]]:
    """
    A toy example for defining some TSK Fuzzy Logic Controller.

    Args:
        t_norm: The t-norm to use for the fuzzy logic rules.
        device: The device to use for the PyTorch objects.

    Returns:
        A list of antecedents and a set of rules.
    """
    rules = [
        Rule(
            premise=t_norm((0, 0), (1, 0), device=device),
            consequence=NAryRelation((0, 0), device=AVAILABLE_DEVICE),
        ),
        Rule(
            premise=t_norm((0, 0), (1, 1), device=device),
            consequence=NAryRelation((0, 1), device=AVAILABLE_DEVICE),
        ),
        Rule(
            premise=t_norm((0, 1), (1, 0), device=device),
            consequence=NAryRelation((0, 2), device=AVAILABLE_DEVICE),
        ),
        Rule(
            premise=t_norm((0, 1), (1, 1), device=device),
            consequence=NAryRelation((0, 3), device=AVAILABLE_DEVICE),
        ),
        Rule(
            premise=t_norm((0, 1), (1, 2), device=device),
            consequence=NAryRelation((0, 4), device=AVAILABLE_DEVICE),
        ),
    ]
    return get_premises(device=device), [], rules


def toy_mamdani(
    t_norm: Type[TNorm],
    device: torch.device,
) -> Tuple[List[Gaussian], List[Gaussian], List[Rule]]:
    """
    A toy example for defining some Mamdani Fuzzy Logic Controller.

    Args:
        t_norm: The t-norm to use for the fuzzy logic rules.
        device: The device to use for the PyTorch objects.

    Returns:
        A list of antecedents, a list of consequents, and a list of rules.
    """
    consequences = [
        Gaussian(
            centers=np.array([0.5, 0.3]),
            widths=np.array([0.1, 0.4]),
            device=device,
        ),
        Gaussian(
            centers=np.array([-0.2, -0.7, -0.9]),
            widths=np.array([0.4, 0.4, 0.5]),
            device=device,
        ),
    ]
    rules = [
        Rule(
            premise=t_norm((0, 0), (1, 0), device=device),
            consequence=NAryRelation((0, 0), (1, 1), device=device),
            # used to have to start counting from input variables' size
            # consequence=NAryRelation((2, 0), (3, 1), device=device),
        ),
        Rule(
            premise=t_norm((0, 1), (1, 0), device=device),
            consequence=NAryRelation((0, 1), (1, 2), device=device),
            # consequence=NAryRelation((2, 1), (3, 2), device=device),
        ),
        Rule(
            premise=t_norm((0, 1), (1, 1), device=device),
            consequence=NAryRelation((0, 0), (1, 0), device=device),
            # consequence=NAryRelation((2, 0), (3, 0), device=device),
        ),
    ]
    return get_premises(device=device), consequences, rules


def print_parameters(fuzzy_logic_controller) -> None:
    """
    Print the parameters of the given fuzzy logic controller to the terminal.

    Args:
        fuzzy_logic_controller: Either a ZeroOrderTSK or Mamdani FLC.

    Returns:
        None
    """
    print(f"sigmas: {fuzzy_logic_controller.input.widths}")
    print(f"centers: {fuzzy_logic_controller.input.centers}")


def train_model(model, input_x, target_y):
    """
    Train the model to map the input to the target.

    Args:
        model:
        input_x:
        target_y:

    Returns:

    """
    # setup training

    # define loss function
    criterion = torch.nn.MSELoss()
    # define learning rate
    learning_rate = 3e-2
    # define number of epochs
    epochs = 300
    # initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # run training and print the loss to make sure that we are actually fitting to the training set
    print("Training the model. Make sure that loss decreases after each epoch.\n")

    losses = []
    num_of_epochs = 0
    while (len(losses) > 1 and losses[-2] > losses[-1]) or num_of_epochs < epochs:
        # print(num_of_epochs)
        params_before = deepcopy(list(model.parameters()))
        prediction = model(input_x)
        loss = criterion(prediction, target_y)
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        params_after = deepcopy(list(model.parameters()))
        try:
            if not all([(b == a).all() for b, a in zip(params_before, params_after)]):
                print("updating optimizer")
                optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        except RuntimeError:
            print("updating optimizer")
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # if num_of_epochs == 0:
        #     if model.granulation_layers["input"].centers.grad is None:
        #         print("Error: The gradients for the antecedents' centers is None.")
        #     if isinstance(model.consequences(), Gaussian):
        #         if model.consequences().centers.grad is None:
        #             print("Error: The gradients for the consequences is None.")
        #     else:
        #         if model.consequences().grad is None:
        #             print("Error: The gradients for the consequences is None.")
        optimizer.step()
        num_of_epochs += 1
        losses.append(loss.item())
    print(losses[-1])


if __name__ == "__main__":
    x = torch.tensor(
        [[1.2, 0.2], [1.1, 0.3], [2.1, 0.1], [2.7, 0.15], [1.7, 0.25]],
        dtype=torch.float32,
        device=AVAILABLE_DEVICE,
    )
    for flc_type in [toy_tsk, toy_mamdani]:
        y = torch.tensor(
            np.array([[0.5, 2.5], [0.1, 1.5], [0.75, 1.0], [0.8, 2.3], [2.0, 0.3]]),
            dtype=torch.float32,
            device=AVAILABLE_DEVICE,
        )
        print(f"({x}, {y})")

        antecedents, consequents, fuzzy_rules = flc_type(
            t_norm=Product, device=AVAILABLE_DEVICE
        )
        knowledge_base = KnowledgeBase.create(
            linguistic_variables=LinguisticVariables(
                inputs=antecedents, targets=consequents
            ),
            rules=fuzzy_rules,
        )
        if flc_type == toy_tsk:
            flc = FLC(
                source=knowledge_base,
                inference=ZeroOrder,
                device=AVAILABLE_DEVICE,
            )
        elif flc_type == toy_mamdani:
            flc = FLC(
                source=knowledge_base,
                inference=Mamdani,
                device=AVAILABLE_DEVICE,
            )
        print(f"Demonstrating {flc}")

        predicted_y = flc(x)
        print(f"Predicted output before training:{predicted_y}")
        print_parameters(flc)

        train_model(flc, x, y)

        new_predicted_y = flc(x)
        print(f"Predicted output after training:{new_predicted_y}")
        print_parameters(flc)
