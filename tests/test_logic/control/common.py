"""
Shared "missing data" test fixtures used by both the Mamdani and TSK controller tests.
Both controllers are built from the same antecedent centers/widths and exercised against
the same 6-observation batch (with one NaN input), so both hit identical behavior in
FuzzyLogicController's input-granulation layer. Only the resulting rule activations differ
(different rule/consequence structures) - see the callers for those expected values.
"""

from typing import List, Tuple

import numpy as np
import torch

from fuzzy.logic.control.controller import FuzzyLogicController as FLC
from fuzzy.logic.knowledge_base import KnowledgeBase
from fuzzy.logic.rule import Rule
from fuzzy.logic.variables import LinguisticVariables
from fuzzy.relations.t_norm import Product
from fuzzy.sets.membership import Membership
from tests import AVAILABLE_DEVICE

from .demo_flcs import toy_mamdani


def build_mamdani_knowledge_base(
    device: torch.device,
) -> Tuple[KnowledgeBase, List[Rule]]:
    """
    Returns:
        A KnowledgeBase built from the toy Mamdani antecedents/consequents/rules,
        and the original list of Rule objects passed into it (distinct from
        knowledge_base.rules - the recovered rules read back out of the graph - so
        callers can still compare the two), shared by every test that just needs a
        valid Mamdani source.
    """
    antecedents, consequents, rules = toy_mamdani(
        t_norm=Product, device=device)
    knowledge_base = KnowledgeBase.create(
        linguistic_variables=LinguisticVariables(
            inputs=antecedents, targets=consequents
        ),
        rules=rules,
    )
    return knowledge_base, rules


def make_data_with_missing(device: torch.device) -> torch.Tensor:
    """
    Returns:
        A batch of 6 observations for 2 input variables, where the last observation's
        first feature is NaN (missing).
    """
    return torch.tensor(
        [
            [1.2, 0.2],
            [1.1, 0.3],
            [2.1, 0.1],
            [2.7, 0.15],
            [1.7, 0.25],
            [np.nan, 0.25],
        ],
        device=device,
    )


def assert_input_granulation_handles_missing_data(
    flc: FLC, data_with_missing: torch.Tensor, device: torch.device
) -> None:
    """
    Assert the shared input-granulation layer correctly propagates the missing (NaN)
    observation: the output shape is unaffected, and the resulting degrees match the
    expected values (NaN preserved for the affected term, real values elsewhere).

    Returns:
        None
    """
    assert flc.input_granulation(data_with_missing).degrees.shape == (6, 2, 4)
    assert torch.allclose(
        flc.input_granulation(data_with_missing).degrees.to_dense().float(),
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
            device=device,
        ),
        rtol=3e-3,
        atol=3e-3,
        equal_nan=True,
    )


def assert_compile_fullgraph_matches_eager(
    flc: FLC, input_data: torch.Tensor, atol: float = 1e-6
) -> None:
    """
    Assert that wrapping flc with torch.compile(fullgraph=True) does not change its
    output relative to plain eager execution. fullgraph=True requires Dynamo to trace
    the entire forward pass with zero graph breaks - a stricter guarantee than the
    default torch.compile (which silently falls back to eager for any untraceable
    piece) - so this also proves nothing in the traced call path secretly depends on
    Python-only behavior torch.compile cannot represent.

    Returns:
        None
    """
    eager_output = flc(input_data)
    compiled_output = torch.compile(flc, fullgraph=True)(input_data)
    assert torch.allclose(
        eager_output,
        compiled_output,
        atol=atol,
        equal_nan=True)


def _assert_rule_activations_match(
    actual_degrees: torch.Tensor, expected_values, device: torch.device
) -> None:
    """
    Assert a rule-activation tensor (lower/upper bound, with or without NaN
    replacement) matches its expected values.

    Returns:
        None
    """
    assert torch.allclose(
        actual_degrees,
        torch.tensor(expected_values, device=device),
        rtol=3e-3,
        atol=3e-3,
        equal_nan=True,
    )


# pylint: disable-next=too-many-arguments,too-many-positional-arguments
def _assert_engine_handles_missing_data(
    flc: FLC,
    data_with_missing: torch.Tensor,
    device: torch.device,
    expected_lower,
    expected_temp_upper,
    expected_upper,
    sort_lower: bool = False,
) -> None:
    """
    Walk the inference engine through the documented missing-data protocol - the lower
    bound (nan_replacement=0.0), the intermediate rule mask before aggregation
    (nan_replacement=1.0, via apply_mask()), and the upper bound (the same
    nan_replacement=1.0, via a full forward pass) - asserting each stage against its
    expected values. sort_lower accounts for Mamdani's rule ordering not being stable
    across its sparse mask, unlike TSK's.

    Returns:
        None
    """
    flc.engine.nan_replacement = 0.0
    lower_rule_activations: Membership = flc.engine(
        flc.input_granulation(data_with_missing)
    )
    actual_lower = (
        lower_rule_activations.degrees.sort().values
        if sort_lower
        else lower_rule_activations.degrees
    )
    _assert_rule_activations_match(actual_lower, expected_lower, device)

    flc.engine.nan_replacement = 1.0
    temp_upper_rule_activations: torch.Tensor = flc.engine.apply_mask(
        flc.input_granulation(data_with_missing)
    )
    _assert_rule_activations_match(
        temp_upper_rule_activations, expected_temp_upper, device
    )

    upper_rule_activations: Membership = flc.engine(
        flc.input_granulation(data_with_missing)
    )
    _assert_rule_activations_match(
        upper_rule_activations.degrees, expected_upper, device
    )


# pylint: disable-next=too-few-public-methods
class MissingDataHandlingMixin:
    """
    Shared test_missing_data_handling implementation, mixed into TestMamdani and
    TestTSK alongside unittest.TestCase. Both controllers are built from the same
    2-variable, 4-granule antecedents make_data_with_missing() targets, so the test
    procedure itself is identical; only the resulting rule activations differ (each
    controller has its own rule/consequence structure), which subclasses supply via
    _missing_data_expected().
    """

    fuzzy_logic_controller: FLC

    def _missing_data_expected(self) -> dict:
        """
        Returns:
            Keyword arguments (expected_lower/expected_temp_upper/expected_upper, and
            optionally sort_lower) for _assert_engine_handles_missing_data().
        """
        raise NotImplementedError

    def test_missing_data_handling(self) -> None:
        """
        Test that the FLC can handle missing data.

        Returns:
            None
        """
        data_with_missing = make_data_with_missing(AVAILABLE_DEVICE)
        assert_input_granulation_handles_missing_data(
            self.fuzzy_logic_controller, data_with_missing, AVAILABLE_DEVICE
        )
        _assert_engine_handles_missing_data(
            self.fuzzy_logic_controller,
            data_with_missing,
            AVAILABLE_DEVICE,
            **self._missing_data_expected(),
        )
