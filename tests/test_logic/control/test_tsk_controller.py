"""
Test TSK fuzzy systems are working as intended (e.g., their output is correctly calculated).
"""

# a couple of tests deliberately reach into FuzzyLogicController._locate_engine_save_dir
# (a white-box test of its own save/load-directory-layout contract)
# pylint: disable=protected-access

import shutil
import unittest
from collections import OrderedDict
from pathlib import Path
from typing import List, Tuple
from unittest import mock

import numpy as np
import torch

from fuzzy.logic.control.configurations.data import ExecutionOptions
from fuzzy.logic.control.controller import FuzzyLogicController as FLC
from fuzzy.logic.control.defuzzification import TSK, ZeroOrder
from fuzzy.logic.knowledge_base import KnowledgeBase
from fuzzy.logic.rule import Rule
from fuzzy.logic.variables import LinguisticVariables
from fuzzy.relations.n_ary import NAryRelation
from fuzzy.relations.t_norm import Product
from fuzzy.sets.abstract import FuzzySet
from fuzzy.sets.impl import Gaussian
from fuzzy.utils import load_module_class, module_class
from tests import AVAILABLE_DEVICE

from .common import MissingDataHandlingMixin, assert_compile_fullgraph_matches_eager


# pylint: disable-next=too-many-public-methods
class TestTSK(MissingDataHandlingMixin, unittest.TestCase):
    """
    Test the zero-order TSK neuro-fuzzy network.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        (
            self.fuzzy_logic_controller,
            self.input_data,
            self.rules,
        ) = self.test_create_tsk()

    def test_gradient_1(self) -> None:
        """
        First test that the gradient of PyTorch is working as intended.

        Returns:
            None
        """
        input_data = torch.tensor(
            [[1.2, 0.2], [1.1, 0.3], [2.1, 0.1], [2.7, 0.15], [1.7, 0.25]],
            device=AVAILABLE_DEVICE,
        )
        # the first variable has fuzzy sets with centers 0, 1, 2 (the column)
        centers = torch.nn.Parameter(
            torch.tensor([[0, 1], [1, 2], [2, 3]], device=AVAILABLE_DEVICE).double()
        )
        actual_result = input_data.unsqueeze(dim=-1) - centers.T
        expected_result = torch.tensor(
            [
                [[1.2000, 0.2000, -0.8000], [-0.8000, -1.8000, -2.8000]],
                [[1.1000, 0.1000, -0.9000], [-0.7000, -1.7000, -2.7000]],
                [[2.1000, 1.1000, 0.1000], [-0.9000, -1.9000, -2.9000]],
                [[2.7000, 1.7000, 0.7000], [-0.8500, -1.8500, -2.8500]],
                [[1.7000, 0.7000, -0.3000], [-0.7500, -1.7500, -2.7500]],
            ],
            device=AVAILABLE_DEVICE,
        )

        assert torch.allclose(actual_result.float(), expected_result)

    def test_gradient_2(self) -> None:
        """
        Second test that the gradient of PyTorch is working as intended.

        Returns:
            None
        """
        value_1 = torch.nn.Parameter(
            torch.tensor([0, 1], device=AVAILABLE_DEVICE).double()
        )
        value_3 = 2**value_1
        assert value_3.grad_fn is not None

    def test_to_device(self) -> None:
        """
        Check that we can move the FLC to another device.

        Returns:
            None
        """
        self.fuzzy_logic_controller.to(torch.device("cpu"))
        self.assertEqual(torch.device("cpu"), self.fuzzy_logic_controller.device)
        # check that this is reflected in each of its torch.nn.Modules
        for module in self.fuzzy_logic_controller.children():
            self.assertEqual(torch.device("cpu"), module.device)
        self.fuzzy_logic_controller.to(AVAILABLE_DEVICE)
        self.assertEqual(AVAILABLE_DEVICE, self.fuzzy_logic_controller.device)
        # check that this is reflected in each of its torch.nn.Modules
        for module in self.fuzzy_logic_controller.children():
            self.assertEqual(AVAILABLE_DEVICE, module.device)

    def test_tsk(self) -> None:
        """
        Test the zero-order TSK neuro-fuzzy network.

        Returns:
            None
        """
        # check that the output is correct
        predicted_y = self.fuzzy_logic_controller(self.input_data)
        self.assertIsNotNone(predicted_y.grad_fn)
        self.assertEqual(predicted_y.shape, torch.Size([5, 1]))

        # check that the input granulation was correctly created
        assert torch.allclose(
            self.fuzzy_logic_controller.input_granulation.centers,
            torch.tensor(
                [[1.2000, 3.0000, 5.0000, 7.0000], [0.2000, 0.6000, 0.9000, 1.2000]],
                device=AVAILABLE_DEVICE,
            ),
        )
        assert torch.allclose(
            self.fuzzy_logic_controller.input_granulation.widths,
            torch.tensor(
                [[0.1000, 0.4000, 0.6000, 0.8000], [0.4000, 0.4000, 0.5000, 0.4500]],
                device=AVAILABLE_DEVICE,
            ),
        )
        # check that the output granulation was correctly created
        self.assertEqual(
            self.fuzzy_logic_controller.defuzzification.consequences.shape,
            torch.Size([4, 1]),
        )

    def test_forward_matches_independent_numpy_reference(self) -> None:
        """
        Golden-value/drift-detection test: test_tsk() above only checks
        predicted_y's shape, never its values - TSK.consequences are randomly
        initialized (torch.randn, unseeded) whenever an FLC is built through the
        normal FLC(source=knowledge_base, inference=TSK, ...) path (see
        configurations/abstract.py's defuzzification(), which always passes
        source=None for non-Mamdani inference types), so self.fuzzy_logic_controller
        itself produces a different output every run and cannot be pinned directly.

        Works around that by building a small, fully deterministic TSK FLC here:
        one input variable, two Gaussian terms (centers 0.0 and 2.0, width 1.0),
        one rule per term, and explicit (not randomly initialized) consequences
        injected via TSK's own source= constructor argument (the same mechanism
        save()/load() already round-trip through) - rule 0's consequence is
        y = 1.0*x + 0.0, rule 1's is y = -1.0*x + 5.0.

        The expected output is computed independently in NumPy (Gaussian
        membership, product t-norm - trivial for a single-term rule, normalized
        firing-strength weighting, and the per-rule linear consequences) rather
        than by calling any of the FLC's own building blocks, so this catches a
        regression in the underlying math - not just a wiring/threading bug two
        copies of the same formula would share.

        Returns:
            None
        """
        antecedent = Gaussian(
            centers=np.array([0.0, 2.0]),
            widths=np.array([1.0, 1.0]),
            device=AVAILABLE_DEVICE,
        )
        rules = [
            Rule(
                premise=Product((0, 0), device=AVAILABLE_DEVICE),
                consequence=NAryRelation((0, 0), device=AVAILABLE_DEVICE),
            ),
            Rule(
                premise=Product((0, 1), device=AVAILABLE_DEVICE),
                consequence=NAryRelation((0, 1), device=AVAILABLE_DEVICE),
            ),
        ]
        knowledge_base = KnowledgeBase.create(
            linguistic_variables=LinguisticVariables(inputs=[antecedent], targets=[]),
            rules=rules,
        )
        flc = FLC(source=knowledge_base, inference=TSK, device=AVAILABLE_DEVICE)
        # rule 0: y = 1.0*x + 0.0 ; rule 1: y = -1.0*x + 5.0 - shape is
        # (n_outputs=1, n_rules=2, n_inputs + 1=2), first column is bias (see
        # TSK.consequences)
        deterministic_consequences = np.array(
            [[[0.0, 1.0], [5.0, -1.0]]], dtype=np.float32
        )
        flc.defuzzification = TSK(
            shape=flc.defuzzification.shape,
            source=deterministic_consequences,
            device=AVAILABLE_DEVICE,
            rule_base=None,
        )

        input_data = torch.tensor([[0.5], [1.5], [1.0]], device=AVAILABLE_DEVICE)
        predicted_y = flc(input_data)
        self.assertIsNotNone(predicted_y.grad_fn)

        # vectorized over both rules at once (columns), rather than one named
        # variable pair per rule, to keep this within pylint's too-many-locals
        # budget
        observations = input_data.cpu().detach().numpy()  # (N, 1)
        term_centers = np.array([0.0, 2.0])
        memberships = np.exp(-1.0 * np.power(observations - term_centers, 2))
        firing_strengths = memberships / memberships.sum(axis=1, keepdims=True)
        rule_outputs = observations * np.array([1.0, -1.0]) + np.array([0.0, 5.0])
        expected = (firing_strengths * rule_outputs).sum(axis=1)

        self.assertTrue(
            np.allclose(
                predicted_y.cpu().detach().numpy().flatten(), expected, atol=1e-5
            )
        )

    def test_save_and_load_round_trip(self) -> None:
        """
        Coverage/regression test: FuzzyLogicController.save()/.load() - and the
        configurations.impl.Defined class load() reconstructs a FuzzySystem
        through - had no test coverage at all. A loaded FLC must produce identical
        output to the original for the same input.

        Returns:
            None
        """
        path = Path("test_flc_save_and_load")
        self.fuzzy_logic_controller.save(path)
        try:
            loaded_flc: FLC = FLC.load(path, device=AVAILABLE_DEVICE)

            self.assertEqual(
                loaded_flc.shape.n_inputs, self.fuzzy_logic_controller.shape.n_inputs
            )
            self.assertEqual(
                loaded_flc.shape.n_outputs, self.fuzzy_logic_controller.shape.n_outputs
            )

            original_output = self.fuzzy_logic_controller(self.input_data)
            loaded_output = loaded_flc(self.input_data)
            self.assertTrue(torch.allclose(original_output, loaded_output))
        finally:
            shutil.rmtree(path, ignore_errors=True)

    def test_engine_is_saved_under_a_module_path_named_subdirectory(self) -> None:
        """
        Contract test: external tooling relies on
        [flc_dir]/engine/[module path to the engine's class] as the on-disk
        layout, so it can determine (and dynamically import) the engine's
        concrete type - including a custom TNorm subclass not bundled with
        fuzzy-theory - straight from the directory listing, without unpickling
        anything first.

        Returns:
            None
        """
        path = Path("test_flc_engine_save_layout")
        self.fuzzy_logic_controller.save(path)
        try:
            engine_dir = path / "engine"
            subdirs = [entry.name for entry in engine_dir.iterdir() if entry.is_dir()]
            self.assertEqual(
                subdirs, [module_class(self.fuzzy_logic_controller.engine)]
            )
            self.assertTrue((engine_dir / subdirs[0] / "state_dict.pt").is_file())
        finally:
            shutil.rmtree(path, ignore_errors=True)

    def test_load_dynamically_imports_the_engines_module(self) -> None:
        """
        FLC.load() must import the engine's module (via load_module_class) before
        it can resolve the engine's own class object by name - otherwise a custom
        TNorm subclass whose module has never been imported would not be
        discoverable by TorchJitModule.get_subclass() (which can only search
        classes Python already knows about).

        Returns:
            None
        """
        path = Path("test_flc_engine_dynamic_import")
        self.fuzzy_logic_controller.save(path)
        try:
            with mock.patch(
                "fuzzy.logic.control.controller.load_module_class",
                wraps=load_module_class,
            ) as mocked_load_module_class:
                FLC.load(path, device=AVAILABLE_DEVICE)
            mocked_load_module_class.assert_called_once_with(
                module_class(self.fuzzy_logic_controller.engine)
            )
        finally:
            shutil.rmtree(path, ignore_errors=True)

    def test_load_dispatches_through_the_engines_own_class(self) -> None:
        """
        FLC.load() must call load() through the engine's own concrete class
        (e.g. Product.load(...)) rather than through the base NAryRelation.load(...)
        directly, so that a custom TNorm subclass which overrides load() with extra
        behavior actually has that override invoked instead of silently bypassed.
        Patching Product.load specifically (not NAryRelation.load) and asserting it
        was called proves dispatch went through the subclass, since a bare
        NAryRelation.load(...) call would not go through this patched attribute.

        Returns:
            None
        """
        path = Path("test_flc_engine_load_dispatch")
        self.fuzzy_logic_controller.save(path)
        try:
            with mock.patch.object(
                Product, "load", wraps=Product.load
            ) as mocked_product_load:
                FLC.load(path, device=AVAILABLE_DEVICE)
            mocked_product_load.assert_called_once()
        finally:
            shutil.rmtree(path, ignore_errors=True)

    def test_locate_engine_save_dir_returns_the_single_subdirectory(self) -> None:
        """
        Returns:
            None
        """
        path = Path("test_locate_engine_save_dir_single")
        try:
            engine_dir = path / "engine"
            only_subdir = engine_dir / "fuzzy.relations.t_norm.Product"
            only_subdir.mkdir(parents=True)
            self.assertEqual(FLC._locate_engine_save_dir(engine_dir), only_subdir)
        finally:
            shutil.rmtree(path, ignore_errors=True)

    def test_locate_engine_save_dir_rejects_zero_subdirectories(self) -> None:
        """
        Returns:
            None
        """
        path = Path("test_locate_engine_save_dir_empty")
        try:
            engine_dir = path / "engine"
            engine_dir.mkdir(parents=True)
            with self.assertRaises(ValueError):
                FLC._locate_engine_save_dir(engine_dir)
        finally:
            shutil.rmtree(path, ignore_errors=True)

    def test_locate_engine_save_dir_rejects_multiple_subdirectories(self) -> None:
        """
        Returns:
            None
        """
        path = Path("test_locate_engine_save_dir_multiple")
        try:
            engine_dir = path / "engine"
            (engine_dir / "fuzzy.relations.t_norm.Product").mkdir(parents=True)
            (engine_dir / "fuzzy.relations.t_norm.Minimum").mkdir(parents=True)
            with self.assertRaises(ValueError):
                FLC._locate_engine_save_dir(engine_dir)
        finally:
            shutil.rmtree(path, ignore_errors=True)

    def test_linguistic_variables_rejects_unexpected_granulation_layer_count(
        self,
    ) -> None:
        """
        Coverage/regression test: linguistic_variables() must reject a
        split_granules_by_type() result with fewer than 1 or more than 2 entries
        (neither reachable via normal construction - every real FLC has exactly an
        input, and optionally an output, granulation layer - so the guard is
        exercised directly against a mocked return value).

        Returns:
            None
        """
        with mock.patch.object(
            self.fuzzy_logic_controller,
            "split_granules_by_type",
            return_value=OrderedDict(),
        ):
            with self.assertRaises(ValueError):
                self.fuzzy_logic_controller.linguistic_variables()

        with mock.patch.object(
            self.fuzzy_logic_controller,
            "split_granules_by_type",
            return_value=OrderedDict(a=[], b=[], c=[]),
        ):
            with self.assertRaises(ValueError):
                self.fuzzy_logic_controller.linguistic_variables()

    def test_compile_fullgraph_matches_eager(self) -> None:
        """
        Regression guard: torch.compile(fullgraph=True) must trace the TSK FLC's
        forward pass without any graph break (e.g. the membership cache's
        @torch.compiler.disable'd lookup, or FuzzySetGroup's former custom
        __getattribute__), and its output must match eager execution exactly.

        Returns:
            None
        """
        assert_compile_fullgraph_matches_eager(
            self.fuzzy_logic_controller, self.input_data
        )

    def test_compile_fullgraph_matches_eager_with_gradient_checkpointing(
        self,
    ) -> None:
        """
        Regression guard: gradient_checkpointing=True used to break
        torch.compile(fullgraph=True), since NAryRelation.apply_mask() stashed its
        result on self.applied_mask as a side effect, and Dynamo forbids in-place
        module-attribute mutation inside torch.utils.checkpoint's traced subgraph
        (the checkpointed function here is self.engine, an NAryRelation). Fixed by
        threading the applied mask through as a return value instead - see
        NAryRelation._apply_mask_with_mask.

        Returns:
            None
        """
        flc = FLC(
            source=self.fuzzy_logic_controller.source,
            inference=ZeroOrder,
            device=AVAILABLE_DEVICE,
            execution=ExecutionOptions(gradient_checkpointing=True),
        )
        flc.train()
        assert_compile_fullgraph_matches_eager(flc, self.input_data)

    def _missing_data_expected(self) -> dict:
        return {
            "expected_lower": [
                [1.0000e00, 1.6052e-09, 5.9053e-10, 3.6788e-01],
                [3.4559e-01, 1.4931e-10, 9.0561e-11, 5.6978e-01],
                [6.2376e-36, 5.9462e-03, 1.3268e-03, 2.0961e-01],
                [0.0000e00, 5.6095e-01, 1.6071e-01, 2.8206e-01],
                [1.3673e-11, 2.5467e-05, 1.2030e-05, 4.6504e-01],
                [0.0000e00, 0.0000e00, 0.0000e00, 0.0000e00],
            ],
            "expected_temp_upper": [
                [
                    [1.0000e00, 1.6052e-09, 1.6052e-09, 1.0000e00],
                    [1.0000e00, 1.0000e00, 3.6788e-01, 3.6788e-01],
                ],
                [
                    [3.6788e-01, 1.5894e-10, 1.5894e-10, 1.0000e00],
                    [9.3941e-01, 9.3941e-01, 5.6978e-01, 5.6978e-01],
                ],
                [
                    [6.6399e-36, 6.3297e-03, 6.3297e-03, 1.0000e00],
                    [9.3941e-01, 9.3941e-01, 2.0961e-01, 2.0961e-01],
                ],
                [
                    [0.0000e00, 5.6978e-01, 5.6978e-01, 1.0000e00],
                    [9.8450e-01, 9.8450e-01, 2.8206e-01, 2.8206e-01],
                ],
                [
                    [1.3888e-11, 2.5868e-05, 2.5868e-05, 1.0000e00],
                    [9.8450e-01, 9.8450e-01, 4.6504e-01, 4.6504e-01],
                ],
                [
                    [1.0000e00, 1.0000e00, 1.0000e00, 1.0000e00],
                    [9.8450e-01, 9.8450e-01, 4.6504e-01, 4.6504e-01],
                ],
            ],
            "expected_upper": [
                [1.0000e00, 1.6052e-09, 5.9053e-10, 3.6788e-01],
                [3.4559e-01, 1.4931e-10, 9.0561e-11, 5.6978e-01],
                [6.2376e-36, 5.9462e-03, 1.3268e-03, 2.0961e-01],
                [0.0000e00, 5.6095e-01, 1.6071e-01, 2.8206e-01],
                [1.3673e-11, 2.5467e-05, 1.2030e-05, 4.6504e-01],
                [9.8450e-01, 9.8450e-01, 4.6504e-01, 4.6504e-01],
            ],
        }

    def test_create_tsk(self) -> Tuple[FLC, torch.Tensor, List[Rule]]:
        """
        Test the creation of a TSK model.

        Returns:
            The FLC, input data, and rules.
        """
        input_data = torch.tensor(
            [[1.2, 0.2], [1.1, 0.3], [2.1, 0.1], [2.7, 0.15], [1.7, 0.25]],
            device=AVAILABLE_DEVICE,
        )
        antecedents = [
            Gaussian(
                centers=np.array([1.2, 3.0, 5.0, 7.0]),
                widths=np.array([0.1, 0.4, 0.6, 0.8]),
                device=AVAILABLE_DEVICE,
            ),
            Gaussian(
                centers=np.array([0.2, 0.6, 0.9, 1.2]),
                widths=np.array([0.4, 0.4, 0.5, 0.45]),
                device=AVAILABLE_DEVICE,
            ),
        ]
        # check that antecedents were correctly created
        self.assertTrue(
            torch.equal(
                antecedents[0].get_centers(),
                torch.tensor([[1.2, 3.0, 5.0, 7.0]], device=AVAILABLE_DEVICE),
            )
        )
        self.assertTrue(
            torch.equal(
                antecedents[0].get_widths(),
                torch.tensor([[0.1, 0.4, 0.6, 0.8]], device=AVAILABLE_DEVICE),
            )
        )
        self.assertTrue(
            torch.equal(
                antecedents[1].get_centers(),
                torch.tensor([[0.2, 0.6, 0.9, 1.2]], device=AVAILABLE_DEVICE),
            )
        )
        self.assertTrue(
            torch.equal(
                antecedents[1].get_widths(),
                torch.tensor([[0.4, 0.4, 0.5, 0.45]], device=AVAILABLE_DEVICE),
            )
        )
        rules = (
            [  # could be a set of rules, but a list is used here for reproducibility
                Rule(
                    premise=Product((0, 0), (1, 0), device=AVAILABLE_DEVICE),
                    consequence=NAryRelation((0, 0), device=AVAILABLE_DEVICE),
                ),
                Rule(
                    premise=Product((0, 1), (1, 0), device=AVAILABLE_DEVICE),
                    consequence=NAryRelation((0, 1), device=AVAILABLE_DEVICE),
                ),
                Rule(
                    premise=Product((0, 1), (1, 1), device=AVAILABLE_DEVICE),
                    consequence=NAryRelation((0, 2), device=AVAILABLE_DEVICE),
                ),
                Rule(
                    premise=Product((1, 1), device=AVAILABLE_DEVICE),
                    consequence=NAryRelation((0, 3), device=AVAILABLE_DEVICE),
                ),
            ]
        )
        # check that rules were correctly created
        knowledge_base = KnowledgeBase.create(
            linguistic_variables=LinguisticVariables(inputs=antecedents, targets=[]),
            rules=rules,
        )
        rule_vertex = knowledge_base.graph.vs.find(item_eq=rules[0])
        self.assertEqual(
            rule_vertex["item"], rules[0]
        )  # it is the correct relation we wanted
        # it has 'item' attribute
        self.assertIn("item", rule_vertex.attributes())
        rule_vertices = knowledge_base.select_by_tags("rule")
        self.assertEqual(
            len(rule_vertices), len(rules)
        )  # the number of rule vertices should equal len(rules)
        # there should be 2 rules that use (1, 1);
        # the last rule has been simplified (redundant mention of condition)
        self.assertIn(rules[2].premise, knowledge_base[(1, 1)].keys())
        self.assertEqual(
            set(knowledge_base[(1, 1)].values()),
            {
                frozenset({(0, 1), (1, 1)}),
                frozenset({(1, 1)}),
            },
        )
        # the recovered rules should be in the same order as the rules
        for expected_rule, actual_rule in zip(rules, knowledge_base.rules):
            self.assertEqual(expected_rule, actual_rule)

        # check a zero-order TSK cannot be created with an incorrect number of consequences
        # self.assertRaises(
        #     ValueError,
        #     ZeroOrderTSK,
        #     specifications=Specifications(
        #         type="tsk",
        #         t_norm="algebraic_product",
        #     ),
        #     knowledge_base=knowledge_base,
        #     consequences=torch.zeros(len(rules) - 1),
        # )

        # check that the zero-order TSK neuro-fuzzy network was correctly
        # created
        flc = FLC(
            source=knowledge_base,
            inference=ZeroOrder,
            device=AVAILABLE_DEVICE,
        )

        # check that the number of input/output features are correct
        assert flc.shape.n_inputs == 2
        assert flc.shape.n_outputs == 1

        actual_variables: List[FuzzySet] = flc.linguistic_variables().inputs
        for actual_variable, expected_variable in zip(actual_variables, antecedents):
            assert torch.allclose(
                actual_variable.get_centers(), expected_variable.get_centers()
            )
            assert torch.allclose(
                actual_variable.get_widths(), expected_variable.get_widths()
            )

        return flc, input_data, rules

    def test_execution_options_defaults_match_previous_flat_kwargs(self) -> None:
        """
        FuzzyLogicController used to take disabled_parameters/max_batch_chunk/
        gradient_checkpointing as three flat keyword arguments; they are now bundled
        into a single ExecutionOptions dataclass (execution=None defaults to
        ExecutionOptions()). Confirm the defaults it falls back to still match what
        the old flat defaults were.

        Returns:
            None
        """
        self.assertEqual([], self.fuzzy_logic_controller.disabled_parameters)
        self.assertIsNone(self.fuzzy_logic_controller.max_batch_chunk)
        self.assertFalse(self.fuzzy_logic_controller.gradient_checkpointing)

    def test_max_batch_chunk_produces_the_same_output_as_unchunked(self) -> None:
        """
        FuzzyLogicController.forward() splits the batch into chunks of at most
        max_batch_chunk observations (and concatenates the per-chunk outputs) purely
        as a memory/compute trade-off - it must not change the result. Toggles
        max_batch_chunk on the SAME instance (rather than building a second FLC from
        the same source) since ZeroOrder consequences are randomly reinitialized on
        every FLC construction when no explicit source is given - a second instance
        would produce a different result for reasons unrelated to chunking.

        Returns:
            None
        """
        unchunked_output = self.fuzzy_logic_controller(self.input_data)

        self.fuzzy_logic_controller.max_batch_chunk = 2
        try:
            chunked_output = self.fuzzy_logic_controller(self.input_data)
        finally:
            self.fuzzy_logic_controller.max_batch_chunk = None

        self.assertTrue(torch.allclose(unchunked_output, chunked_output))

    def test_execution_options_are_applied_at_construction(self) -> None:
        """
        Confirm ExecutionOptions passed to the FLC constructor actually reach the
        instance attributes forward() reads, not just that the (already covered
        separately) defaults work.

        Returns:
            None
        """
        execution = ExecutionOptions(
            disabled_parameters=["some_param"],
            max_batch_chunk=3,
            gradient_checkpointing=True,
        )
        flc = FLC(
            source=self.fuzzy_logic_controller.source,
            inference=ZeroOrder,
            device=AVAILABLE_DEVICE,
            execution=execution,
        )
        self.assertEqual(["some_param"], flc.disabled_parameters)
        self.assertEqual(3, flc.max_batch_chunk)
        self.assertTrue(flc.gradient_checkpointing)

    def test_tsk_consequences_are_built_on_the_requested_device(self) -> None:
        """
        Regression test: TSK.__init__ builds its randomly-initialized consequences with
        torch.randn(..., dtype=torch.float32) when source=None (the common "random
        init" case), without a device= argument, so self.weights/self.bias were always
        created on the CPU regardless of the device the FLC was requested on. On a
        machine where AVAILABLE_DEVICE is actually a CUDA device, this made the very
        first forward pass crash with "Expected all tensors to be on the same device"
        as soon as observations (on the requested device) were multiplied against
        self.weights (stuck on the CPU).

        Returns:
            None
        """
        antecedents = [
            Gaussian(
                centers=np.array([1.2, 3.0, 5.0]),
                widths=np.array([0.1, 0.4, 0.6]),
                device=AVAILABLE_DEVICE,
            ),
        ]
        rules = [
            Rule(
                premise=Product((0, 0), device=AVAILABLE_DEVICE),
                consequence=NAryRelation((0, 0), device=AVAILABLE_DEVICE),
            ),
        ]
        knowledge_base = KnowledgeBase.create(
            linguistic_variables=LinguisticVariables(inputs=antecedents, targets=[]),
            rules=rules,
        )
        flc = FLC(source=knowledge_base, inference=TSK, device=AVAILABLE_DEVICE)

        self.assertEqual(flc.defuzzification.weights.device.type, AVAILABLE_DEVICE.type)
        self.assertEqual(flc.defuzzification.bias.device.type, AVAILABLE_DEVICE.type)

        # the actual symptom: this used to raise a device-mismatch RuntimeError
        output = flc(torch.tensor([[2.0]], device=AVAILABLE_DEVICE))
        self.assertEqual(output.device.type, AVAILABLE_DEVICE.type)

    def test_tsk_does_not_produce_nan_when_product_rule_strengths_underflow(
        self,
    ) -> None:
        """
        Regression test: a Product t-norm's rule strengths are the product of one
        membership degree per input variable, each typically < 1.0 - multiplying
        enough of them together (empirically, as few as ~32-64 input variables)
        underflows to exactly 0.0 in float32. TSK.forward() divided by
        sum(rule_activations.degrees, 1) with no epsilon offset, unlike the sibling
        Defuzzification.forward() (the "standard"/Mamdani path), which already guards
        the analogous division. Whenever every rule's strength underflowed for a given
        batch element, that division became 0.0 / 0.0 = NaN, silently poisoning that
        element's entire output (and gradient) rather than producing a well-defined
        (if degenerate) result.

        Returns:
            None
        """
        n_inputs = 64  # empirically underflows every rule's strength to exactly 0.0
        antecedents = [
            Gaussian(
                centers=np.array([0.0, 1.0]),
                widths=np.array([0.3, 0.3]),
                device=AVAILABLE_DEVICE,
            )
            for _ in range(n_inputs)
        ]
        rules = [
            Rule(
                premise=Product(
                    *[(var_idx, 0) for var_idx in range(n_inputs)],
                    device=AVAILABLE_DEVICE,
                ),
                consequence=NAryRelation((0, 0), device=AVAILABLE_DEVICE),
            ),
        ]
        knowledge_base = KnowledgeBase.create(
            linguistic_variables=LinguisticVariables(inputs=antecedents, targets=[]),
            rules=rules,
        )
        flc = FLC(source=knowledge_base, inference=TSK, device=AVAILABLE_DEVICE)

        observations = torch.rand(8, n_inputs, device=AVAILABLE_DEVICE)
        with torch.no_grad():
            rule_strengths = flc.engine(flc.input_granulation(observations))
        # confirm the premise (underflow) actually occurs, so this test would have
        # caught the regression - not vacuously passing because the setup no longer
        # triggers it
        self.assertTrue(bool((rule_strengths.degrees == 0.0).all()))

        output = flc(observations)
        self.assertFalse(bool(output.isnan().any()))

        # and the gradient must be well-defined too, not just the forward value
        output.sum().backward()
        for param in flc.defuzzification.parameters():
            self.assertFalse(bool(param.grad.isnan().any()))

    def test_tsk_epsilon_offset_does_not_perturb_normal_scale_output(self) -> None:
        """
        The epsilon offset added to guard against the underflow case above must not
        meaningfully change TSK's output when rule strengths are a normal, non-tiny
        scale (the common case, e.g. a handful of input variables) - 1e-32 is many
        orders of magnitude below any realistic sum of rule strengths. Verified by
        directly replicating the old (unguarded) formula from the real intermediate
        rule strengths and confirming it agrees with the current, guarded output.

        Returns:
            None
        """
        antecedents = [
            Gaussian(
                centers=np.array([1.2, 3.0, 5.0, 7.0]),
                widths=np.array([0.1, 0.4, 0.6, 0.8]),
                device=AVAILABLE_DEVICE,
            ),
            Gaussian(
                centers=np.array([0.2, 0.6, 0.9, 1.2]),
                widths=np.array([0.4, 0.4, 0.5, 0.45]),
                device=AVAILABLE_DEVICE,
            ),
        ]
        rules = [
            Rule(
                premise=Product((0, 0), (1, 0), device=AVAILABLE_DEVICE),
                consequence=NAryRelation((0, 0), device=AVAILABLE_DEVICE),
            ),
            Rule(
                premise=Product((0, 1), (1, 1), device=AVAILABLE_DEVICE),
                consequence=NAryRelation((0, 1), device=AVAILABLE_DEVICE),
            ),
        ]
        knowledge_base = KnowledgeBase.create(
            linguistic_variables=LinguisticVariables(inputs=antecedents, targets=[]),
            rules=rules,
        )
        flc = FLC(source=knowledge_base, inference=TSK, device=AVAILABLE_DEVICE)
        input_data = torch.tensor(
            [[1.2, 0.2], [1.1, 0.3], [2.1, 0.1], [2.7, 0.15], [1.7, 0.25]],
            device=AVAILABLE_DEVICE,
        )

        with torch.no_grad():
            output = flc(input_data)

            rule_strengths = flc.engine(flc.input_granulation(input_data))
            # confirm this is actually the normal (non-underflowed) case, so this
            # test is meaningfully exercising "epsilon negligible at normal
            # scale"
            self.assertFalse(bool((rule_strengths.degrees.sum(dim=1) == 0.0).any()))

            tsk = flc.defuzzification
            rule_output = (input_data @ tsk.weights).view(
                input_data.shape[0], tsk.r, tsk.o
            )
            rule_output = rule_output + tsk.bias
            rule_output = rule_output.transpose(1, 2)
            fir_str_bar_unguarded = rule_strengths.degrees / torch.sum(
                rule_strengths.degrees, 1
            ).unsqueeze(1)
            expected = torch.einsum("NRC,NR->NC", rule_output, fir_str_bar_unguarded)
        self.assertTrue(torch.allclose(output, expected, atol=1e-6))
