"""
Test the TNormPipeline class, which allows customization of t-norm operations.
"""

import shutil
import unittest
from pathlib import Path

import torch

from fuzzy.relations.confidence import CertaintyFactors
from fuzzy.relations.custom_t_norm import TNormPipeline
from fuzzy.utils.options.impl.impl_enums import RuleElevationEnum, RuleWeightsEnum
from fuzzy.utils.options.impl.impl_options import InferenceConfig, RuleConfig
from tests import AVAILABLE_DEVICE


class TestTNormPipeline(unittest.TestCase):
    """
    Test the TNormPipeline class' save and load behavior.
    """

    def test_layer_norm_passed_via_kwargs_is_used_directly(self) -> None:
        """
        __init__'s "layer_norm" kwarg lets a caller supply an already-built
        torch.nn.LayerNorm directly, bypassing the configuration.rule.elevation-driven
        construction path. Verify the exact object passed in is stored, not a new one.

        Returns:
            None
        """
        n_relations = 4
        layer_norm = torch.nn.LayerNorm([n_relations], device=AVAILABLE_DEVICE)
        pipeline = TNormPipeline(
            configuration=InferenceConfig(),
            n_relations=n_relations,
            device=AVAILABLE_DEVICE,
            layer_norm=layer_norm,
        )
        self.assertIs(layer_norm, pipeline.layer_norm)

    def test_certainty_passed_via_kwargs_is_used_directly(self) -> None:
        """
        __init__'s "certainty" kwarg lets a caller supply an already-built
        CertaintyFactors directly, bypassing the configuration.rule.weights-driven
        construction path. Verify the exact object passed in is stored, not a new one.

        Returns:
            None
        """
        n_relations = 4
        certainty = CertaintyFactors.create_default(
            n_features=n_relations, device=AVAILABLE_DEVICE
        )
        pipeline = TNormPipeline(
            configuration=InferenceConfig(),
            n_relations=n_relations,
            device=AVAILABLE_DEVICE,
            certainty=certainty,
        )
        self.assertIs(certainty, pipeline.certainty)

    def test_certainty_factors_enabled_via_configuration(self) -> None:
        """
        When configuration.rule.weights requests CERTAINTY_FACTORS (and no "certainty"
        kwarg is supplied), __init__ must build a default CertaintyFactors itself.

        Returns:
            None
        """
        n_relations = 3
        configuration = InferenceConfig(
            rule=RuleConfig(weights=RuleWeightsEnum.CERTAINTY_FACTORS)
        )
        pipeline = TNormPipeline(
            configuration=configuration,
            n_relations=n_relations,
            device=AVAILABLE_DEVICE,
        )
        self.assertIsInstance(pipeline.certainty, CertaintyFactors)

    def test_forward_applies_layer_norm_and_certainty(self) -> None:
        """
        forward() must route x through layer_norm and certainty (when configured)
        before the final activation - neither branch had any test coverage before.

        Returns:
            None
        """
        n_relations = 3
        configuration = InferenceConfig(
            rule=RuleConfig(
                elevation=RuleElevationEnum.LAYER_NORMALIZATION,
                weights=RuleWeightsEnum.CERTAINTY_FACTORS,
            )
        )
        pipeline = TNormPipeline(
            configuration=configuration,
            n_relations=n_relations,
            device=AVAILABLE_DEVICE,
        )
        x = torch.rand(5, n_relations, device=AVAILABLE_DEVICE)
        result = pipeline(x)
        self.assertEqual(x.shape, result.shape)
        self.assertFalse(bool(result.isnan().any()))
        # PremiseActivation's softmax-family output sums to 1 along the last
        # dim
        self.assertTrue(
            torch.allclose(
                result.sum(
                    dim=-1),
                torch.ones(
                    5,
                    device=AVAILABLE_DEVICE)))

    def test_load_invalid_path_raises(self) -> None:
        """
        load() must reject a path that is not a directory (e.g. it was never saved,
        or points at a plain file) rather than silently doing the wrong thing.

        Returns:
            None
        """
        with self.assertRaises(ValueError):
            TNormPipeline.load(
                Path("this_path_does_not_exist"), device=AVAILABLE_DEVICE
            )

    def test_save_with_certainty_factors(self) -> None:
        """
        save() must also persist self.certainty (to a "certainty" subdirectory) when
        it is not None - only the certainty-less path had coverage before.

        Returns:
            None
        """
        n_relations = 3
        configuration = InferenceConfig(
            rule=RuleConfig(weights=RuleWeightsEnum.CERTAINTY_FACTORS)
        )
        pipeline = TNormPipeline(
            configuration=configuration,
            n_relations=n_relations,
            device=AVAILABLE_DEVICE,
        )
        path = Path("t_norm_pipeline_with_certainty")
        try:
            pipeline.save(path)
            self.assertTrue((path / "certainty").is_dir())
            self.assertTrue((path / "state_dict.pt").is_file())
        finally:
            shutil.rmtree(path)

    @unittest.skipUnless(
        torch.cuda.is_available(), "requires a second device (CUDA) to move to"
    )
    def test_load_respects_requested_device(self) -> None:
        """
        Regression guard: TNormPipeline.load used to call torch.load without
        map_location=device, unlike the equivalent (already-fixed) pattern in
        CertaintyFactors.load. Without it, torch.load deserializes tensors back onto
        whatever device they were *saved* from - here, a pipeline saved on CUDA and
        then loaded requesting the CPU must actually come back on the CPU, not
        silently stay on (or crash trying to restore) the original device.

        Returns:
            None
        """
        n_relations = 4
        configuration = InferenceConfig(
            rule=RuleConfig(elevation=RuleElevationEnum.LAYER_NORMALIZATION)
        )
        pipeline = TNormPipeline(
            configuration=configuration,
            n_relations=n_relations,
            device=torch.device("cuda"),
        )
        self.assertEqual("cuda", pipeline.layer_norm.weight.device.type)

        path = Path("t_norm_pipeline")
        pipeline.save(path)
        # TNormPipeline.load reads the configuration from path.parent /
        # "configuration.yaml" - save() does not write it, so it must be placed there
        # directly for this round trip
        configuration.save(path.parent / "configuration.yaml")

        try:
            loaded_pipeline = TNormPipeline.load(
                path, device=torch.device("cpu"))
            self.assertEqual(
                "cpu", loaded_pipeline.layer_norm.weight.device.type)
            self.assertEqual(
                "cpu", loaded_pipeline.layer_norm.bias.device.type)
            self.assertTrue(
                torch.allclose(
                    pipeline.layer_norm.weight.cpu(),
                    loaded_pipeline.layer_norm.weight,
                )
            )
        finally:
            shutil.rmtree(path)
            (path.parent / "configuration.yaml").unlink()


if __name__ == "__main__":
    unittest.main()
