"""
Test the TNormPipeline class, which allows customization of t-norm operations.
"""

import shutil
import unittest
from pathlib import Path

import torch

from fuzzy.relations.custom_t_norm import TNormPipeline
from fuzzy.utils.options.impl.impl_enums import RuleElevationEnum
from fuzzy.utils.options.impl.impl_options import InferenceConfig, RuleConfig

AVAILABLE_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TestTNormPipeline(unittest.TestCase):
    """
    Test the TNormPipeline class' save and load behavior.
    """

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
