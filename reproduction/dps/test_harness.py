import tempfile
import unittest
from pathlib import Path

import torch

from _common import (
    load_record,
    load_tensors,
    read_json,
    requires_transition_noise,
    save_tensors,
    tensor_dict_sha256,
    update_run_manifest,
)
from compare import (
    alignment_environment,
    differences,
    image_metrics,
    validate_alignment_environments,
)


class HarnessTest(unittest.TestCase):
    def test_transition_noise_requirement(self):
        self.assertTrue(
            requires_transition_noise({"sampler": {"name": "ddpm", "eta": 0}})
        )
        self.assertFalse(
            requires_transition_noise({"sampler": {"name": "ddim", "eta": 0}})
        )
        self.assertTrue(
            requires_transition_noise({"sampler": {"name": "ddim", "eta": 0.5}})
        )

    def test_tensor_artifact_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "artifact.pt"
            artifact = {"value": torch.arange(6, dtype=torch.float32).reshape(2, 3)}
            record = save_tensors(path, artifact)

            loaded = load_tensors(path, required=("value",))
            self.assertTrue(torch.equal(loaded["value"], artifact["value"]))
            self.assertEqual(record["tensor_sha256"], tensor_dict_sha256(loaded))
            with self.assertRaises(FileExistsError):
                save_tensors(path, artifact)
            self.assertTrue(
                torch.equal(load_record(directory, record)["value"], artifact["value"])
            )
            with path.open("ab") as handle:
                handle.write(b"corrupt")
            with self.assertRaises(ValueError):
                load_record(directory, record)

    def test_difference_and_image_metrics(self):
        expected = torch.zeros(1, 3, 16, 16)
        actual = torch.full_like(expected, 0.1)

        delta = differences(actual, expected)
        self.assertAlmostEqual(delta["mae"], 0.1, places=6)
        self.assertAlmostEqual(delta["rmse"], 0.1, places=6)
        self.assertAlmostEqual(delta["max_abs"], 0.1, places=6)
        metrics = image_metrics(expected, expected)
        self.assertAlmostEqual(metrics["ssim"], 1.0)
        self.assertIs(type(metrics["psnr_db"]), float)
        self.assertIs(type(metrics["ssim"]), float)

    def test_metric_crop_excludes_only_psnr_ssim_border(self):
        target = torch.zeros(1, 3, 24, 24)
        result = torch.full_like(target, 0.1)
        result[..., :4, :] = 1
        cropped = image_metrics(result, target, crop_border=4)
        expected = image_metrics(result[..., 4:-4, 4:-4], target[..., 4:-4, 4:-4])
        self.assertEqual(cropped, expected)
        self.assertGreater(cropped["psnr_db"], image_metrics(result, target)["psnr_db"])
        with self.assertRaises(ValueError):
            image_metrics(result, target, crop_border=12)

    def test_run_manifest_pins_fixture_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            arguments = {
                "setting_id": "setting",
                "setting_sha256": "setting-sha",
                "fixture_id": "fixture",
                "run_id": "run",
            }
            update_run_manifest(
                path,
                fixture_manifest_sha256="fixture-sha",
                implementation="reference",
                record={},
                **arguments,
            )
            update_run_manifest(
                path,
                fixture_manifest_sha256="fixture-sha",
                implementation="deepinv",
                record={},
                **arguments,
            )
            self.assertEqual(
                set(read_json(path)["implementations"]), {"reference", "deepinv"}
            )
            with self.assertRaises(ValueError):
                update_run_manifest(
                    path,
                    fixture_manifest_sha256="changed",
                    implementation="other",
                    record={},
                    **arguments,
                )

    def test_alignment_environment_ignores_gpu_location(self):
        first = {
            "torch": "test",
            "python_prefix": "/reference/.venv",
            "device": "cuda:0",
            "deterministic_algorithms": True,
            "cudnn_benchmark": False,
            "cuda_matmul_allow_tf32": False,
            "cudnn_allow_tf32": False,
            "gpu": {
                "name": "same-model",
                "compute_capability": [12, 0],
                "uuid": "gpu-0",
            },
        }
        second = {
            **first,
            "torch": "different-version",
            "python_prefix": "/deepinv/.venv",
            "device": "cuda:3",
            "gpu": {**first["gpu"], "uuid": "gpu-3"},
        }
        self.assertEqual(alignment_environment(first), alignment_environment(second))
        validate_alignment_environments(
            {"environment": first},
            {"environment": second},
            separate_uv_projects=True,
        )

        second["python_prefix"] = first["python_prefix"]
        with self.assertRaisesRegex(ValueError, "distinct recorded"):
            validate_alignment_environments(
                {"environment": first},
                {"environment": second},
                separate_uv_projects=True,
            )

        second["gpu"]["name"] = "different-model"
        self.assertNotEqual(alignment_environment(first), alignment_environment(second))


if __name__ == "__main__":
    unittest.main()
