import sys
import unittest
from pathlib import Path

import torch

DPS_DIR = Path(__file__).resolve().parents[1] / "dps"
if str(DPS_DIR) not in sys.path:
    sys.path.insert(0, str(DPS_DIR))

from _common import REPO_ROOT, fixed_randn_like, load_setting, run_dir  # noqa: E402


class DSGHarnessTest(unittest.TestCase):
    def test_setting_discovery_and_run_directory(self):
        _, setting = load_setting("ffhq256_inpainting_ddim100_eta1_dsg_v1")

        self.assertEqual(setting["algorithm"]["name"], "dsg")
        self.assertEqual(
            run_dir(Path("artifacts"), setting["id"], "trial", "dsg"),
            Path("artifacts/runs/dsg") / setting["id"] / "trial",
        )
        self.assertEqual(REPO_ROOT, Path(__file__).resolve().parents[2])

    def test_fixed_noise_is_consumed_once(self):
        noise = torch.tensor([1.0])
        draw, count = fixed_randn_like(noise)

        self.assertTrue(torch.equal(draw(torch.zeros_like(noise)), noise))
        self.assertEqual(count(), 1)
        with self.assertRaises(RuntimeError):
            draw(torch.zeros_like(noise))


if __name__ == "__main__":
    unittest.main()
