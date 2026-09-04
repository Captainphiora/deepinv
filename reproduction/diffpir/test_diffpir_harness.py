import sys
import unittest
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import deepinv as dinv  # noqa: E402
from prepare_inputs import official_random_mask  # noqa: E402


class ZeroEpsilon(torch.nn.Module):
    def predict(self, x, condition, **kwargs):
        return torch.zeros_like(x), None


class DiffPIRAlignmentTest(unittest.TestCase):
    def test_official_mask_and_closed_form_step(self):
        np.random.seed(42)
        mask = official_random_mask(4, 0.5)
        self.assertEqual(mask[0, 0].sum().item(), 8)
        self.assertTrue(torch.equal(mask[:, :1], mask[:, 1:2]))

        y = torch.full((1, 1, 2, 2), 0.75)
        mask = torch.tensor([[[[1.0, 0.0], [1.0, 0.0]]]])
        physics = dinv.physics.Inpainting(img_size=(1, 2, 2), mask=mask)
        algorithm = dinv.sampling.DiffPIR(
            ZeroEpsilon(),
            dinv.optim.data_fidelity.L2(),
            sigma=0.001,
            max_iter=2,
            zeta=1,
        )
        initial = torch.full_like(y, 0.001)
        output, trajectory = algorithm(
            y,
            physics,
            initial_state=initial,
            transition_noise=torch.zeros(1, *initial.shape),
            get_trajectory=True,
        )

        model_alpha = torch.cumprod(
            1 - torch.linspace(0.0001, 0.02, 1000, dtype=torch.float64), dim=0
        )
        x0 = (
            torch.rsqrt(model_alpha[-1]).to(initial.dtype) * initial
        ).clamp(-1, 1)
        rho = algorithm.rhos[-1]
        x0_01 = x0 / 2 + 0.5
        prox = (mask * y + rho * x0_01) / (mask + rho)
        expected_internal = algorithm.sqrt_alphas_cumprod[0] * (2 * prox - 1)
        self.assertTrue(torch.equal(algorithm.seq.cpu(), torch.tensor([0, 999])))
        self.assertTrue(torch.allclose(output, expected_internal / 2 + 0.5))
        self.assertEqual(trajectory.shape[0], 3)


if __name__ == "__main__":
    unittest.main()
