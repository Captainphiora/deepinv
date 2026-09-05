import sys
import unittest
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import deepinv as dinv  # noqa: E402
from prepare_inputs import (  # noqa: E402
    official_gaussian_kernel,
    official_motion_kernel,
    official_random_mask,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_deepinv import official_diffpir_deblur_prox  # noqa: E402


class ZeroEpsilon(torch.nn.Module):
    def predict(self, x, condition, **kwargs):
        return torch.zeros_like(x), None


class DiffPIRAlignmentTest(unittest.TestCase):
    def test_sr_prox_solves_normal_equation(self):
        # Non-symmetric odd kernel detects convolution direction and sampling phase.
        generator = torch.Generator().manual_seed(4)
        kernel = torch.rand(1, 1, 5, 5, generator=generator)
        kernel /= kernel.sum()
        physics = dinv.physics.Downsampling(
            img_size=(1, 16, 16), factor=4, filter=kernel
        )
        z = torch.rand(1, 1, 16, 16, generator=generator)
        y = torch.rand(1, 1, 4, 4, generator=generator)
        rho = torch.tensor(0.3)
        result = official_diffpir_deblur_prox(z, y, kernel, gamma=1 / rho, factor=4)
        residual = physics.A_adjoint(physics.A(result) - y) + rho * (result - z)
        self.assertLess(residual.abs().max().item(), 1e-6)
        self.assertTrue(
            torch.allclose(
                (physics.A(z) * y).sum(), (z * physics.A_adjoint(y)).sum(), atol=1e-6
            )
        )

    def test_motion_kernel_uses_official_second_construction(self):
        class FakeKernel:
            def __init__(self, size, intensity):
                self.kernelMatrix = np.full(size, np.random.rand(), dtype=np.float32)

        np.random.seed(30)
        np.random.rand()
        expected = np.random.rand()
        kernel = official_motion_kernel(FakeKernel, 3, 0.5, case_index=3)
        self.assertEqual(kernel.shape, (1, 1, 3, 3))
        self.assertTrue(torch.equal(kernel, torch.full_like(kernel, expected)))

    def test_deblur_prox_matches_official_fft_closed_form(self):
        torch.manual_seed(0)
        kernel = official_gaussian_kernel(5, 1.0)
        x = torch.rand(1, 1, 16, 16)
        y = torch.rand_like(x)
        rho = torch.tensor(0.3)

        physics = dinv.physics.BlurFFT(img_size=(1, 16, 16), filter=kernel)
        actual = dinv.optim.data_fidelity.L2().prox(x, y, physics, gamma=1 / rho)

        psf = torch.zeros_like(x)
        psf[..., :5, :5] = kernel
        psf = torch.roll(psf, shifts=(-2, -2), dims=(-2, -1))
        transfer = torch.fft.fftn(psf, dim=(-2, -1))
        expected = torch.fft.ifftn(
            (
                transfer.conj() * torch.fft.fftn(y, dim=(-2, -1))
                + rho * torch.fft.fftn(x, dim=(-2, -1))
            )
            / (transfer.abs().square() + rho),
            dim=(-2, -1),
        ).real
        self.assertTrue(torch.allclose(actual, expected, atol=2e-6, rtol=2e-6))
        official_order = official_diffpir_deblur_prox(x, y, kernel, gamma=1 / rho)
        self.assertTrue(torch.allclose(official_order, expected, atol=2e-6, rtol=2e-6))

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
        x0 = (torch.rsqrt(model_alpha[-1]).to(initial.dtype) * initial).clamp(-1, 1)
        rho = algorithm.rhos[-1]
        x0_01 = x0 / 2 + 0.5
        prox = (mask * y + rho * x0_01) / (mask + rho)
        expected_internal = algorithm.sqrt_alphas_cumprod[0] * (2 * prox - 1)
        self.assertTrue(torch.equal(algorithm.seq.cpu(), torch.tensor([0, 999])))
        self.assertTrue(torch.allclose(output, expected_internal / 2 + 0.5))
        self.assertEqual(trajectory.shape[0], 3)


if __name__ == "__main__":
    unittest.main()
