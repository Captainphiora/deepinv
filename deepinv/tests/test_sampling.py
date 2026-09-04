import gc
import pytest
import torch.nn
import numpy as np

import deepinv as dinv
from deepinv.optim.data_fidelity import L2
from deepinv.sampling import (
    ULA,
    SKRock,
    DiffPIR,
    DPS,
    DiscreteDPS,
    sampling_builder,
    DDRM,
    VarianceExplodingDiffusion,
    VariancePreservingDiffusion,
    EDMDiffusionSDE,
    FlowMatching,
    PosteriorDiffusion,
    DPSDataFidelity,
    EulerSolver,
    HeunSolver,
)
from deepinv.models import NCSNpp, ADMUNet, DRUNet

SAMPLING_ALGOS = ["DDRM", "ULA", "SKRock"]


class _DiscretePrediction(torch.nn.Module):
    def __init__(self, epsilon=0.0, variance=0.0):
        super().__init__()
        self.epsilon = epsilon
        self.variance = variance
        self.last_condition = None

    def predict(self, x, condition, **kwargs):
        assert kwargs == {
            "condition_type": "timestep",
            "output_type": "epsilon",
            "input_in_minus_one_one": True,
        }
        self.last_condition = condition.detach().clone()
        return x * 0 + self.epsilon, x * 0 + self.variance


class _IdentityPhysics:
    @staticmethod
    def A(x):
        return x

    @staticmethod
    def A_dagger(y):
        return y


def choose_algo(algo, likelihood, thresh_conv, sigma, sigma_prior):
    if algo == "ULA":
        out = ULA(
            GaussianScore(sigma_prior),
            likelihood,
            max_iter=500,
            thinning=1,
            step_size=0.01 / (1 / sigma**2 + 1 / sigma_prior**2),
            clip=(-100, 100),
            thresh_conv=thresh_conv,
            sigma=1,
            verbose=True,
        )
    elif algo == "SKRock":
        out = SKRock(
            GaussianScore(sigma_prior),
            likelihood,
            max_iter=500,
            inner_iter=5,
            step_size=1 / (1 / sigma**2 + 1 / sigma_prior**2),
            clip=(-100, 100),
            thresh_conv=thresh_conv,
            sigma=1,
            verbose=True,
        )
    elif algo == "DDRM":
        diff = dinv.sampling.DDRM(
            denoiser=GaussianDenoiser(sigma_prior),
            eta=1,
            sigmas=np.linspace(1, 0, 100),
        )
        out = dinv.sampling.DiffusionSampler(diff, clip=(-100, 100), max_iter=500)
    else:
        raise Exception("The sampling algorithm doesnt exist")

    return out


class GaussianScore(torch.nn.Module):
    def __init__(self, sigma_prior):
        super().__init__()
        self.sigma_prior2 = sigma_prior**2

    def grad(self, x, sigma):
        return x / self.sigma_prior2


class GaussianDenoiser(torch.nn.Module):
    def __init__(self, sigma_prior):
        super().__init__()
        self.sigma_prior2 = sigma_prior**2

    def forward(self, x, sigma):
        return x / (1 + sigma**2 / self.sigma_prior2)


@pytest.mark.parametrize("algo", SAMPLING_ALGOS)
def test_sampling_algo(algo, imsize, device):
    test_sample = torch.ones((1, *imsize))

    sigma = 1
    sigma_prior = 1
    physics = dinv.physics.Denoising()
    physics.noise_model = dinv.physics.GaussianNoise(sigma)
    y = physics(test_sample)

    convergence_crit = 0.1  # for fast tests
    likelihood = L2(sigma=sigma)
    f = choose_algo(
        algo,
        likelihood,
        thresh_conv=convergence_crit,
        sigma=sigma,
        sigma_prior=sigma_prior,
    )

    xmean, xvar = f(y, physics, seed=0)

    tol = 5  # can be lowered?
    sigma2 = sigma**2
    sigma_prior2 = sigma_prior**2

    # the posterior of a gaussian likelihood with a gaussian prior is gaussian
    post_var = (sigma2 * sigma_prior2) / (sigma2 + sigma_prior2)
    post_mean = y / (1 + sigma2 / sigma_prior2)

    mean_ok = (
        torch.sum((xmean - post_mean).abs() / post_mean < tol)
        > np.prod(xmean.shape) / 2
    )

    var_ok = (
        torch.sum((xvar - post_var).abs() / post_var < tol) > np.prod(xvar.shape) / 2
    )

    assert f.mean_has_converged and f.var_has_converged and mean_ok and var_ok


@pytest.mark.parametrize("name_algo", ["DiffPIR", "DPS"])
def test_algo(name_algo, device):
    test_sample = torch.ones((1, 3, 64, 64), device=device)

    sigma = 1
    # choose physics that changes the image size
    physics = dinv.physics.Blur(
        dinv.physics.functional.gaussian_blur(sigma=(3, 3)), device=device
    )
    physics.noise_model = dinv.physics.GaussianNoise(sigma)
    y = physics(test_sample)

    likelihood = L2(sigma=sigma)

    if name_algo == "DiffPIR":
        f = DiffPIR(
            dinv.models.DiffUNet().to(device),
            likelihood,
            max_iter=5,
            verbose=False,
            device=device,
        )
    elif name_algo == "DPS":
        f = DPS(
            dinv.models.DiffUNet().to(device),
            num_steps=5,
            verbose=False,
            device=device,
        )
    else:
        raise Exception("The sampling algorithm doesn't exist")

    x = f(y, physics)

    assert x.shape == test_sample.shape


@pytest.mark.parametrize("name_algo", ["DiffPIR", "DPS", "DDRM"])
def test_algo_inpaint(name_algo, device):
    x = torch.ones((1, 3, 32, 32)).to(device)
    x[:, 0, ...] = 0  # create a colored image

    torch.manual_seed(10)

    mask = torch.ones_like(x)
    mask[:, :, 10:20, 10:20] = 0

    physics = dinv.physics.Inpainting(mask=mask, img_size=x.shape[1:], device=device)

    y = physics(x)

    model = dinv.models.DRUNet(device=device)
    likelihood = L2()

    if name_algo == "DiffPIR":
        algorithm = DiffPIR(
            model, likelihood, max_iter=20, verbose=False, device=device, sigma=0.01
        )
    elif name_algo == "DPS":
        algorithm = DPS(
            model, num_steps=50, weight=2.0, alpha=0.5, verbose=False, device=device
        )
    elif name_algo == "DDRM":
        algorithm = DDRM(model)

    with torch.no_grad():
        out = algorithm(y, physics)

    assert out.shape == x.shape

    mean_crop = out[:, :, 10:20, 10:20].flatten().mean()

    mask = mask.bool()
    masked_out = out[mask]
    mean_outside_crop = masked_out.mean()

    masked_target = x[mask]
    mean_target_masked = masked_target.mean()
    mean_target_inmask = 2 / 3.0

    assert (mean_target_inmask - mean_crop).abs() < 0.2
    assert (mean_target_masked - mean_outside_crop).abs() < 0.02


def test_discrete_dps_timestep_respacing():
    betas = np.linspace(0.001, 0.01, 10, dtype=np.float64)
    model = _DiscretePrediction()
    dps = DiscreteDPS(model, betas=betas, timestep_respacing=4)

    assert dps.timestep_map.tolist() == [0, 3, 6, 9]
    expected = np.cumprod(1.0 - betas)[[0, 3, 6, 9]]
    np.testing.assert_allclose(dps.alphas_cumprod.numpy(), expected, rtol=1e-14)

    ddim = DiscreteDPS(
        model, sampler="ddim", betas=betas, timestep_respacing="ddim5"
    )
    assert ddim.timestep_map.tolist() == [0, 2, 4, 6, 8]

    model.variance_type = "learned"
    with pytest.raises(ValueError, match="learned-range"):
        DiscreteDPS(model, sampler="ddpm", betas=betas, timestep_respacing=4)


def test_discrete_dps_ddpm_learned_range_step():
    model = _DiscretePrediction(epsilon=0.0, variance=1.0)
    dps = DiscreteDPS(
        model,
        sampler="ddpm",
        betas=np.array([0.1, 0.2]),
        timestep_respacing=2,
        clip_denoised=False,
    )
    x = torch.full((1, 1, 1, 1), 0.25)
    noise = torch.full_like(x, 0.4)
    out = dps.p_sample(x, torch.tensor([1]), noise=noise)

    alpha_bar = 0.9 * 0.8
    pred_xstart = x / np.sqrt(alpha_bar)
    mean = (
        0.2 * np.sqrt(0.9) / (1.0 - alpha_bar) * pred_xstart
        + 0.1 * np.sqrt(0.8) / (1.0 - alpha_bar) * x
    )
    expected = mean + np.sqrt(0.2) * noise
    torch.testing.assert_close(out["sample"], expected)
    assert model.last_condition.tolist() == [1]

    at_zero_a = dps.p_sample(x, 0, noise=torch.ones_like(x))["sample"]
    at_zero_b = dps.p_sample(x, 0, noise=torch.zeros_like(x))["sample"]
    torch.testing.assert_close(at_zero_a, at_zero_b)


def test_discrete_dps_ddim_eta_step():
    eta = 0.5
    model = _DiscretePrediction(epsilon=0.1, variance=0.0)
    dps = DiscreteDPS(
        model,
        sampler="ddim",
        betas=np.array([0.1, 0.2]),
        timestep_respacing=2,
        eta=eta,
        clip_denoised=False,
    )
    x = torch.full((1, 1, 1, 1), 0.25)
    noise = torch.full_like(x, -0.3)
    out = dps.p_sample(x, 1, noise=noise)

    alpha_bar = 0.9 * 0.8
    alpha_bar_prev = 0.9
    pred_xstart = x / np.sqrt(alpha_bar) - np.sqrt(1 / alpha_bar - 1) * 0.1
    epsilon = (x / np.sqrt(alpha_bar) - pred_xstart) / np.sqrt(
        1 / alpha_bar - 1
    )
    sigma = eta * np.sqrt(
        (1 - alpha_bar_prev) / (1 - alpha_bar)
    ) * np.sqrt(1 - alpha_bar / alpha_bar_prev)
    expected = (
        np.sqrt(alpha_bar_prev) * pred_xstart
        + np.sqrt(1 - alpha_bar_prev - sigma**2) * epsilon
        + sigma * noise
    )
    torch.testing.assert_close(out["sample"], expected)


def test_discrete_dps_fixed_inputs_noise_and_gradient():
    betas = np.array([0.01, 0.02])
    model = _DiscretePrediction(epsilon=0.0, variance=0.0)
    dps = DiscreteDPS(
        model,
        sampler="ddpm",
        betas=betas,
        timestep_respacing=2,
        scale=0.1,
        clip_denoised=False,
        rng=torch.Generator(),
    )
    physics = _IdentityPhysics()
    y = torch.zeros(1, 1, 1, 2)
    x_init = torch.tensor([[[[0.2, -0.1]]]])
    tape = torch.tensor([[[[[0.3, -0.2]]]], [[[[0.9, 0.7]]]]])

    probe = x_init.clone().requires_grad_()
    first = dps.p_sample(probe, 1, noise=tape[0])
    distance = torch.linalg.vector_norm(y - physics.A(first["pred_xstart"]))
    gradient = torch.autograd.grad(distance, probe)[0]
    expected_first = (first["sample"] - 0.1 * gradient).detach()

    global_state = torch.random.get_rng_state()
    with torch.no_grad():
        out_a, trajectory = dps(
            y,
            physics,
            x_init=x_init,
            seed=1,
            transition_noise=tape,
            get_trajectory=True,
        )
    out_b = dps(
        y,
        physics,
        x_init=x_init,
        seed=999,
        transition_noise=tape,
    )
    torch.testing.assert_close(out_a, out_b, rtol=0, atol=0)
    torch.testing.assert_close(trajectory[1], expected_first)
    assert trajectory.shape == (3, *x_init.shape)
    assert torch.equal(torch.random.get_rng_state(), global_state)

    seeded_a = dps(y, physics, seed=7, transition_noise=tape)
    seeded_b = dps(y, physics, seed=7, transition_noise=tape)
    torch.testing.assert_close(seeded_a, seeded_b, rtol=0, atol=0)


# tests for sample_builder
BUILD_ALGOS = ["ULA", "SKRock"]


def choose_algo_build(algo, likelihood, thresh_conv, sigma, sigma_prior):
    prior = GaussianScore(sigma_prior)

    if algo == "ULA":
        params = {
            "step_size": 0.01 / (1 / sigma**2 + 1 / sigma_prior**2),
            "alpha": 1.0,
            "sigma": 1.0,
        }
    elif algo == "SKRock":
        params = {
            "step_size": 1 / (1 / sigma**2 + 1 / sigma_prior**2),
            "alpha": 1.0,
            "inner_iter": 5,
            "eta": 0.05,
            "sigma": 1.0,
        }
    else:
        raise Exception("The sampling algorithm doesn't exist")

    out = sampling_builder(
        iterator=algo,
        data_fidelity=likelihood,
        prior=prior,
        thresh_conv=thresh_conv,
        params_algo=params,
        max_iter=500,
        burnin_ratio=0.2,
        thinning=1,
        verbose=True,
        clip=(-100, 100),
    )

    return out


@pytest.mark.parametrize("algo", BUILD_ALGOS)
def test_build_algo(algo, imsize, device):
    # NOTE: redundancy here with the above test_sample_algo
    test_sample = torch.ones((1, *imsize))

    sigma = 1
    sigma_prior = 1
    physics = dinv.physics.Denoising()
    physics.noise_model = dinv.physics.GaussianNoise(sigma)
    y = physics(test_sample)

    convergence_crit = 0.1  # for fast tests
    likelihood = L2(sigma=sigma)
    f = choose_algo_build(
        algo,
        likelihood,
        thresh_conv=convergence_crit,
        sigma=sigma,
        sigma_prior=sigma_prior,
    )

    xmean, xvar = f.sample(y, physics, seed=0)

    tol = 5  # can be lowered?
    sigma2 = sigma**2
    sigma_prior2 = sigma_prior**2

    # the posterior of a gaussian likelihood with a gaussian prior is gaussian
    post_var = (sigma2 * sigma_prior2) / (sigma2 + sigma_prior2)
    post_mean = y / (1 + sigma2 / sigma_prior2)

    mean_ok = (
        torch.sum((xmean - post_mean).abs() / post_mean < tol)
        > np.prod(xmean.shape) / 2
    )

    var_ok = (
        torch.sum((xvar - post_var).abs() / post_var < tol) > np.prod(xvar.shape) / 2
    )

    assert f.mean_has_converged and f.var_has_converged and mean_ok and var_ok


@torch.no_grad()
@pytest.mark.parametrize(
    "sde_class",
    [
        FlowMatching,
        VarianceExplodingDiffusion,
        VariancePreservingDiffusion,
        EDMDiffusionSDE,
    ],
)
@pytest.mark.parametrize("solver_class", [EulerSolver, HeunSolver])
@pytest.mark.parametrize("denoiser_class", [NCSNpp, ADMUNet, DRUNet])
def test_sde(device, load_example_image, sde_class, solver_class, denoiser_class):
    try:
        if denoiser_class == ADMUNet:
            kwargs = dict(class_labels=torch.eye(1000, device=device)[0:1])
        else:
            kwargs = dict()
        denoiser = denoiser_class(pretrained="download").to(device)
        x = load_example_image(
            "celeba_example.jpg",
            img_size=64,
            resize_mode="resize",
        ).to(device)

        # Set up the SDEs
        num_steps = 2
        rng = torch.Generator(device)
        # Set up solvers
        timesteps = torch.linspace(0.99, 0.001, num_steps, device=device)
        solver = solver_class(
            timesteps=timesteps,
            rng=rng,
        )

        if sde_class == EDMDiffusionSDE:
            sigma_t = lambda t: 100 * t**2
            scale_t = lambda t: 1 / (1 + sigma_t(t) ** 2) ** 0.5
            sde = sde_class(
                sigma_t=sigma_t,
                scale_t=scale_t,
                denoiser=denoiser,
                solver=solver,
                device=device,
            )
        else:
            sde = sde_class(
                denoiser=denoiser,
                solver=solver,
                device=device,
            )
        # Test generation
        sample, _trajectory = sde.sample(
            (2, 3, 64, 64),
            seed=10,
            get_trajectory=True,
            **kwargs,
        )
        assert sample.shape == (2, 3, 64, 64)

        # Test posterior sampling
        posterior = PosteriorDiffusion(
            data_fidelity=DPSDataFidelity(denoiser=denoiser),
            sde=sde,
            denoiser=denoiser,
            solver=solver,
            dtype=torch.float64,
            device=device,
        )
        physics = dinv.physics.Inpainting(img_size=x.shape[1:], mask=0.5, device=device)
        y = physics(x)

        x_hat = posterior(
            y,
            physics,
            x_init=(2, 3, 64, 64),
            seed=111,
            **kwargs,
        )
        # Test output shape
        assert x_hat.shape == (2, 3, 64, 64)
    finally:
        # pytest seems to not clean objects properly, which can cause OOM errors.
        del denoiser, sde, posterior
        gc.collect()
        torch.cuda.empty_cache()


VE_VP_SDE_CLASSES = [VarianceExplodingDiffusion, VariancePreservingDiffusion]


@pytest.mark.parametrize("sde_class", VE_VP_SDE_CLASSES)
@pytest.mark.parametrize("t", [0.2, 0.5, 0.8, 0.95])
def test_sigma_scale_prime_matches_finite_difference(sde_class, t):
    """`sigma_prime_t`, `scale_prime_t` must match finite difference"""
    sde = sde_class(dtype=torch.float64)
    h = 1e-6
    # For sigma
    fd = (float(sde.sigma_t(t + h)) - float(sde.sigma_t(t - h))) / (2 * h)
    assert float(sde.sigma_prime_t(t)) == pytest.approx(fd, rel=1e-4)
    # For scale
    fd = (float(sde.scale_t(t + h)) - float(sde.scale_t(t - h))) / (2 * h)
    assert float(sde.scale_prime_t(t)) == pytest.approx(fd, rel=1e-4, abs=1e-9)


@pytest.mark.parametrize("sde_class", VE_VP_SDE_CLASSES)
def test_T_is_settable(sde_class):
    """The documented end time `T` must be reachable from the constructor."""
    assert sde_class(T=0.9).T == 0.9


@pytest.mark.parametrize("sde_class", VE_VP_SDE_CLASSES)
def test_sample_init_uses_given_time_step(sde_class, rng, device):
    """`sample_init` draws at the requested time, defaulting to `T`."""
    sde = sde_class(device=device)
    shape = (1, 3, 64, 64)
    for t in (sde.T, 0.5):
        init = sde.sample_init(shape, rng=rng, t=t)
        expected = float(sde.sigma_t(t) * sde.scale_t(t))
        assert float(init.std()) == pytest.approx(expected, rel=5e-2)

    # Default sample must be at time T
    rng.manual_seed(0)
    default = sde.sample_init(shape, rng=rng)
    rng.manual_seed(0)
    assert torch.allclose(default, sde.sample_init(shape, rng=rng, t=sde.T))


@torch.no_grad()
@pytest.mark.parametrize(
    "sde_class",
    [
        FlowMatching,
        VarianceExplodingDiffusion,
        VariancePreservingDiffusion,
        EDMDiffusionSDE,
    ],
)
def test_diffusion_reproducibility(load_example_image, device, rng, sde_class):
    timesteps = torch.linspace(0.99, 0.001, 2, device=device)
    denoiser = NCSNpp(pretrained="download").to(device)
    solver = EulerSolver(timesteps=timesteps, rng=rng)

    sigma_t = lambda t: 100 * t**2
    scale_t = lambda t: 1 / (1 + sigma_t(t) ** 2) ** 0.5
    kwargs = (
        {"sigma_t": sigma_t, "scale_t": scale_t} if sde_class == EDMDiffusionSDE else {}
    )
    sde = sde_class(
        denoiser=denoiser,
        solver=solver,
        device=device,
        **kwargs,
    )
    x = load_example_image(
        "celeba_example.jpg",
        img_size=64,
        resize_mode="resize",
    ).to(device)
    physics = dinv.physics.Inpainting(img_size=x.shape[1:], mask=0.5, device=device)
    y = physics(x)

    # Test posterior sampling
    posterior = PosteriorDiffusion(
        data_fidelity=DPSDataFidelity(denoiser=denoiser),
        sde=sde,
        denoiser=denoiser,
        solver=solver,
        dtype=torch.float64,
        device=device,
    )

    x_hat_1 = posterior(
        y,
        physics,
        x_init=(2, 3, 64, 64),
        seed=111,
    )
    # Test output shape
    assert x_hat_1.shape == (2, 3, 64, 64)
    # Test reproducibility
    x_hat_2 = posterior(
        y,
        physics,
        x_init=(2, 3, 64, 64),
        seed=111,
    )
    torch.testing.assert_close(x_hat_1, x_hat_2, rtol=1e-2, atol=1e-2)


@torch.no_grad()
def test_noisy_data_fidelity(device):
    from deepinv.sampling import DPSDataFidelity, NoisyDataFidelity
    import itertools

    all_data_fid_classes = [NoisyDataFidelity, DPSDataFidelity]
    all_clip = [None, (-100, 100)]
    denoiser = dinv.models.DRUNet(pretrained="download").to(device)
    x = torch.rand(2, 3, 64, 64, device=device)
    physics = dinv.physics.Blur(
        filter=dinv.physics.functional.gaussian_blur(sigma=(3, 3)), device=device
    )
    y = physics(x)
    sigma = 0.1
    for data_fid_class, clip in itertools.product(all_data_fid_classes, all_clip):
        data_fid = data_fid_class(
            denoiser=denoiser,
            clip=clip,
        )
        # Test forward pass
        assert data_fid(x, y, physics, sigma).shape == torch.Size([x.size(0)])
        # Test grad pass
        assert data_fid.grad(x, y, physics, sigma).shape == x.shape
        # Test preconditioning
        try:
            output = data_fid.precond(y, physics, sigma)
            assert output.shape == x.shape
        except NotImplementedError:
            pass
