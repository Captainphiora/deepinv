"""Run DeepInv DiffPIR on the same immutable official-style fixture."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

DPS_DIR = Path(__file__).resolve().parents[1] / "dps"
if str(DPS_DIR) not in sys.path:
    sys.path.insert(0, str(DPS_DIR))

from _common import (  # noqa: E402
    REPO_ROOT,
    artifact_root,
    command_line,
    configure_determinism,
    environment,
    file_sha256,
    fixture_dir,
    git_revision,
    load_record,
    load_setting,
    read_json,
    require_clean_repo,
    run_dir,
    save_tensors,
    select_cases,
    update_run_manifest,
    utc_now,
    validate_tensor_dict,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--setting", required=True)
    parser.add_argument("--fixture-id", default="ffhq256_inpainting_diffpir_quad20_v1")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--case", action="append")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--artifact-root")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def official_diffpir_deblur_prox(
    z: torch.Tensor,
    y: torch.Tensor,
    kernel: torch.Tensor,
    gamma: float | torch.Tensor,
    factor: int = 1,
) -> torch.Tensor:
    """Torch port of DiffPIR's FFT order for circular blur and blur/decimation."""
    if (
        not isinstance(factor, int)
        or factor < 1
        or z.shape[-2:] != tuple(size * factor for size in y.shape[-2:])
    ):
        raise ValueError("factor must be a positive integer matching z/y spatial sizes")
    rho = torch.as_tensor(1 / gamma, dtype=z.dtype, device=z.device)
    rho = rho[(...,) + (None,) * (z.ndim - rho.ndim)]
    psf = torch.zeros(kernel.shape[:-2] + z.shape[-2:], dtype=z.dtype, device=z.device)
    kernel = kernel.to(dtype=z.dtype, device=z.device)
    psf[..., : kernel.shape[-2], : kernel.shape[-1]].copy_(kernel)
    psf = torch.roll(
        psf,
        shifts=(-(kernel.shape[-2] // 2), -(kernel.shape[-1] // 2)),
        dims=(-2, -1),
    )
    transfer = torch.fft.fftn(psf, dim=(-2, -1))
    transfer_conj = transfer.conj()
    transfer_sq = transfer.abs().square()
    if factor > 1:
        # Official upsample allocates contiguous NCHW even for channels-last z/y.
        upsampled = torch.zeros(z.shape, dtype=z.dtype, device=z.device)
        upsampled[..., ::factor, ::factor].copy_(y)
    else:
        upsampled = y
    residual = transfer_conj * torch.fft.fftn(upsampled, dim=(-2, -1))
    residual = residual + torch.fft.fftn(rho * z, dim=(-2, -1))
    if factor == 1:
        inverse = transfer * residual / (transfer_sq + rho)
    else:

        def alias_mean(value):
            blocks = torch.stack(torch.chunk(value, factor, dim=2), dim=4)
            blocks = torch.cat(torch.chunk(blocks, factor, dim=3), dim=4)
            return blocks.mean(dim=-1)

        inverse = alias_mean(transfer * residual) / (alias_mean(transfer_sq) + rho)
        inverse = inverse.repeat(1, 1, factor, factor)
    solution = (residual - transfer_conj * inverse) / rho
    return torch.fft.ifftn(solution, dim=(-2, -1)).real


def build_algorithm(
    setting: dict, checkpoint: Path, schedule: dict, device: torch.device
):
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    import deepinv as dinv

    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    validate_tensor_dict(state)
    score_model = dinv.models.DiffUNet(pretrained=None)
    incompatible = score_model.load_state_dict(state, strict=False)
    expected_missing = {"sqrt_1m_alphas_cumprod", "sqrt_alphas_cumprod"}
    if (
        set(incompatible.missing_keys) != expected_missing
        or incompatible.unexpected_keys
    ):
        raise ValueError(
            f"checkpoint does not match DiffUNet: {incompatible.missing_keys}, "
            f"{incompatible.unexpected_keys}"
        )
    score_model = score_model.to(device).eval()

    sampler = setting["sampler"]
    model_alpha = schedule["model_alpha_cumprod"]
    wrapped = dinv.models.ScoreModelWrapper(
        score_model,
        prediction_type=setting["model"]["prediction_type"],
        model_input_type=setting["model"]["model_input_type"],
        variance_type=setting["model"]["variance_type"],
        model_kwargs={"type_t": "timestep"},
        sigma_t=torch.sqrt((1 - model_alpha) / model_alpha).float(),
        scale_t=torch.sqrt(model_alpha).float(),
        n_timesteps=sampler["train_steps"],
        device=device,
    )
    return dinv.sampling.DiffPIR(
        wrapped,
        dinv.optim.data_fidelity.L2(),
        sigma=setting["task"]["algorithm_sigma"],
        max_iter=sampler["sampling_steps"],
        zeta=sampler["zeta"],
        lambda_=setting["algorithm"]["lambda_"],
        guidance_scale=setting["algorithm"]["guidance_scale"],
        eta=sampler["eta"],
        betas=schedule["betas"],
        alphas_cumprod=schedule["alpha_cumprod"],
        verbose=True,
        device=device,
    )


def main() -> None:
    args = parse_args()
    setting_file, setting = load_setting(args.setting)
    if setting["algorithm"]["name"] != "diffpir":
        raise ValueError("DiffPIR runner requires algorithm.name='diffpir'")
    checkpoint = Path(args.checkpoint).resolve()
    checkpoint_hash = file_sha256(checkpoint)
    if checkpoint_hash != setting["model"]["checkpoint_sha256"]:
        raise ValueError("checkpoint SHA256 does not match the setting")

    root = artifact_root(args.artifact_root)
    fixture = fixture_dir(root, args.fixture_id)
    fixture_manifest_path = fixture / "manifest.json"
    fixture_manifest = read_json(fixture_manifest_path)
    if fixture_manifest.get("source_setting_sha256") != file_sha256(setting_file):
        raise ValueError("fixture was generated from a different setting JSON")
    cases = select_cases(fixture_manifest, args.case)
    destination = run_dir(root, setting["id"], args.run_id, "diffpir")
    revision = git_revision(REPO_ROOT)
    if args.dry_run:
        print(
            {
                "setting": setting["id"],
                "fixture": args.fixture_id,
                "cases": [case["id"] for case in cases],
                "deepinv_revision": revision,
                "destination": str(destination),
            }
        )
        return
    require_clean_repo(
        REPO_ROOT,
        (
            "deepinv/models/wrapper.py",
            "deepinv/sampling/diffusion.py",
            "deepinv/sampling/__init__.py",
            "reproduction/dps",
            "reproduction/diffpir",
        ),
    )
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but CUDA is unavailable")
    if (destination / "manifest.json").exists() and "deepinv" in read_json(
        destination / "manifest.json"
    ).get("implementations", {}):
        raise FileExistsError("this run already contains DeepInv outputs")

    configure_determinism(setting["randomness"]["fixture_seed"])
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    import deepinv as dinv

    device = torch.device(args.device)
    schedule = load_record(
        fixture,
        fixture_manifest["schedule"],
        required=(
            "sampled_timesteps",
            "betas",
            "alpha_cumprod",
            "model_alpha_cumprod",
            "noise_levels",
        ),
    )
    algorithm = build_algorithm(setting, checkpoint, schedule, device)
    sampled_timesteps = torch.stack(
        [
            algorithm.find_nearest(
                algorithm.reduced_alpha_cumprod,
                algorithm.sigmas[index],
            )
            for index in algorithm.seq
        ]
    ).to(dtype=torch.int64, device="cpu")
    if not torch.equal(sampled_timesteps, schedule["sampled_timesteps"]):
        raise ValueError("DeepInv sampled timesteps differ from the fixture")
    sampled_noise_levels = algorithm.reduced_alpha_cumprod[
        sampled_timesteps.to(device)
    ].cpu()
    expected_noise_levels = schedule["noise_levels"][sampled_timesteps]
    if not torch.equal(sampled_noise_levels, expected_noise_levels):
        raise ValueError("DeepInv noise levels differ from the fixture")

    probes = setting["trajectory_probe_steps"]
    task = setting["task"]
    case_records = []
    for case in cases:
        case_id = case["id"]
        required = (
            ("measurement", "mask", "x_init")
            if task["name"] == "inpainting"
            else ("measurement", "kernel", "x_init")
        )
        tensors = load_record(fixture, case, required=required)
        y = tensors["measurement"].to(device)
        initial_state = tensors["x_init"].to(device)
        transition_noise = load_record(
            fixture,
            case["transition_noise"],
            required=("transition_noise",),
        )["transition_noise"]
        if task["name"] == "inpainting":
            physics = dinv.physics.Inpainting(
                img_size=tuple(initial_state.shape[1:]),
                mask=tensors["mask"].to(device),
                device=device,
            )
        elif task["name"] in {"gaussian_deblur", "motion_deblur"}:

            class OfficialDiffPIRBlurFFT(dinv.physics.BlurFFT):
                def prox_l2(self, z, y, gamma, **kwargs):
                    return official_diffpir_deblur_prox(z, y, self.filter, gamma)

            physics = OfficialDiffPIRBlurFFT(
                img_size=tuple(initial_state.shape[1:]),
                filter=tensors["kernel"].to(device),
                device=device,
            )
        elif task["name"] == "super_resolution":

            class OfficialDiffPIRDownsampling(dinv.physics.Downsampling):
                def prox_l2(self, z, y, gamma, **kwargs):
                    return official_diffpir_deblur_prox(
                        z, y, self.filter, gamma, factor=self.factor
                    )

            physics = OfficialDiffPIRDownsampling(
                img_size=tuple(initial_state.shape[1:]),
                filter=tensors["kernel"].to(device),
                factor=task["factor"],
                padding="circular",
                device=device,
            )
        else:
            raise ValueError(f"unsupported DiffPIR task: {task['name']}")
        reconstruction, full_trajectory = algorithm(
            y,
            physics,
            initial_state=initial_state,
            transition_noise=transition_noise,
            get_trajectory=True,
        )
        trajectory_indices = torch.tensor([0, *(step + 1 for step in probes)])
        output = {
            "reconstruction": reconstruction.detach().cpu().mul(2).sub(1),
            "trajectory": full_trajectory[trajectory_indices].detach().cpu(),
            "trajectory_steps": torch.tensor([-1, *probes], dtype=torch.int64),
            "timesteps": sampled_timesteps,
            "noise_levels": sampled_noise_levels,
        }
        output_path = destination / "cases" / case_id / "deepinv.pt"
        record = save_tensors(output_path, output)
        record.update(
            {
                "id": case_id,
                "path": str(output_path.relative_to(destination)),
                "fixture_tensor_sha256": case["tensor_sha256"],
            }
        )
        case_records.append(record)

    update_run_manifest(
        destination / "manifest.json",
        setting_id=setting["id"],
        setting_sha256=file_sha256(setting_file),
        fixture_id=args.fixture_id,
        fixture_manifest_sha256=file_sha256(fixture_manifest_path),
        run_id=args.run_id,
        implementation="deepinv",
        record={
            "created_at": utc_now(),
            "command": command_line(),
            "repository": str(REPO_ROOT),
            "revision": revision,
            "checkpoint_sha256": checkpoint_hash,
            "environment": environment(args.device),
            "cases": case_records,
        },
    )
    print(destination / "manifest.json")


if __name__ == "__main__":
    main()
