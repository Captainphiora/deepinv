"""Run DeepInv DiscreteDPS on a canonical fixture."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

from _common import (
    REPO_ROOT,
    artifact_root,
    command_line,
    configure_determinism,
    environment,
    file_sha256,
    fixture_dir,
    git_revision,
    load_setting,
    load_record,
    read_json,
    require_clean_repo,
    requires_transition_noise,
    run_dir,
    save_tensors,
    select_cases,
    update_run_manifest,
    utc_now,
    validate_tensor_dict,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--setting", required=True, help="setting JSON path or name")
    parser.add_argument("--fixture-id", default="ffhq256_inpainting_v1")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--case", action="append", help="case id; repeat as needed")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--artifact-root")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def build_algorithm(setting: dict, checkpoint: Path, device: torch.device):
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    import deepinv as dinv

    state_dict = torch.load(checkpoint, map_location="cpu", weights_only=True)
    validate_tensor_dict(state_dict)
    score_model = dinv.models.DiffUNet(pretrained=None)
    incompatible = score_model.load_state_dict(state_dict, strict=False)
    expected_missing = {
        "sqrt_1m_alphas_cumprod",
        "sqrt_alphas_cumprod",
    }
    if set(incompatible.missing_keys) != expected_missing or incompatible.unexpected_keys:
        raise ValueError(
            "checkpoint does not match DiffUNet: "
            f"missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}"
        )
    score_model = score_model.to(device).eval()

    sampler = setting["sampler"]
    betas = np.linspace(
        sampler["beta_start"],
        sampler["beta_end"],
        sampler["train_steps"],
        dtype=np.float64,
    )
    alpha_cumprod = torch.from_numpy(np.cumprod(1 - betas))
    wrapped_model = dinv.models.ScoreModelWrapper(
        score_model,
        prediction_type=setting["model"]["prediction_type"],
        model_input_type=setting["model"]["model_input_type"],
        variance_type=setting["model"]["variance_type"],
        model_kwargs={"type_t": "timestep"},
        sigma_t=torch.sqrt((1 - alpha_cumprod) / alpha_cumprod).float(),
        scale_t=torch.sqrt(alpha_cumprod).float(),
        n_timesteps=sampler["train_steps"],
        device=device,
    )
    rng = torch.Generator(device=device).manual_seed(
        setting["randomness"]["transition_seed"]
    )
    return dinv.sampling.DiscreteDPS(
        wrapped_model,
        sampler=sampler["name"],
        betas=betas,
        timestep_respacing=sampler["timestep_respacing"],
        eta=sampler["eta"],
        scale=setting["algorithm"]["scale"],
        clip_denoised=sampler["clip_denoised"],
        rng=rng,
        verbose=True,
    )


def main() -> None:
    args = parse_args()
    setting_file, setting = load_setting(args.setting)
    setting_hash = file_sha256(setting_file)
    checkpoint = Path(args.checkpoint).resolve()
    checkpoint_hash = file_sha256(checkpoint)
    if checkpoint_hash != setting["model"]["checkpoint_sha256"]:
        raise ValueError("checkpoint SHA256 does not match the setting")

    root = artifact_root(args.artifact_root)
    fixture = fixture_dir(root, args.fixture_id)
    fixture_manifest = read_json(fixture / "manifest.json")
    cases = select_cases(fixture_manifest, args.case)
    if not cases:
        raise ValueError("fixture contains no selected cases")
    destination = run_dir(root, setting["id"], args.run_id)
    revision = git_revision(REPO_ROOT)
    if args.dry_run:
        print(
            {
                "setting": setting["id"],
                "fixture": args.fixture_id,
                "cases": [case["id"] for case in cases],
                "deepinv_revision": revision,
                "checkpoint_sha256": checkpoint_hash,
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
    algorithm = build_algorithm(setting, checkpoint, device)
    schedule = load_record(
        fixture,
        fixture_manifest["schedule"],
        required=("timesteps", "betas", "alpha_cumprod", "noise_levels"),
    )
    sampled_timesteps = algorithm.timestep_map.detach().cpu().flip(0)
    sampled_noise_levels = algorithm.noise_levels.detach().cpu().flip(0)
    if not torch.allclose(
        sampled_noise_levels,
        schedule["noise_levels"][sampled_timesteps],
        rtol=0,
        atol=1e-12,
    ):
        raise ValueError(
            "DeepInv sampler noise levels differ from the fixture schedule"
        )
    probes = setting["trajectory_probe_steps"]
    sampler = setting["sampler"]
    case_records = []
    for case in cases:
        case_id = case["id"]
        tensors = load_record(
            fixture,
            case,
            required=("measurement", "mask", "x_init"),
        )
        measurement = tensors["measurement"].to(device)
        mask = tensors["mask"].to(device)
        x_init = tensors["x_init"].to(device)
        physics = dinv.physics.Inpainting(
            img_size=tuple(x_init.shape[1:]), mask=mask, device=device
        )
        transition_noise = None
        if requires_transition_noise(setting):
            noise_record = case.get("transition_noise")
            if noise_record is None:
                raise FileNotFoundError(
                    "DDPM and DDIM eta>0 require an explicit transition-noise "
                    "tape; rerun prepare_inputs with --with-transition-noise"
                )
            transition_noise = load_record(
                fixture, noise_record, required=("transition_noise",)
            )["transition_noise"]
        else:  # DDIM eta=0: use the same explicit zero tape as the reference.
            transition_noise = torch.zeros(
                (algorithm.num_timesteps, *x_init.shape), dtype=x_init.dtype
            )

        reconstruction, full_trajectory = algorithm(
            measurement,
            physics,
            x_init=x_init,
            seed=setting["randomness"]["transition_seed"] + int(case_id),
            transition_noise=transition_noise,
            get_trajectory=True,
        )
        trajectory_steps = torch.tensor([-1, *probes], dtype=torch.int64)
        trajectory_indices = torch.tensor([0, *(step + 1 for step in probes)])
        output = {
            "reconstruction": reconstruction.detach().cpu(),
            "trajectory": full_trajectory[trajectory_indices].detach().cpu(),
            "trajectory_steps": trajectory_steps,
            "timesteps": sampled_timesteps,
            "noise_levels": sampled_noise_levels,
        }
        output_path = destination / "cases" / case_id / "deepinv.pt"
        record = save_tensors(output_path, output)
        record["id"] = case_id
        record["path"] = str(output_path.relative_to(destination))
        record["fixture_tensor_sha256"] = case["tensor_sha256"]
        case_records.append(record)

    update_run_manifest(
        destination / "manifest.json",
        setting_id=setting["id"],
        setting_sha256=setting_hash,
        fixture_id=args.fixture_id,
        run_id=args.run_id,
        implementation="deepinv",
        record={
            "created_at": utc_now(),
            "command": command_line(),
            "environment": environment(args.device),
            "repository": str(REPO_ROOT),
            "revision": revision,
            "checkpoint_sha256": checkpoint_hash,
            "cases": case_records,
        },
    )
    print(destination / "manifest.json")


if __name__ == "__main__":
    main()
