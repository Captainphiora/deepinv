"""Run the pinned original DSG implementation on a canonical fixture."""

from __future__ import annotations

import argparse
import subprocess
import sys
from functools import partial
from pathlib import Path
from unittest import mock

import torch

DPS_DIR = Path(__file__).resolve().parents[1] / "dps"
if str(DPS_DIR) not in sys.path:
    sys.path.insert(0, str(DPS_DIR))

from _common import (  # noqa: E402
    artifact_root,
    command_line,
    configure_determinism,
    environment,
    file_sha256,
    fixed_randn_like,
    fixture_dir,
    git_revision,
    load_record,
    load_setting,
    read_json,
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
    parser.add_argument(
        "--fixture-id", default="ffhq256_inpainting_ddim100_eta1_dsg_v1"
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--reference-repo",
        required=True,
        help="path to the DSG2024 Linear_Inverse_Problems directory",
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--case", action="append", help="case id; repeat as needed")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--artifact-root")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def verify_reference(repo: Path, commit: str) -> str:
    if not (repo / "guided_diffusion").is_dir():
        raise ValueError(
            "--reference-repo must be the DSG2024 Linear_Inverse_Problems directory"
        )
    actual = git_revision(repo)
    if actual != commit:
        raise RuntimeError(
            f"DSG reference must be checked out at compatibility commit {commit}; "
            f"got {actual}"
        )
    status = subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "status",
            "--porcelain",
            "--untracked-files=all",
            "--",
            "guided_diffusion",
            "util",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if status:
        raise RuntimeError("reference implementation is dirty:\n" + status)
    return actual


def import_reference(repo: Path):
    sys.path.insert(0, str(repo))
    from guided_diffusion.condition_methods import get_conditioning_method
    from guided_diffusion.gaussian_diffusion import create_sampler
    from guided_diffusion.measurements import get_noise, get_operator
    from guided_diffusion.unet import create_model

    return (
        create_model,
        create_sampler,
        get_operator,
        get_noise,
        get_conditioning_method,
    )


def main() -> None:
    args = parse_args()
    setting_file, setting = load_setting(args.setting)
    if setting["algorithm"]["name"] != "dsg":
        raise ValueError("DSG runner requires an algorithm.name of 'dsg'")
    setting_hash = file_sha256(setting_file)
    repo = Path(args.reference_repo).resolve()
    checkpoint = Path(args.checkpoint).resolve()
    compatibility_commit = setting["reference"]["compatibility_commit"]
    actual_commit = verify_reference(repo, compatibility_commit)
    checkpoint_hash = file_sha256(checkpoint)
    if checkpoint_hash != setting["model"]["checkpoint_sha256"]:
        raise ValueError("checkpoint SHA256 does not match the setting")

    root = artifact_root(args.artifact_root)
    fixture = fixture_dir(root, args.fixture_id)
    fixture_manifest_path = fixture / "manifest.json"
    fixture_manifest = read_json(fixture_manifest_path)
    fixture_manifest_hash = file_sha256(fixture_manifest_path)
    cases = select_cases(fixture_manifest, args.case)
    if not cases:
        raise ValueError("fixture contains no selected cases")
    destination = run_dir(root, setting["id"], args.run_id, "dsg")
    if args.dry_run:
        print(
            {
                "setting": setting["id"],
                "fixture": args.fixture_id,
                "cases": [case["id"] for case in cases],
                "reference_commit": actual_commit,
                "upstream_commit": setting["reference"]["upstream_commit"],
                "checkpoint_sha256": checkpoint_hash,
                "destination": str(destination),
            }
        )
        return
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but CUDA is unavailable")
    if (destination / "manifest.json").exists() and "reference" in read_json(
        destination / "manifest.json"
    ).get("implementations", {}):
        raise FileExistsError("this run already contains reference outputs")

    configure_determinism(setting["randomness"]["fixture_seed"])
    create_model, create_sampler, get_operator, get_noise, get_conditioner = (
        import_reference(repo)
    )
    device = torch.device(args.device)
    state_dict = torch.load(checkpoint, map_location="cpu", weights_only=True)
    validate_tensor_dict(state_dict)
    model_config = {
        "image_size": 256,
        "num_channels": 128,
        "num_res_blocks": 1,
        "channel_mult": "",
        "learn_sigma": True,
        "class_cond": False,
        "use_checkpoint": False,
        "attention_resolutions": 16,
        "num_heads": 4,
        "num_head_channels": 64,
        "num_heads_upsample": -1,
        "use_scale_shift_norm": True,
        "dropout": 0.0,
        "resblock_updown": True,
        "use_fp16": False,
        "use_new_attention_order": False,
        "model_path": str(checkpoint),
    }
    with mock.patch.object(torch, "load", return_value=state_dict):
        model = create_model(**model_config)
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device).eval()

    sampler_config = setting["sampler"]
    sampler = create_sampler(
        sampler=sampler_config["name"],
        steps=sampler_config["train_steps"],
        noise_schedule=sampler_config["noise_schedule"],
        model_mean_type=setting["model"]["prediction_type"],
        model_var_type=setting["model"]["variance_type"],
        dynamic_threshold=False,
        clip_denoised=sampler_config["clip_denoised"],
        rescale_timesteps=sampler_config["rescale_timesteps"],
        timestep_respacing=sampler_config["timestep_respacing"],
        eta=sampler_config["eta"],
    )
    operator = get_operator(name="inpainting", device=device)
    noiser = get_noise(
        name="gaussian", sigma=setting["task"]["measurement_noise_sigma"]
    )
    algorithm_config = setting["algorithm"]
    conditioner = get_conditioner(
        "DSG",
        operator,
        noiser,
        guidance_scale=algorithm_config["guidance_scale"],
        interval=algorithm_config["interval"],
    )
    schedule = load_record(
        fixture,
        fixture_manifest["schedule"],
        required=("timesteps", "betas", "alpha_cumprod", "noise_levels"),
    )
    timestep_map = torch.as_tensor(sampler.timestep_map, dtype=torch.int64).flip(0)
    used_noise_levels = torch.from_numpy(
        ((1.0 - sampler.alphas_cumprod) / sampler.alphas_cumprod) ** 0.5
    ).flip(0)
    if not torch.allclose(
        used_noise_levels,
        schedule["noise_levels"][timestep_map],
        rtol=0,
        atol=1e-12,
    ):
        raise ValueError("reference sampler noise levels differ from the fixture")

    probes = set(setting["trajectory_probe_steps"])
    case_records = []
    for case in cases:
        case_id = case["id"]
        tensors = load_record(fixture, case, required=("measurement", "mask", "x_init"))
        measurement = tensors["measurement"].to(device)
        mask = tensors["mask"].to(device)
        image = tensors["x_init"].to(device)
        if image.shape[0] != setting["task"]["batch_size"]:
            raise ValueError(f"fixture batch size differs for case {case_id}")
        if not requires_transition_noise(setting):
            raise ValueError("the DSG eta=1 setting requires transition noise")
        noise_record = case.get("transition_noise")
        if noise_record is None:
            raise FileNotFoundError(
                "DSG DDIM eta>0 requires an explicit transition-noise tape; "
                "rerun prepare_inputs with --with-transition-noise"
            )
        transition_noise = load_record(
            fixture, noise_record, required=("transition_noise",)
        )["transition_noise"]
        if transition_noise.shape != (sampler.num_timesteps, *image.shape):
            raise ValueError(f"invalid transition-noise tape for case {case_id}")

        trajectory = [image.detach().cpu()]
        trajectory_steps = [-1]
        distances = []
        conditioning = partial(conditioner.conditioning, mask=mask)
        for step, sampler_index in enumerate(range(sampler.num_timesteps - 1, -1, -1)):
            time = torch.full(
                (image.shape[0],), sampler_index, device=device, dtype=torch.long
            )
            image = image.requires_grad_()
            draw, draw_count = fixed_randn_like(transition_noise[step])
            with mock.patch.object(torch, "randn_like", side_effect=draw):
                prediction = sampler.p_sample(model, image, time)
            if draw_count() != 1:
                raise RuntimeError(
                    "reference p_sample did not request one noise tensor"
                )
            image, distance = conditioning(
                x_t=prediction["sample"],
                x_t_mean=prediction["mean"],
                measurement=measurement,
                noisy_measurement=None,
                x_prev=image,
                x_0_hat=prediction["pred_xstart"],
                sigma_t=prediction["sigma_t"],
                idx=sampler_index,
                func=prediction["func"],
            )
            image = image.detach()
            distances.append(distance.detach().cpu())
            if step in probes:
                trajectory.append(image.cpu())
                trajectory_steps.append(step)

        output = {
            "reconstruction": image.cpu(),
            "trajectory": torch.stack(trajectory),
            "trajectory_steps": torch.tensor(trajectory_steps, dtype=torch.int64),
            "timesteps": timestep_map,
            "noise_levels": used_noise_levels,
            "distances": torch.stack(distances),
        }
        output_path = destination / "cases" / case_id / "reference.pt"
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
        fixture_manifest_sha256=fixture_manifest_hash,
        run_id=args.run_id,
        implementation="reference",
        record={
            "created_at": utc_now(),
            "command": command_line(),
            "environment": environment(args.device),
            "repository": str(repo),
            "revision": actual_commit,
            "compatibility_revision": compatibility_commit,
            "upstream_revision": setting["reference"]["upstream_commit"],
            "checkpoint_sha256": checkpoint_hash,
            "cases": case_records,
        },
    )
    print(destination / "manifest.json")


if __name__ == "__main__":
    main()
