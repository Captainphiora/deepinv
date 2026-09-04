"""Run the pinned official DiffPIR model and task update."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

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
    fixture_dir,
    git_revision,
    load_record,
    load_setting,
    read_json,
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
    parser.add_argument("--reference-repo", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--case", action="append")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--artifact-root")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def verify_reference(repo: Path, commit: str, task: str) -> str:
    actual = git_revision(repo)
    if actual != commit:
        raise RuntimeError(f"DiffPIR reference must be at {commit}; got {actual}")
    paths = [
        "guided_diffusion",
        "utils/utils_model.py",
    ]
    if task == "inpainting":
        paths.extend(
            ("main_ddpir.py", "utils/utils_inpaint.py", "configs/inpaint.yaml")
        )
    elif task in {"gaussian_deblur", "motion_deblur"}:
        paths.extend(("main_ddpir_deblur.py", "utils/utils_sisr.py"))
    else:
        raise ValueError(f"unsupported DiffPIR reference task: {task}")
    status = subprocess.run(
        ["git", "-C", str(repo), "status", "--porcelain", "--", *paths],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if status:
        raise RuntimeError("reference implementation is dirty:\n" + status)
    return actual


def build_model(repo: Path, checkpoint: Path, device: torch.device):
    sys.path.insert(0, str(repo))
    from guided_diffusion.script_util import create_model_and_diffusion

    model, diffusion = create_model_and_diffusion(
        image_size=256,
        class_cond=False,
        learn_sigma=True,
        num_channels=128,
        num_res_blocks=1,
        channel_mult="",
        num_heads=4,
        num_head_channels=64,
        num_heads_upsample=-1,
        attention_resolutions="16",
        dropout=0.1,
        diffusion_steps=1000,
        noise_schedule="linear",
        timestep_respacing="",
        use_kl=False,
        predict_xstart=False,
        rescale_timesteps=False,
        rescale_learned_sigmas=False,
        use_checkpoint=False,
        use_scale_shift_norm=True,
        resblock_updown=True,
        use_fp16=False,
        use_new_attention_order=False,
    )
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    validate_tensor_dict(state)
    model.load_state_dict(state, strict=True)
    return model.to(device).eval(), diffusion


def main() -> None:
    args = parse_args()
    setting_file, setting = load_setting(args.setting)
    if setting["algorithm"]["name"] != "diffpir":
        raise ValueError("DiffPIR runner requires algorithm.name='diffpir'")
    task = setting["task"]
    repo = Path(args.reference_repo).resolve()
    checkpoint = Path(args.checkpoint).resolve()
    revision = verify_reference(repo, setting["reference"]["commit"], task["name"])
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
    if args.dry_run:
        print(
            {
                "setting": setting["id"],
                "fixture": args.fixture_id,
                "cases": [case["id"] for case in cases],
                "reference_commit": revision,
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
    device = torch.device(args.device)
    model, diffusion = build_model(repo, checkpoint, device)
    if task["name"] in {"gaussian_deblur", "motion_deblur"}:
        from utils import utils_sisr as sr

    schedule = load_record(
        fixture,
        fixture_manifest["schedule"],
        required=("sampled_timesteps", "alpha_cumprod", "noise_levels"),
    )
    sampled_timesteps = schedule["sampled_timesteps"]
    sampled_noise_levels = schedule["noise_levels"][sampled_timesteps]
    probes = setting["trajectory_probe_steps"]
    algorithm = setting["algorithm"]
    sampler = setting["sampler"]
    case_records = []

    with torch.no_grad():
        for case in cases:
            case_id = case["id"]
            required = (
                ("measurement", "mask", "x_init")
                if task["name"] == "inpainting"
                else ("measurement", "kernel", "x_init")
            )
            tensors = load_record(fixture, case, required=required)
            y = tensors["measurement"].to(device)
            x = tensors["x_init"].to(device)
            if task["name"] == "inpainting":
                mask = tensors["mask"].to(device)
            else:
                kernel = tensors["kernel"].to(device)
                FB, FBC, F2B, FBFy = sr.pre_calculate(y, kernel, sf=1)
            noise = load_record(
                fixture,
                case["transition_noise"],
                required=("transition_noise",),
            )["transition_noise"].to(device)
            if noise.shape != (sampler["sampling_steps"] - 1, *x.shape):
                raise ValueError(f"invalid transition-noise tape for {case_id}")

            trajectory = [x.cpu()]
            for step, timestep in enumerate(sampled_timesteps.tolist()):
                time = torch.full(
                    (x.shape[0],), timestep, dtype=torch.long, device=device
                )
                x0 = diffusion.p_mean_variance(
                    model, x, time, clip_denoised=sampler["clip_denoised"]
                )["pred_xstart"]
                if step < sampler["sampling_steps"] - 1:
                    rho = (
                        algorithm["lambda_"]
                        * task["algorithm_sigma"] ** 2
                        / schedule["noise_levels"][timestep].to(device) ** 2
                    )
                    if task["name"] == "inpainting":
                        x0_prox = (mask * (2 * y - 1) + rho * x0) / (mask + rho)
                    else:
                        x0_prox = (
                            sr.data_solution(
                                x0.float().div(2).add(0.5),
                                FB,
                                FBC,
                                F2B,
                                FBFy,
                                rho.float().repeat(1, 1, 1, 1),
                                sf=1,
                            )
                            .mul(2)
                            .sub(1)
                        )
                    x0 = x0 + algorithm["guidance_scale"] * (x0_prox - x0)
                    next_timestep = sampled_timesteps[step + 1].item()
                    eps = (
                        x
                        - torch.sqrt(schedule["alpha_cumprod"][timestep]).to(device)
                        * x0
                    ) / torch.sqrt(1 - schedule["alpha_cumprod"][timestep]).to(device)
                    next_alpha = schedule["alpha_cumprod"][next_timestep].to(device)
                    x = (
                        torch.sqrt(next_alpha) * x0
                        + (1 - sampler["zeta"]) ** 0.5
                        * torch.sqrt(1 - next_alpha)
                        * eps
                        + sampler["zeta"] ** 0.5
                        * torch.sqrt(1 - next_alpha)
                        * noise[step]
                    )
                trajectory.append(x.cpu())

            full_trajectory = torch.stack(trajectory)
            trajectory_indices = torch.tensor([0, *(step + 1 for step in probes)])
            output = {
                "reconstruction": x.cpu(),
                "trajectory": full_trajectory[trajectory_indices],
                "trajectory_steps": torch.tensor([-1, *probes], dtype=torch.int64),
                "timesteps": sampled_timesteps,
                "noise_levels": sampled_noise_levels,
            }
            output_path = destination / "cases" / case_id / "reference.pt"
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
        implementation="reference",
        record={
            "created_at": utc_now(),
            "command": command_line(),
            "repository": str(repo),
            "revision": revision,
            "checkpoint_sha256": checkpoint_hash,
            "environment": environment(args.device),
            "cases": case_records,
        },
    )
    print(destination / "manifest.json")


if __name__ == "__main__":
    main()
