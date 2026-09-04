"""Create immutable inputs matching DiffPIR's official random-mask convention."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

DPS_DIR = Path(__file__).resolve().parents[1] / "dps"
if str(DPS_DIR) not in sys.path:
    sys.path.insert(0, str(DPS_DIR))

from _common import (  # noqa: E402
    artifact_root,
    file_sha256,
    fixture_dir,
    load_setting,
    save_tensors,
    tensor_dict_sha256,
    utc_now,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--setting", required=True)
    parser.add_argument(
        "--fixture-id", default="ffhq256_inpainting_diffpir_quad20_v1"
    )
    parser.add_argument("--images", required=True)
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--artifact-root")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def official_random_mask(size: int, missing_probability: float) -> torch.Tensor:
    probability = np.random.uniform(missing_probability, missing_probability)
    missing = np.random.choice(size * size, int(size * size * probability), False)
    mask = torch.ones(size * size, dtype=torch.float32)
    mask[torch.from_numpy(missing)] = 0
    return mask.view(1, 1, size, size).repeat(1, 3, 1, 1)


def main() -> None:
    args = parse_args()
    setting_file, setting = load_setting(args.setting)
    if setting["algorithm"]["name"] != "diffpir":
        raise ValueError("DiffPIR fixture builder requires algorithm.name='diffpir'")
    if args.limit < 1:
        raise ValueError("--limit must be positive")

    root = artifact_root(args.artifact_root)
    destination = fixture_dir(root, args.fixture_id)
    if destination.exists() and not args.dry_run:
        raise FileExistsError(f"fixture already exists: {destination}")
    images = sorted(Path(args.images).glob("*.png"))[: args.limit]
    if len(images) != args.limit:
        raise ValueError("not enough PNG images for --limit")
    if args.dry_run:
        print({"fixture": str(destination), "images": [p.name for p in images]})
        return

    task = setting["task"]
    sampler = setting["sampler"]
    height, width = task["image_size"][-2:]
    if height != width:
        raise ValueError("the official random-mask generator requires square images")

    betas = torch.from_numpy(
        np.linspace(
            sampler["beta_start"],
            sampler["beta_end"],
            sampler["train_steps"],
            dtype=np.float32,
        )
    )
    alpha_cumprod = np.cumprod(1 - betas, axis=0)
    model_betas = torch.from_numpy(
        np.linspace(
            sampler["beta_start"],
            sampler["beta_end"],
            sampler["train_steps"],
            dtype=np.float64,
        )
    )
    model_alpha_cumprod = torch.from_numpy(np.cumprod(1 - model_betas.numpy()))
    noise_levels = torch.sqrt(1 - alpha_cumprod) / torch.sqrt(alpha_cumprod)
    sequence = np.sqrt(
        np.linspace(0, sampler["train_steps"] ** 2, sampler["sampling_steps"])
    ).astype(np.int64)
    sequence[-1] -= 1
    sampled_timesteps = torch.tensor(
        [
            torch.abs(noise_levels - noise_levels.flip(0)[index]).argmin()
            for index in sequence
        ],
        dtype=torch.int64,
    )
    schedule = {
        "timesteps": torch.arange(sampler["train_steps"], dtype=torch.int64),
        "sampled_timesteps": sampled_timesteps,
        "betas": betas,
        "alpha_cumprod": alpha_cumprod,
        "model_alpha_cumprod": model_alpha_cumprod,
        "noise_levels": noise_levels,
    }

    destination.mkdir(parents=True)
    schedule_record = save_tensors(destination / "schedule.pt", schedule)
    schedule_record["path"] = "schedule.pt"
    np.random.seed(setting["randomness"]["fixture_seed"])
    cases = []
    for index, image_path in enumerate(images):
        case_id = f"{index:05d}"
        with Image.open(image_path) as source:
            image = np.asarray(source.convert("RGB"))
        if image.shape != (height, width, 3):
            raise ValueError(f"official input must be {height}x{width}: {image_path}")

        clean_01 = torch.from_numpy(image.copy()).permute(2, 0, 1).unsqueeze(0)
        clean_01 = clean_01.float().div(255)
        mask = official_random_mask(height, task["missing_probability"])
        measurement = clean_01 * mask
        measurement_noise = torch.from_numpy(
            np.random.normal(
                0, task["measurement_noise_sigma"] * 2, (height, width, 3)
            ).astype(np.float32)
        ).permute(2, 0, 1).unsqueeze(0).div(2)
        measurement = measurement + measurement_noise

        initial_generator = torch.Generator().manual_seed(
            setting["randomness"]["x_init_seed"] + index
        )
        initial_noise = torch.randn(clean_01.shape, generator=initial_generator)
        initial_state = (
            torch.sqrt(alpha_cumprod[-1]) * (2 * measurement - 1)
            + torch.sqrt(1 - alpha_cumprod[-1]) * initial_noise
        )
        transition_generator = torch.Generator().manual_seed(
            setting["randomness"]["transition_seed"] + index
        )
        transition_noise = torch.randn(
            (sampler["sampling_steps"] - 1, *clean_01.shape),
            generator=transition_generator,
        )

        tensors = {
            "ground_truth": clean_01.mul(2).sub(1),
            "measurement": measurement,
            "effective_measurement": measurement * mask,
            "mask": mask,
            "measurement_noise": measurement_noise,
            "initial_noise": initial_noise,
            "x_init": initial_state,
        }
        record = save_tensors(destination / "cases" / f"{case_id}.pt", tensors)
        record.update(
            {
                "id": case_id,
                "path": f"cases/{case_id}.pt",
                "sources": {
                    "image": {
                        "name": image_path.name,
                        "sha256": file_sha256(image_path),
                    }
                },
            }
        )
        noise_record = save_tensors(
            destination / "noise" / f"{case_id}.pt",
            {"transition_noise": transition_noise},
        )
        record["transition_noise"] = {
            **noise_record,
            "path": f"noise/{case_id}.pt",
            "seed": setting["randomness"]["transition_seed"] + index,
        }
        cases.append(record)

    manifest = {
        "schema_version": 1,
        "fixture_id": args.fixture_id,
        "created_at": utc_now(),
        "source_setting_id": setting["id"],
        "source_setting_sha256": file_sha256(setting_file),
        "schedule": schedule_record,
        "schedule_tensor_sha256": tensor_dict_sha256(schedule),
        "randomness": setting["randomness"],
        "cases": cases,
    }
    write_json(destination / "manifest.json", manifest)
    print(destination / "manifest.json")


if __name__ == "__main__":
    main()
