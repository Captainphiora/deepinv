"""Convert fixed legacy DPS inputs into immutable tensor-dict fixtures."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from _common import (
    artifact_root,
    file_sha256,
    fixture_dir,
    load_setting,
    requires_transition_noise,
    save_tensors,
    tensor_dict_sha256,
    utc_now,
    write_json,
)


CASE_FILE = re.compile(r"^[0-9]{5}\.pt$")


def image_tensor(path: Path, size: tuple[int, int]) -> torch.Tensor:
    with Image.open(path) as source:
        image = source.convert("RGB").resize(
            size[::-1], Image.Resampling.BILINEAR
        )
    array = np.asarray(image, dtype=np.float32).copy() / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).mul(2).sub(1)


def legacy_measurement(path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    value = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise TypeError(f"expected legacy [measurement, mask] in {path}")
    measurement, mask = value
    if not isinstance(measurement, torch.Tensor) or not isinstance(mask, torch.Tensor):
        raise TypeError(f"expected two tensors in {path}")
    return measurement, mask


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--setting", required=True, help="setting JSON path or name")
    parser.add_argument("--fixture-id", default="ffhq256_inpainting_v1")
    parser.add_argument("--images", required=True, help="directory of clean PNG images")
    parser.add_argument(
        "--measurements", required=True, help="directory of legacy 00000.pt files"
    )
    parser.add_argument(
        "--x-init",
        help="directory of legacy 00000_xstart.pt files; generated if omitted",
    )
    parser.add_argument(
        "--x-init-seed",
        action="append",
        default=[],
        metavar="CASE=SEED",
        help="replace one case's x_init with a named seed, e.g. 00002=43",
    )
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--artifact-root")
    parser.add_argument(
        "--with-transition-noise",
        action="store_true",
        help=(
            "save an explicit [steps,B,C,H,W] noise tape; required for DDPM "
            "and DDIM eta>0"
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setting_file, setting = load_setting(args.setting)
    if args.limit < 1:
        raise ValueError("--limit must be positive")
    root = artifact_root(args.artifact_root)
    destination = fixture_dir(root, args.fixture_id)
    seed_overrides = {}
    for item in args.x_init_seed:
        case_id, separator, seed = item.partition("=")
        if not separator or not re.fullmatch(r"[0-9]{5}", case_id):
            raise ValueError(f"invalid --x-init-seed {item!r}; expected CASE=SEED")
        seed_overrides[case_id] = int(seed)
    if destination.exists() and not args.dry_run:
        raise FileExistsError(
            f"fixture already exists: {destination}; choose a new fixture id"
        )

    images = sorted(Path(args.images).glob("*.png"))[: args.limit]
    measurements = sorted(
        path for path in Path(args.measurements).iterdir() if CASE_FILE.match(path.name)
    )[: args.limit]
    x_init_files = (
        sorted(Path(args.x_init).glob("[0-9][0-9][0-9][0-9][0-9]_xstart.pt"))[
            : args.limit
        ]
        if args.x_init
        else []
    )
    if len(images) != args.limit or len(measurements) != args.limit:
        raise ValueError("not enough clean images or measurements for --limit")
    if args.x_init and len(x_init_files) != args.limit:
        raise ValueError("not enough x-init files for --limit")

    sampler = setting["sampler"]
    train_steps = sampler["train_steps"]
    betas = torch.from_numpy(
        np.linspace(
            sampler["beta_start"],
            sampler["beta_end"],
            train_steps,
            dtype=np.float64,
        )
    )
    alpha_cumprod = torch.from_numpy(np.cumprod(1 - betas.numpy()))
    noise_levels = torch.sqrt((1 - alpha_cumprod) / alpha_cumprod)
    schedule = {
        "timesteps": torch.arange(train_steps, dtype=torch.int64),
        "betas": betas,
        "alpha_cumprod": alpha_cumprod,
        "noise_levels": noise_levels,
    }
    preview = {
        "fixture": str(destination),
        "setting": setting["id"],
        "cases": [f"{index:05d}" for index in range(args.limit)],
        "x_init_seed_overrides": seed_overrides,
        "transition_noise": args.with_transition_noise,
        "transition_noise_required": requires_transition_noise(setting),
    }
    if args.dry_run:
        print(preview)
        return

    destination.mkdir(parents=True)
    schedule_record = save_tensors(destination / "schedule.pt", schedule)
    schedule_record["path"] = "schedule.pt"
    case_records = []
    image_size = tuple(setting["task"]["image_size"][-2:])
    seeds = setting["randomness"]
    for index, (image_path, measurement_path) in enumerate(
        zip(images, measurements, strict=True)
    ):
        case_id = f"{index:05d}"
        ground_truth = image_tensor(image_path, image_size)
        measurement, mask = legacy_measurement(measurement_path)
        if case_id in seed_overrides:
            x_init_seed = seed_overrides[case_id]
            generator = torch.Generator(device="cpu").manual_seed(x_init_seed)
            x_init = torch.randn(ground_truth.shape, generator=generator)
        elif x_init_files:
            x_init = torch.load(
                x_init_files[index], map_location="cpu", weights_only=True
            )
            if not isinstance(x_init, torch.Tensor):
                raise TypeError(f"expected tensor in {x_init_files[index]}")
        else:
            x_init_seed = seeds["x_init_seed"] + index
            generator = torch.Generator(device="cpu").manual_seed(
                x_init_seed
            )
            x_init = torch.randn(ground_truth.shape, generator=generator)
        if measurement.shape != ground_truth.shape or not torch.broadcast_shapes(
            mask.shape, ground_truth.shape
        ) == ground_truth.shape:
            raise ValueError(f"shape mismatch in case {case_id}")

        tensors = {
            "ground_truth": ground_truth,
            "measurement": measurement,
            "mask": mask,
            "measurement_noise": measurement - ground_truth * mask,
            "x_init": x_init,
        }
        record = save_tensors(destination / "cases" / f"{case_id}.pt", tensors)
        record["id"] = case_id
        record["path"] = f"cases/{case_id}.pt"
        record["sources"] = {
            "image": {"name": image_path.name, "sha256": file_sha256(image_path)},
            "measurement": {
                "name": measurement_path.name,
                "sha256": file_sha256(measurement_path),
            },
            "x_init": (
                {
                    "name": x_init_files[index].name,
                    "sha256": file_sha256(x_init_files[index]),
                }
                if x_init_files and case_id not in seed_overrides
                else {"seed": x_init_seed}
            ),
        }
        if args.with_transition_noise:
            generator = torch.Generator(device="cpu").manual_seed(
                seeds["transition_seed"] + index
            )
            tape = torch.randn(
                (sampler["sampling_steps"], *ground_truth.shape), generator=generator
            )
            noise_record = save_tensors(
                destination / "noise" / f"{case_id}.pt",
                {"transition_noise": tape},
            )
            record["transition_noise"] = {
                **noise_record,
                "path": f"noise/{case_id}.pt",
                "seed": seeds["transition_seed"] + index,
            }
        case_records.append(record)

    manifest = {
        "schema_version": 1,
        "fixture_id": args.fixture_id,
        "created_at": utc_now(),
        "source_setting_id": setting["id"],
        "source_setting_sha256": file_sha256(setting_file),
        "schedule": schedule_record,
        "schedule_tensor_sha256": tensor_dict_sha256(schedule),
        "randomness": seeds,
        "cases": case_records,
    }
    write_json(destination / "manifest.json", manifest)
    print(destination / "manifest.json")


if __name__ == "__main__":
    main()
