"""Create immutable inputs matching DiffPIR's official task conventions."""

from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy import ndimage

DPS_DIR = Path(__file__).resolve().parents[1] / "dps"
if str(DPS_DIR) not in sys.path:
    sys.path.insert(0, str(DPS_DIR))

from _common import (  # noqa: E402
    REPO_ROOT,
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
    parser.add_argument("--fixture-id", default="ffhq256_inpainting_diffpir_quad20_v1")
    parser.add_argument("--images", required=True)
    parser.add_argument("--motionblur-repo")
    parser.add_argument("--reference-repo")
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


def official_gaussian_kernel(size: int, std: float) -> torch.Tensor:
    if size < 1 or size % 2 == 0 or std <= 0:
        raise ValueError("Gaussian kernel size must be positive and odd; std > 0")
    impulse = np.zeros((size, size), dtype=np.float64)
    impulse[size // 2, size // 2] = 1
    kernel = ndimage.gaussian_filter(impulse, sigma=std).astype(np.float32)
    return torch.from_numpy(kernel).view(1, 1, size, size)


def load_motion_kernel_type(repo: Path):
    source = repo / "motionblur.py"
    spec = importlib.util.spec_from_file_location("_diffpir_motionblur", source)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load motionblur source: {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Kernel


def official_motion_kernel(
    kernel_type: type, size: int, intensity: float, case_index: int
) -> torch.Tensor:
    """Match MotionBlurOperator's two Kernel constructions per image."""
    np.random.seed(case_index * 10)
    kernel_type(size=(size, size), intensity=intensity).kernelMatrix
    kernel = kernel_type(size=(size, size), intensity=intensity).kernelMatrix
    return torch.from_numpy(kernel.copy()).float().view(1, 1, size, size)


def official_circular_blur(image: np.ndarray, kernel: torch.Tensor) -> torch.Tensor:
    """Match the official uint8 SciPy degradation before float conversion."""
    kernel_np = kernel.squeeze().numpy()
    blurred = ndimage.convolve(image, np.expand_dims(kernel_np, axis=2), mode="wrap")
    return (
        torch.from_numpy(blurred.copy()).permute(2, 0, 1).unsqueeze(0).float().div(255)
    )


def validate_random_streams(randomness: dict) -> None:
    policy = randomness.get("stream_policy", "legacy")
    if policy == "legacy":
        return
    if policy != "independent_generators_with_distinct_seeds":
        raise ValueError(f"unsupported random stream policy: {policy}")
    if randomness["x_init_seed"] == randomness["transition_seed"]:
        raise ValueError("independent x_init and transition streams need distinct seeds")


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
    validate_random_streams(setting["randomness"])
    height, width = task["image_size"][-2:]
    if height != width and task["name"] == "inpainting":
        raise ValueError("the official random-mask generator requires square images")
    if task["name"] not in {
        "inpainting",
        "gaussian_deblur",
        "motion_deblur",
        "super_resolution",
    }:
        raise ValueError(f"unsupported DiffPIR fixture task: {task['name']}")

    sr_kernel = None
    sr_dependency = None
    if task["name"] == "super_resolution":
        if not args.reference_repo:
            raise ValueError("--reference-repo is required for SR")
        sys.path.insert(0, str(REPO_ROOT))
        from reproduction.diffpir.run_reference import verify_reference
        from scipy.io import loadmat
        import cv2

        repo = Path(args.reference_repo).resolve()
        revision = verify_reference(repo, setting["reference"]["commit"], task["name"])
        kernel_file = repo / task["kernel_file"]
        if file_sha256(kernel_file) != task["kernel_file_sha256"]:
            raise ValueError("SR kernel file SHA256 does not match the setting")
        factor = task["factor"]
        if factor != 4 or height % factor or width % factor:
            raise ValueError(
                "this SR setting requires factor=4 and divisible image sizes"
            )
        sr_kernel = torch.from_numpy(
            loadmat(kernel_file)["kernels"][0, factor - 2].copy()
        ).float()[None, None]
        sys.path.insert(0, str(repo))
        from utils.utils_image import imresize_np

        sr_dependency = {
            "repository": str(repo),
            "revision": revision,
            "kernel_file": task["kernel_file"],
            "kernel_file_sha256": task["kernel_file_sha256"],
            "opencv": cv2.__version__,
        }

    motionblur = None
    motion_kernel_type = None
    if task["name"] == "motion_deblur":
        if not args.motionblur_repo:
            raise ValueError("--motionblur-repo is required for motion deblur")
        motionblur_repo = Path(args.motionblur_repo).resolve()
        source = motionblur_repo / "motionblur.py"
        revision = subprocess.run(
            [
                "git",
                "-c",
                f"safe.directory={motionblur_repo}",
                "-C",
                str(motionblur_repo),
                "rev-parse",
                "HEAD",
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if revision != task["motionblur_commit"]:
            raise ValueError(
                "motionblur repository revision does not match the setting"
            )
        if file_sha256(source) != task["motionblur_source_sha256"]:
            raise ValueError("motionblur.py SHA256 does not match the setting")
        motion_kernel_type = load_motion_kernel_type(motionblur_repo)
        motionblur = {
            "repository": str(motionblur_repo),
            "revision": revision,
            "source_sha256": task["motionblur_source_sha256"],
        }

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
    kernel = (
        official_gaussian_kernel(task["kernel_size"], task["kernel_std"])
        if task["name"] == "gaussian_deblur"
        else sr_kernel
    )
    cases = []
    for index, image_path in enumerate(images):
        case_id = f"{index:05d}"
        with Image.open(image_path) as source:
            image = np.asarray(source.convert("RGB"))
        if image.shape != (height, width, 3):
            raise ValueError(f"official input must be {height}x{width}: {image_path}")

        clean_01 = torch.from_numpy(image.copy()).permute(2, 0, 1).unsqueeze(0)
        clean_01 = clean_01.float().div(255)
        if task["name"] == "motion_deblur":
            kernel = official_motion_kernel(
                motion_kernel_type,
                task["kernel_size"],
                task["kernel_intensity"],
                index,
            )
        if task["name"] == "inpainting":
            mask = official_random_mask(height, task["missing_probability"])
            clean_measurement = clean_01 * mask
            measurement_noise = (
                torch.from_numpy(
                    np.random.normal(
                        0, task["measurement_noise_sigma"] * 2, (height, width, 3)
                    ).astype(np.float32)
                )
                .permute(2, 0, 1)
                .unsqueeze(0)
                .div(2)
            )
            measurement = clean_measurement + measurement_noise
        else:
            if task["name"] == "super_resolution":
                resized = imresize_np(image.astype(np.float32) / 255, 1 / factor)
                clean_measurement = (
                    torch.from_numpy(resized.copy()).permute(2, 0, 1).unsqueeze(0)
                )
            else:
                clean_measurement = official_circular_blur(image, kernel)
            noisy = clean_measurement.squeeze(0).permute(1, 2, 0).numpy().copy()
            noisy = noisy * 2 - 1
            noise_rng = np.random.RandomState(setting["randomness"]["measurement_seed"])
            noisy += noise_rng.normal(
                0, task["measurement_noise_sigma"] * 2, noisy.shape
            )
            noisy = noisy / 2 + 0.5
            measurement = torch.from_numpy(noisy).permute(2, 0, 1).unsqueeze(0)
            measurement_noise = measurement - clean_measurement

        initial_generator = torch.Generator().manual_seed(
            setting["randomness"]["x_init_seed"] + index
        )
        initial_noise = torch.randn(clean_01.shape, generator=initial_generator)
        initial_image = measurement
        if task["name"] == "super_resolution":
            upsampled = cv2.resize(
                measurement.squeeze(0).permute(1, 2, 0).numpy(),
                (width, height),
                interpolation=cv2.INTER_CUBIC,
            )
            initial_image = (
                torch.from_numpy(upsampled.copy()).permute(2, 0, 1).unsqueeze(0)
            )
        if task["name"] in {"gaussian_deblur", "motion_deblur"}:
            t_y = torch.abs(noise_levels - 2 * task["noise_level"]).argmin()
            sqrt_alpha_effective = torch.sqrt(alpha_cumprod[-1]) / torch.sqrt(
                alpha_cumprod[t_y]
            )
            initial_state = (
                sqrt_alpha_effective * (2 * measurement - 1)
                + torch.sqrt(
                    (1 - alpha_cumprod[-1])
                    - sqrt_alpha_effective**2 * (1 - alpha_cumprod[t_y])
                )
                * initial_noise
            )
        else:
            initial_state = (
                torch.sqrt(alpha_cumprod[-1]) * (2 * initial_image - 1)
                + torch.sqrt(1 - alpha_cumprod[-1]) * initial_noise
            )
        transition_generator = torch.Generator().manual_seed(
            setting["randomness"]["transition_seed"] + index
        )
        transition_noise = torch.randn(
            (sampler["sampling_steps"] - 1, *clean_01.shape),
            generator=transition_generator,
        )
        if setting["randomness"].get("stream_policy") == (
            "independent_generators_with_distinct_seeds"
        ) and torch.equal(initial_noise, transition_noise[0]):
            raise RuntimeError("initial and first transition noise must be independent")

        tensors = {
            "ground_truth": clean_01.mul(2).sub(1),
            "measurement": measurement,
            "clean_measurement": clean_measurement,
            "measurement_noise": measurement_noise,
            "initial_noise": initial_noise,
            "x_init": initial_state,
        }
        if task["name"] == "inpainting":
            tensors.update({"effective_measurement": measurement * mask, "mask": mask})
        else:
            tensors["kernel"] = kernel
        if task["name"] == "super_resolution":
            tensors["initial_image"] = initial_image
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
        "task": task["name"],
        "schedule": schedule_record,
        "schedule_tensor_sha256": tensor_dict_sha256(schedule),
        "randomness": setting["randomness"],
        "cases": cases,
    }
    if motionblur is not None:
        manifest["dependencies"] = {"motionblur": motionblur}
    if sr_dependency is not None:
        manifest["dependencies"] = {"official_sr": sr_dependency}
    write_json(destination / "manifest.json", manifest)
    print(destination / "manifest.json")


if __name__ == "__main__":
    main()
