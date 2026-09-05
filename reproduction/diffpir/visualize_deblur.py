"""Visualize fixed deblur/SR tensor artifacts using comparison.json metrics."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

DPS_DIR = Path(__file__).resolve().parents[1] / "dps"
sys.path.insert(0, str(DPS_DIR))
from _common import load_record, read_json  # noqa: E402


def display_tensor(value: torch.Tensor, *, range_01: bool = False):
    value = value.detach().float().cpu()
    if value.ndim == 4 and value.shape[0] == 1:
        value = value[0]
    if value.ndim != 3 or value.shape[0] not in (1, 3):
        raise ValueError(f"expected C=1 or 3 image tensor, got {tuple(value.shape)}")
    value = value.clamp(0, 1) if range_01 else value.clamp(-1, 1).add(1).div(2)
    return value[0].numpy() if value.shape[0] == 1 else value.permute(1, 2, 0).numpy()


def metric_title(name: str, metrics: dict) -> str:
    return (
        f"{name}\nPSNR {metrics['psnr_db']:.3f} dB | "
        f"SSIM {metrics['ssim']:.4f}\nLPIPS {metrics['lpips']:.4f}"
    )


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture-dir", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()

    fixture_dir = Path(args.fixture_dir).resolve()
    run_dir = Path(args.run_dir).resolve()
    output = (
        Path(args.output).resolve() if args.output else run_dir / "visualization.png"
    )
    fixture_manifest = read_json(fixture_dir / "manifest.json")
    run_manifest = read_json(run_dir / "manifest.json")
    comparison = read_json(run_dir / "comparison.json")
    if run_manifest["fixture_id"] != fixture_manifest["fixture_id"]:
        raise ValueError("run and fixture IDs do not match")

    implementations = run_manifest["implementations"]
    reference = {case["id"]: case for case in implementations["reference"]["cases"]}
    deepinv = {case["id"]: case for case in implementations["deepinv"]["cases"]}
    metrics = {case["id"]: case for case in comparison["cases"]}
    cases = fixture_manifest["cases"]
    task_name = fixture_manifest.get("task", "deblur")
    display_name = task_name.replace("_", " ").title()
    kernel_name = "Motion kernel" if task_name == "motion_deblur" else "Gaussian kernel"
    if task_name == "super_resolution":
        kernel_name = "Bicubic solver kernel"

    figure, axes = plt.subplots(
        len(cases), 5, squeeze=False, figsize=(16, 3.4 * len(cases))
    )
    for row, case in enumerate(cases):
        case_id = case["id"]
        source = load_record(
            fixture_dir,
            case,
            required=("ground_truth", "measurement", "kernel"),
        )
        official = load_record(
            run_dir, reference[case_id], required=("reconstruction",)
        )["reconstruction"]
        deepinv_output = load_record(
            run_dir, deepinv[case_id], required=("reconstruction",)
        )["reconstruction"]
        kernel = source["kernel"].div(source["kernel"].max())
        images = (
            (
                source["ground_truth"],
                f"{case['sources']['image']['name']}\nGround truth",
                False,
            ),
            (source["measurement"], "Noisy measurement", True),
            (
                kernel,
                f"{kernel.shape[-2]}×{kernel.shape[-1]} {kernel_name}\n(normalized display)",
                True,
            ),
            (
                official,
                metric_title("Original repo", metrics[case_id]["reference"]),
                False,
            ),
            (
                deepinv_output,
                metric_title("DeepInv", metrics[case_id]["deepinv"]),
                False,
            ),
        )
        for axis, (tensor, title, range_01) in zip(axes[row], images, strict=True):
            image = display_tensor(tensor, range_01=range_01)
            axis.imshow(image, cmap="gray" if image.ndim == 2 else None, vmin=0, vmax=1)
            axis.set_title(title, fontsize=9)
            axis.axis("off")

    mean = comparison["mean"]
    figure.suptitle(
        f"{display_name} | {run_manifest['setting_id']} | {run_manifest['run_id']}\n"
        f"PSNR/SSIM crop border={comparison.get('metric_protocol', {}).get('crop_border', 0)} | LPIPS full image"
    )
    figure.text(
        0.5,
        0.01,
        "Five-image mean | "
        f"Original PSNR {mean['reference']['psnr_db']:.3f}, SSIM {mean['reference']['ssim']:.4f}, "
        f"LPIPS {mean['reference']['lpips']:.4f} | "
        f"DeepInv PSNR {mean['deepinv']['psnr_db']:.3f}, SSIM {mean['deepinv']['ssim']:.4f}, "
        f"LPIPS {mean['deepinv']['lpips']:.4f}",
        ha="center",
        fontsize=10,
    )
    figure.tight_layout(rect=(0, 0.035, 1, 0.97))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)
    print(output)


if __name__ == "__main__":
    main()
