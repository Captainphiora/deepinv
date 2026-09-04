"""Visualize inpainting tensor artifacts and annotate PSNR/SSIM."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


DPS_TOOLS = Path(__file__).resolve().parent / "dps"
sys.path.insert(0, str(DPS_TOOLS))
from _common import load_record, read_json  # noqa: E402
from compare import image_metrics  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture-dir", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output", help="default: <run-dir>/visualization.png")
    return parser.parse_args()


def display_tensor(value: torch.Tensor, *, mask: bool = False):
    value = value.detach().float().cpu()
    if value.ndim == 4 and value.shape[0] == 1:
        value = value[0]
    if value.ndim != 3 or value.shape[0] not in (1, 3):
        raise ValueError(f"expected C=1 or 3 image tensor, got {tuple(value.shape)}")
    value = value.clamp(0, 1) if mask else value.clamp(-1, 1).add(1).div(2)
    return value[0].numpy() if value.shape[0] == 1 else value.permute(1, 2, 0).numpy()


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    args = parse_args()
    fixture_dir = Path(args.fixture_dir).resolve()
    run_dir = Path(args.run_dir).resolve()
    output = Path(args.output).resolve() if args.output else run_dir / "visualization.png"
    fixture_manifest = read_json(fixture_dir / "manifest.json")
    run_manifest = read_json(run_dir / "manifest.json")
    if run_manifest["fixture_id"] != fixture_manifest["fixture_id"]:
        raise ValueError("run and fixture IDs do not match")

    implementations = run_manifest["implementations"]
    reference = {case["id"]: case for case in implementations["reference"]["cases"]}
    deepinv = {case["id"]: case for case in implementations["deepinv"]["cases"]}
    fixture = {case["id"]: case for case in fixture_manifest["cases"]}
    case_ids = [case_id for case_id in fixture if case_id in reference and case_id in deepinv]
    if not case_ids or set(reference) != set(deepinv) or set(reference) != set(case_ids):
        raise ValueError("reference, DeepInv, and fixture case sets do not match")

    figure, axes = plt.subplots(
        len(case_ids), 5, squeeze=False, figsize=(16, 3.35 * len(case_ids))
    )
    all_metrics = {"reference": [], "deepinv": []}
    for row, case_id in enumerate(case_ids):
        source = load_record(
            fixture_dir,
            fixture[case_id],
            required=("ground_truth", "measurement", "mask"),
        )
        outputs = {
            "reference": load_record(
                run_dir, reference[case_id], required=("reconstruction",)
            )["reconstruction"],
            "deepinv": load_record(
                run_dir, deepinv[case_id], required=("reconstruction",)
            )["reconstruction"],
        }
        metrics = {
            name: image_metrics(reconstruction, source["ground_truth"])
            for name, reconstruction in outputs.items()
        }
        for name in all_metrics:
            all_metrics[name].append(metrics[name])

        images = (
            (source["ground_truth"], f"Case {case_id}\nGround truth", False),
            (source["measurement"], "Measurement", False),
            (source["mask"], "Mask", True),
            (
                outputs["reference"],
                "Original repo\n"
                f"PSNR {metrics['reference']['psnr_db']:.3f} dB | "
                f"SSIM {metrics['reference']['ssim']:.4f}",
                False,
            ),
            (
                outputs["deepinv"],
                "DeepInv\n"
                f"PSNR {metrics['deepinv']['psnr_db']:.3f} dB | "
                f"SSIM {metrics['deepinv']['ssim']:.4f}",
                False,
            ),
        )
        for axis, (tensor, title, is_mask) in zip(axes[row], images, strict=True):
            image = display_tensor(tensor, mask=is_mask)
            axis.imshow(
                image,
                cmap="gray" if image.ndim == 2 else None,
                vmin=0,
                vmax=1,
            )
            axis.set_title(title, fontsize=10)
            axis.axis("off")
        print(
            f"{case_id}: "
            f"reference PSNR={metrics['reference']['psnr_db']:.6f} "
            f"SSIM={metrics['reference']['ssim']:.6f}; "
            f"DeepInv PSNR={metrics['deepinv']['psnr_db']:.6f} "
            f"SSIM={metrics['deepinv']['ssim']:.6f}"
        )

    means = {
        name: {
            metric: sum(case[metric] for case in cases) / len(cases)
            for metric in ("psnr_db", "ssim")
        }
        for name, cases in all_metrics.items()
    }
    figure.suptitle(
        f"{run_dir.parent.parent.name.upper()} inpainting | {run_manifest['setting_id']} | "
        f"{run_manifest['run_id']}",
        fontsize=13,
    )
    figure.text(
        0.5,
        0.01,
        "Mean | "
        f"Original: PSNR {means['reference']['psnr_db']:.3f} dB, "
        f"SSIM {means['reference']['ssim']:.4f} | "
        f"DeepInv: PSNR {means['deepinv']['psnr_db']:.3f} dB, "
        f"SSIM {means['deepinv']['ssim']:.4f}",
        ha="center",
        fontsize=11,
    )
    figure.tight_layout(rect=(0, 0.035, 1, 0.96))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)
    print(f"Visualization: {output}")


if __name__ == "__main__":
    main()
