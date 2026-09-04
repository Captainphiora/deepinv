"""Compare reference and DeepInv DPS tensor artifacts without image round trips."""

from __future__ import annotations

import argparse

import torch

from _common import (
    artifact_root,
    file_sha256,
    fixture_dir,
    load_setting,
    load_record,
    read_json,
    run_dir,
    select_cases,
    utc_now,
    write_json,
)


def differences(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    if actual.shape != expected.shape:
        raise ValueError(f"shape mismatch: {actual.shape} != {expected.shape}")
    delta = actual.to(torch.float64) - expected.to(torch.float64)
    return {
        "mae": delta.abs().mean().item(),
        "rmse": torch.sqrt(torch.mean(delta**2)).item(),
        "max_abs": delta.abs().max().item(),
        "relative_l2": (
            torch.linalg.vector_norm(delta)
            / torch.linalg.vector_norm(expected.to(torch.float64)).clamp_min(1e-12)
        ).item(),
    }


def image_metrics(
    reconstruction: torch.Tensor, target: torch.Tensor, lpips_metric=None
) -> dict[str, float]:
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity

    reconstruction_raw = reconstruction.detach().float().clamp(-1, 1)
    target_raw = target.detach().float().clamp(-1, 1)
    reconstruction_np = (
        reconstruction_raw.add(1).div(2).squeeze(0).permute(1, 2, 0).cpu().numpy()
    )
    target_np = target_raw.add(1).div(2).squeeze(0).permute(1, 2, 0).cpu().numpy()
    result = {
        "psnr_db": float(
            peak_signal_noise_ratio(target_np, reconstruction_np, data_range=1.0)
        ),
        "ssim": float(
            structural_similarity(
                target_np, reconstruction_np, data_range=1.0, channel_axis=-1
            )
        ),
    }
    if lpips_metric is not None:
        metric_device = next(lpips_metric.parameters()).device
        result["lpips"] = lpips_metric(
            reconstruction_raw.to(metric_device), target_raw.to(metric_device)
        ).mean().item()
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--setting", required=True, help="setting JSON path or name")
    parser.add_argument("--fixture-id", default="ffhq256_inpainting_v1")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--case", action="append", help="case id; repeat as needed")
    parser.add_argument("--artifact-root")
    parser.add_argument("--metric-device", default="cpu")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setting_file, setting = load_setting(args.setting)
    root = artifact_root(args.artifact_root)
    fixture = fixture_dir(root, args.fixture_id)
    manifest = read_json(fixture / "manifest.json")
    cases = select_cases(manifest, args.case)
    if not cases:
        raise ValueError("fixture contains no selected cases")
    output_dir = run_dir(root, setting["id"], args.run_id)
    output_path = output_dir / "comparison.json"
    if args.dry_run:
        print(
            {
                "reference": str(output_dir / "cases/<case>/reference.pt"),
                "deepinv": str(output_dir / "cases/<case>/deepinv.pt"),
                "output": str(output_path),
                "cases": [case["id"] for case in cases],
            }
        )
        return
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite comparison: {output_path}")

    run_manifest = read_json(output_dir / "manifest.json")
    if run_manifest.get("fixture_id") != args.fixture_id:
        raise ValueError("run manifest uses a different fixture")
    if run_manifest.get("setting_id") != setting["id"]:
        raise ValueError("run manifest uses a different setting")
    if run_manifest.get("setting_sha256") != file_sha256(setting_file):
        raise ValueError("setting JSON changed after the run")

    import lpips

    lpips_metric = lpips.LPIPS(net="vgg").to(args.metric_device).eval()
    reference_records = {
        record["id"]: record
        for record in run_manifest["implementations"]["reference"]["cases"]
    }
    deepinv_records = {
        record["id"]: record
        for record in run_manifest["implementations"]["deepinv"]["cases"]
    }

    thresholds = setting["thresholds"]
    results = []
    all_passed = True
    for case in cases:
        case_id = case["id"]
        for implementation, records in (
            ("reference", reference_records),
            ("deepinv", deepinv_records),
        ):
            if records[case_id].get("fixture_tensor_sha256") != case["tensor_sha256"]:
                raise ValueError(
                    f"{implementation} output for {case_id} used a different fixture"
                )
        source = load_record(
            fixture, case, required=("ground_truth", "x_init")
        )
        reference = load_record(
            output_dir,
            reference_records[case_id],
            required=(
                "reconstruction",
                "trajectory",
                "trajectory_steps",
                "timesteps",
                "noise_levels",
            ),
        )
        deepinv = load_record(
            output_dir,
            deepinv_records[case_id],
            required=(
                "reconstruction",
                "trajectory",
                "trajectory_steps",
                "timesteps",
                "noise_levels",
            ),
        )
        if not torch.equal(reference["timesteps"], deepinv["timesteps"]):
            raise ValueError(f"timestep mismatch for case {case_id}")
        if not torch.equal(reference["noise_levels"], deepinv["noise_levels"]):
            raise ValueError(f"noise-level mismatch for case {case_id}")
        if not torch.equal(
            reference["trajectory_steps"], deepinv["trajectory_steps"]
        ):
            raise ValueError(f"trajectory probe mismatch for case {case_id}")

        reference_metrics = image_metrics(
            reference["reconstruction"], source["ground_truth"], lpips_metric
        )
        deepinv_metrics = image_metrics(
            deepinv["reconstruction"], source["ground_truth"], lpips_metric
        )
        final_difference = differences(
            deepinv["reconstruction"], reference["reconstruction"]
        )
        probes = []
        for step in setting["trajectory_probe_steps"]:
            indices = torch.nonzero(
                reference["trajectory_steps"] == step, as_tuple=False
            ).flatten()
            if len(indices) != 1:
                raise IndexError(f"trajectory has no conditioned step {step}")
            trajectory_index = indices.item()
            difference = differences(
                deepinv["trajectory"][trajectory_index],
                reference["trajectory"][trajectory_index],
            )
            passed = (
                difference["mae"] <= thresholds["trajectory_mae"]
                and difference["relative_l2"]
                <= thresholds["trajectory_relative_l2"]
            )
            probes.append({"step": step, **difference, "passed": passed})

        delta_psnr = abs(deepinv_metrics["psnr_db"] - reference_metrics["psnr_db"])
        delta_ssim = abs(deepinv_metrics["ssim"] - reference_metrics["ssim"])
        delta_lpips = abs(
            deepinv_metrics["lpips"] - reference_metrics["lpips"]
        )
        case_passed = (
            all(probe["passed"] for probe in probes)
            and delta_psnr <= thresholds["per_case_delta_psnr_db"]
            and delta_ssim <= thresholds["per_case_delta_ssim"]
            and delta_lpips <= thresholds["per_case_delta_lpips"]
        )
        all_passed &= case_passed
        results.append(
            {
                "id": case_id,
                "reference": reference_metrics,
                "deepinv": deepinv_metrics,
                "delta": {
                    "psnr_db": delta_psnr,
                    "ssim": delta_ssim,
                    "lpips": delta_lpips,
                },
                "final_tensor_difference": final_difference,
                "trajectory": probes,
                "passed": case_passed,
            }
        )

    mean_reference = {
        key: sum(case["reference"][key] for case in results) / len(results)
        for key in ("psnr_db", "ssim", "lpips")
    }
    mean_deepinv = {
        key: sum(case["deepinv"][key] for case in results) / len(results)
        for key in ("psnr_db", "ssim", "lpips")
    }
    mean_delta = {
        key: abs(mean_deepinv[key] - mean_reference[key])
        for key in ("psnr_db", "ssim", "lpips")
    }
    mean_passed = (
        mean_delta["psnr_db"] <= thresholds["mean_delta_psnr_db"]
        and mean_delta["ssim"] <= thresholds["mean_delta_ssim"]
        and mean_delta["lpips"] <= thresholds["mean_delta_lpips"]
    )
    all_passed &= mean_passed
    report = {
        "schema_version": 1,
        "created_at": utc_now(),
        "setting_id": setting["id"],
        "fixture_id": args.fixture_id,
        "run_id": args.run_id,
        "thresholds": thresholds,
        "cases": results,
        "mean": {
            "reference": mean_reference,
            "deepinv": mean_deepinv,
            "delta": mean_delta,
            "passed": mean_passed,
        },
        "passed": all_passed,
    }
    write_json(output_path, report)
    print(f"{'PASS' if all_passed else 'FAIL'}: {output_path}")
    if not all_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
