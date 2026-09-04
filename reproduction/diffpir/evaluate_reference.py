"""Record metrics for an official DiffPIR run before running DeepInv."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

DPS_DIR = Path(__file__).resolve().parents[1] / "dps"
sys.path.insert(0, str(DPS_DIR))

from _common import (  # noqa: E402
    artifact_root,
    file_sha256,
    fixture_dir,
    load_record,
    load_setting,
    read_json,
    run_dir,
    utc_now,
    write_json,
)
from compare import image_metrics  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--setting", required=True)
    parser.add_argument("--fixture-id", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--artifact-root")
    parser.add_argument("--metric-device", default="cpu")
    args = parser.parse_args()

    setting_file, setting = load_setting(args.setting)
    root = artifact_root(args.artifact_root)
    fixture = fixture_dir(root, args.fixture_id)
    fixture_manifest = read_json(fixture / "manifest.json")
    output_dir = run_dir(root, setting["id"], args.run_id, "diffpir")
    run_manifest = read_json(output_dir / "manifest.json")
    if run_manifest["setting_sha256"] != file_sha256(setting_file):
        raise ValueError("setting JSON changed after the reference run")
    if run_manifest["fixture_manifest_sha256"] != file_sha256(
        fixture / "manifest.json"
    ):
        raise ValueError("fixture manifest changed after the reference run")

    import lpips

    metric = lpips.LPIPS(net="vgg").to(args.metric_device).eval()
    records = {
        record["id"]: record
        for record in run_manifest["implementations"]["reference"]["cases"]
    }
    cases = []
    for source_record in fixture_manifest["cases"]:
        case_id = source_record["id"]
        source = load_record(fixture, source_record, required=("ground_truth",))
        output = load_record(output_dir, records[case_id], required=("reconstruction",))
        metrics = image_metrics(
            output["reconstruction"], source["ground_truth"], metric
        )
        cases.append({"id": case_id, **metrics})
        print(f"{setting['id']} {case_id}: {metrics}")

    mean = {
        key: sum(case[key] for case in cases) / len(cases)
        for key in ("psnr_db", "ssim", "lpips")
    }
    report = {
        "schema_version": 1,
        "created_at": utc_now(),
        "setting_id": setting["id"],
        "fixture_id": args.fixture_id,
        "run_id": args.run_id,
        "implementation": "reference",
        "images": len(cases),
        "cases": cases,
        "mean": mean,
        "paper_reference": setting.get("paper_reference"),
        "paper_metrics_comparable": False,
        "paper_metrics_note": (
            "This fixed five-image demo run is not the paper's 100-image aggregate; "
            "FID is intentionally not computed."
        ),
    }
    output_path = output_dir / "reference_metrics.json"
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite metrics: {output_path}")
    write_json(output_path, report)
    print(f"Reference mean: {mean}\n{output_path}")


if __name__ == "__main__":
    main()
