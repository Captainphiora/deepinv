from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACT_ROOT = REPO_ROOT / "reproduction" / "artifacts"


def read_json(path: str | Path) -> dict:
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: str | Path, value: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    temporary.replace(path)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_dict_sha256(value: dict[str, torch.Tensor]) -> str:
    validate_tensor_dict(value)
    digest = hashlib.sha256()
    for key in sorted(value):
        tensor = value[key].detach().cpu().contiguous()
        header = {
            "key": key,
            "dtype": str(tensor.dtype),
            "shape": list(tensor.shape),
        }
        digest.update(json.dumps(header, sort_keys=True).encode())
        digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def validate_tensor_dict(value: object, required: tuple[str, ...] = ()) -> None:
    if not isinstance(value, dict) or not all(
        isinstance(key, str) and isinstance(tensor, torch.Tensor)
        for key, tensor in value.items()
    ):
        raise TypeError("artifact must be a dict[str, torch.Tensor]")
    missing = sorted(set(required).difference(value))
    if missing:
        raise KeyError(f"artifact is missing tensors: {', '.join(missing)}")


def load_tensors(
    path: str | Path, required: tuple[str, ...] = ()
) -> dict[str, torch.Tensor]:
    value = torch.load(Path(path), map_location="cpu", weights_only=True)
    validate_tensor_dict(value, required)
    return value


def load_record(
    base: str | Path, record: dict, required: tuple[str, ...] = ()
) -> dict[str, torch.Tensor]:
    path = Path(base) / record["path"]
    if file_sha256(path) != record["sha256"]:
        raise ValueError(f"file SHA256 mismatch: {path}")
    value = load_tensors(path, required)
    if tensor_dict_sha256(value) != record["tensor_sha256"]:
        raise ValueError(f"tensor-content SHA256 mismatch: {path}")
    return value


def save_tensors(path: str | Path, value: dict[str, torch.Tensor]) -> dict:
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite artifact: {path}")
    normalized = {
        key: tensor.detach().cpu().contiguous() for key, tensor in value.items()
    }
    validate_tensor_dict(normalized)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(normalized, temporary)
    temporary.replace(path)
    return {
        "path": path.name,
        "sha256": file_sha256(path),
        "tensor_sha256": tensor_dict_sha256(normalized),
        "tensors": {
            key: {"dtype": str(tensor.dtype), "shape": list(tensor.shape)}
            for key, tensor in sorted(normalized.items())
        },
    }


def artifact_root(value: str | None) -> Path:
    return Path(
        value or os.environ.get("DEEPINV_REPRO_ROOT") or DEFAULT_ARTIFACT_ROOT
    )


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def configure_determinism(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def environment(device: str) -> dict:
    cudnn = torch.backends.cudnn.version()
    requested_device = torch.device(device)
    gpu = None
    if requested_device.type == "cuda" and torch.cuda.is_available():
        index = (
            torch.cuda.current_device()
            if requested_device.index is None
            else requested_device.index
        )
        properties = torch.cuda.get_device_properties(index)
        uuid = getattr(properties, "uuid", None)
        gpu = {
            "name": properties.name,
            "compute_capability": [properties.major, properties.minor],
            "uuid": str(uuid) if uuid is not None else None,
        }
    return {
        "python": platform.python_version(),
        "python_executable": str(Path(sys.executable).resolve()),
        "python_prefix": str(Path(sys.prefix).resolve()),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "cuda": torch.version.cuda,
        "cudnn": str(cudnn) if cudnn is not None else None,
        "device": device,
        "gpu": gpu,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
    }


def git_revision(repo: str | Path) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def require_clean_repo(repo: str | Path, paths: tuple[str, ...]) -> None:
    status = subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "status",
            "--porcelain",
            "--untracked-files=normal",
            "--",
            *paths,
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if status:
        raise RuntimeError(
            "certification requires committed DeepInv code; dirty paths:\n" + status
        )


def command_line() -> list[str]:
    return [sys.executable, *sys.argv]


def setting_path(value: str) -> Path:
    path = Path(value)
    if path.suffix != ".json":
        path = path.with_suffix(".json")
    candidates = [path]
    if not path.is_absolute():
        candidates.extend(
            [
                REPO_ROOT / path,
                Path(__file__).resolve().parent / "settings" / path,
                *sorted((REPO_ROOT / "reproduction").glob(f"*/settings/{path.name}")),
            ]
        )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[-1]


def load_setting(value: str) -> tuple[Path, dict]:
    path = setting_path(value)
    setting = read_json(path)
    required = {"schema_version", "id", "algorithm", "sampler", "randomness"}
    missing = sorted(required.difference(setting))
    if missing:
        raise KeyError(f"setting is missing keys: {', '.join(missing)}")
    probes = setting.get("trajectory_probe_steps", [])
    sampling_steps = setting["sampler"]["sampling_steps"]
    if (
        not all(isinstance(step, int) for step in probes)
        or probes != sorted(set(probes))
        or any(step < 0 or step >= sampling_steps for step in probes)
    ):
        raise ValueError(
            "trajectory_probe_steps must be unique, increasing reverse-loop "
            f"indices in [0, {sampling_steps - 1}]"
        )
    return path, setting


def fixture_dir(root: Path, fixture_id: str) -> Path:
    return root / "fixtures" / fixture_id


def run_dir(
    root: Path, setting_id: str, run_id: str, algorithm: str = "dps"
) -> Path:
    return root / "runs" / algorithm / setting_id / run_id


def requires_transition_noise(setting: dict) -> bool:
    sampler = setting["sampler"]
    return sampler["name"] == "ddpm" or (
        sampler["name"] == "ddim" and float(sampler["eta"]) > 0
    )


def fixed_randn_like(noise: torch.Tensor):
    """Return a one-shot ``torch.randn_like`` replacement for a fixed tape."""
    calls = 0

    def draw(target: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        nonlocal calls
        calls += 1
        if calls != 1:
            raise RuntimeError("reference p_sample requested more than one noise tensor")
        if args or kwargs:
            raise RuntimeError("unexpected torch.randn_like arguments")
        if noise.shape != target.shape:
            raise ValueError(f"transition-noise shape {noise.shape} != {target.shape}")
        return noise.to(device=target.device, dtype=target.dtype)

    return draw, lambda: calls


def select_cases(manifest: dict, selected: list[str] | None) -> list[dict]:
    cases = manifest.get("cases", [])
    by_id = {case["id"]: case for case in cases}
    if not selected:
        return cases
    missing = sorted(set(selected).difference(by_id))
    if missing:
        raise KeyError(f"fixture has no cases: {', '.join(missing)}")
    return [by_id[case_id] for case_id in selected]


def update_run_manifest(
    path: str | Path,
    *,
    setting_id: str,
    setting_sha256: str,
    fixture_id: str,
    fixture_manifest_sha256: str,
    run_id: str,
    implementation: str,
    record: dict,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = path.with_suffix(path.suffix + ".lock")
    deadline = time.monotonic() + 60
    while True:
        try:
            lock.mkdir()
            break
        except FileExistsError:
            if time.monotonic() >= deadline:
                raise TimeoutError(f"timed out waiting for manifest lock: {lock}")
            time.sleep(0.05)
    try:
        manifest = (
            read_json(path)
            if path.exists()
            else {
                "schema_version": 1,
                "setting_id": setting_id,
                "setting_sha256": setting_sha256,
                "fixture_id": fixture_id,
                "fixture_manifest_sha256": fixture_manifest_sha256,
                "run_id": run_id,
                "implementations": {},
            }
        )
        for key, expected in {
            "setting_id": setting_id,
            "setting_sha256": setting_sha256,
            "fixture_id": fixture_id,
            "fixture_manifest_sha256": fixture_manifest_sha256,
            "run_id": run_id,
        }.items():
            if manifest.get(key) != expected:
                raise ValueError(f"run manifest {key} mismatch")
        if implementation in manifest["implementations"]:
            raise FileExistsError(
                f"run manifest already contains implementation {implementation}"
            )
        manifest["updated_at"] = utc_now()
        manifest["implementations"][implementation] = record
        write_json(path, manifest)
    finally:
        lock.rmdir()
