from pathlib import Path
from typing import Optional, Tuple


def find_latest_timestamped_dir(base_dir: Path, model_name: str) -> Tuple[Path, str]:
    model_dir = base_dir / model_name
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    subdirs = [
        d
        for d in model_dir.iterdir()
        if d.is_dir() and len(d.name) == 13 and d.name[8] == "-"
    ]
    if not subdirs:
        raise ValueError(f"No timestamped directories in {model_dir}")
    latest = max(subdirs, key=lambda d: d.name)
    return latest, latest.name


def get_onnx_model_path(
    onnx_base: Path, model_name: str, timestamp: Optional[str] = None
) -> Tuple[Path, str]:
    model_name = model_name.replace("/", "-")
    if timestamp:
        onnx_path = onnx_base / model_name / timestamp / "model.onnx"
    else:
        latest_dir, timestamp = find_latest_timestamped_dir(onnx_base, model_name)
        onnx_path = latest_dir / "model.onnx"
    return onnx_path, timestamp


def get_engine_path(onnx_path: Path, trt_base: Path) -> Path:
    rel_path = onnx_path.relative_to(onnx_path.parents[2])
    trt_path = trt_base / rel_path.with_suffix(".engine")
    trt_path.parent.mkdir(parents=True, exist_ok=True)
    return trt_path
