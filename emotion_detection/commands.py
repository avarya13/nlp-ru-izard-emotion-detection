#!/usr/bin/env python
from pathlib import Path

import fire
from omegaconf import OmegaConf
from triton_server.build_triton_repo import build_repo

ROOT = Path(__file__).resolve().parents[1]


def load_data_cfg():
    cfg_path = ROOT / "configs" / "data" / "default.yaml"
    return OmegaConf.load(cfg_path)


def _run_command(module_name: str, func_name: str, overrides: list):
    from utils.hydra_utils import compose_cfg

    cfg = compose_cfg("config", overrides)
    module = __import__(module_name, fromlist=[func_name])
    func = getattr(module, func_name)
    return func(cfg)


def download(*args, **kwargs):
    cfg = load_data_cfg()
    from data.download_data import download_data

    return download_data(cfg)


def train(*overrides: str):
    return _run_command("train.train", "run_train", list(overrides))


def eval(*overrides: str):
    return _run_command("eval.evaluate", "run_eval", list(overrides))


def export_onnx(*overrides: str):
    return _run_command("export.export_onnx", "export_onnx", list(overrides))


def export_trt(*overrides: str):
    import subprocess
    from pathlib import Path

    from utils.hydra_utils import compose_cfg
    from utils.model_paths import get_engine_path, get_onnx_model_path

    cfg = compose_cfg("config", list(overrides))
    onnx_base = Path(cfg.paths.onnx_dir)
    model_name = cfg.model.model_name
    timestamp = getattr(cfg.paths, "timestamp", None)

    onnx_path, _ = get_onnx_model_path(onnx_base, model_name, timestamp)
    trt_base = Path(str(onnx_base).replace("onnx_models", "tensorrt_models"))
    engine_path = get_engine_path(onnx_path, trt_base)

    script_path = Path(__file__).parent / "export" / "export_tensorrt.sh"
    subprocess.run([str(script_path), str(onnx_path), str(engine_path)], check=True)
    print(f"TensorRT engine saved to {engine_path}")


def infer_triton(*overrides: str):
    return _run_command("infer.infer_triton", "run_triton_infer", list(overrides))


def infer_ckpt(*overrides: str):
    return _run_command("infer.infer", "run_infer", list(overrides))


def triton_client(*args, **kwargs):
    from infer.triton_client import main as client_main

    return client_main()


def prepare_triton(*overrides: str):
    from utils.hydra_utils import compose_cfg

    cfg = compose_cfg("config", list(overrides))
    build_repo(cfg)


if __name__ == "__main__":
    fire.Fire(
        {
            "download": download,
            "train": train,
            "eval": eval,
            "export": {
                "onnx": export_onnx,
                "trt": export_trt,
            },
            "infer": {
                "ckpt": infer_ckpt,
                "triton": infer_triton,
            },
            "triton-client": triton_client,
            "prepare-triton": prepare_triton,
        }
    )
