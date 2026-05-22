#!/usr/bin/env python
from pathlib import Path

import fire
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]


def load_data_cfg():
    cfg_path = ROOT / "configs" / "data" / "default.yaml"
    return OmegaConf.load(cfg_path)


def download(*args, **kwargs):
    cfg = load_data_cfg()
    from data.download_data import download_data

    return download_data(cfg)


def train(*args, **kwargs):
    from train.train import main as train_main

    return train_main()


def evaluate(*args, **kwargs):
    from eval.evaluate import main as eval_main

    return eval_main()


def export_onnx(*args, **kwargs):
    from export.export_onnx import main as onnx_main

    return onnx_main()


def export_tensorrt(*args, **kwargs):
    import subprocess
    from pathlib import Path

    script_path = Path(__file__).parent / "export" / "export_tensorrt.sh"
    return subprocess.run([str(script_path), *args], check=True)


def predict(*args, **kwargs):
    from infer.predict import main as predict_main

    return predict_main()


def predict_tensorrt(*args, **kwargs):
    from infer.predict_tensorrt import main as trt_main

    return trt_main()


def triton(*args, **kwargs):
    from infer.triton_infer import main as triton_main

    return triton_main()


def triton_client(*args, **kwargs):
    from infer.triton_client import main as client_main

    return client_main()


if __name__ == "__main__":
    fire.Fire(
        {
            "download": download,
            "train": train,
            "eval": evaluate,
            "export": {
                "onnx": export_onnx,
                "tensorrt": export_tensorrt,
            },
            "predict": {
                "local": predict,
                "tensorrt": predict_tensorrt,
                "triton": triton,
            },
            "triton-client": triton_client,
        }
    )
