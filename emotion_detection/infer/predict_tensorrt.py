from pathlib import Path

import hydra
import numpy as np
import pycuda.driver as cuda
import tensorrt as trt
from omegaconf import DictConfig, OmegaConf
from transformers import AutoTokenizer

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


def load_engine(engine_path: Path):
    with open(engine_path, "rb") as f:
        runtime = trt.Runtime(TRT_LOGGER)
        return runtime.deserialize_cuda_engine(f.read())


def infer(engine, input_ids: np.ndarray, attention_mask: np.ndarray):
    context = engine.create_execution_context()
    context.set_input_shape("input_ids", input_ids.shape)
    context.set_input_shape("attention_mask", attention_mask.shape)

    output_shape = context.get_tensor_shape("logits")
    output_shape_tuple = tuple(output_shape)
    h_output = cuda.pagelocked_empty(output_shape_tuple, dtype=np.float32)

    d_input_ids = cuda.mem_alloc(input_ids.nbytes)
    d_attention_mask = cuda.mem_alloc(attention_mask.nbytes)
    d_output = cuda.mem_alloc(h_output.nbytes)

    cuda.memcpy_htod(d_input_ids, input_ids)
    cuda.memcpy_htod(d_attention_mask, attention_mask)
    context.execute_v2([int(d_input_ids), int(d_attention_mask), int(d_output)])
    cuda.memcpy_dtoh(h_output, d_output)
    return h_output


@hydra.main(version_base="1.3", config_path="../../configs", config_name="config")
def main(cfg: DictConfig):
    if not hasattr(cfg, "text") or cfg.text is None:
        raise ValueError("Please provide text via +text='Your text here'")
    text = cfg.text

    script_dir = Path(__file__).parent
    labels_path = script_dir / "../../configs/data/labels.yaml"
    cfg_labels = OmegaConf.load(labels_path.resolve())
    labels = cfg_labels.labels

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.model_name)

    engine_path = Path(cfg.paths.tensorrt_engine)
    if not engine_path.exists():
        raise FileNotFoundError(f"TensorRT engine not found: {engine_path}")
    engine = load_engine(engine_path)

    inputs = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=cfg.data.max_length,
        return_tensors="np",
    )
    input_ids = inputs["input_ids"].astype(np.int64)
    attention_mask = inputs["attention_mask"].astype(np.int64)

    logits = infer(engine, input_ids, attention_mask)[0]
    probs = 1 / (1 + np.exp(-logits))

    for label, prob in zip(labels, probs):
        print(f"{label.title().ljust(12)}: {prob:.4f}")


if __name__ == "__main__":
    main()
