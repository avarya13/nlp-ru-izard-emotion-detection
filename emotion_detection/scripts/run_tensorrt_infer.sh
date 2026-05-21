#!/bin/bash
TEXT="$1"
ENGINE_PATH="$2"
ROOT_DIR=$(git rev-parse --show-toplevel)

docker run --rm --gpus all \
  -v ${ROOT_DIR}:/workspace \
  nvcr.io/nvidia/tensorrt:24.12-py3 \
  bash -c "pip install -q hydra-core omegaconf transformers numpy pycuda && \
           python /workspace/emotion_detection/scripts/predict_tensorrt.py \
             model=rubert_tiny2 \
             '+text=\"$TEXT\"' \
             paths.tensorrt_engine=\"$ENGINE_PATH\""
