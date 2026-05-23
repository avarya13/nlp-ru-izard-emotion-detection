#!/bin/bash
TEXT="$1"
ENGINE_PATH="$2"
ROOT_DIR=$(git rev-parse --show-toplevel)

REL_ENGINE_PATH=$(realpath --relative-to="${ROOT_DIR}" "${ENGINE_PATH}")

docker run --rm --gpus all \
  -v ${ROOT_DIR}:/workspace \
  nvcr.io/nvidia/tensorrt:24.12-py3 \
  bash -c "pip install -q hydra-core omegaconf transformers numpy pycuda torch sentencepiece && \
           python /workspace/emotion_detection/infer/infer_tensorrt.py \
             model=rubert_tiny2 \
             '+text=\"$TEXT\"' \
             paths.tensorrt_engine=\"/workspace/${REL_ENGINE_PATH}\" \
             data.max_length=128"
