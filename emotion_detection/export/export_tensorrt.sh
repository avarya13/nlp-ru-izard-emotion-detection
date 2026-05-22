#!/bin/bash
ONNX_PATH=$1
ENGINE_PATH=$2

ROOT_DIR=$(git rev-parse --show-toplevel)

mkdir -p $(dirname ${ROOT_DIR}/${ENGINE_PATH})

docker run --rm --gpus all \
  -v ${ROOT_DIR}:/workspace \
  nvcr.io/nvidia/tensorrt:24.12-py3 \
  trtexec \
    --onnx=/workspace/${ONNX_PATH} \
    --saveEngine=/workspace/${ENGINE_PATH} \
    --verbose
