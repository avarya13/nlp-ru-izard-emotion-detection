#!/bin/bash
ONNX_PATH=$1
ENGINE_PATH=$2

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
ROOT_DIR=$(cd "$SCRIPT_DIR/.." && pwd)

mkdir -p $(dirname ${ROOT_DIR}/${ENGINE_PATH})

docker run --rm --gpus all \
  -v ${ROOT_DIR}:/workspace \
  nvcr.io/nvidia/tensorrt:24.12-py3 \
  trtexec \
    --onnx=/workspace/${ONNX_PATH} \
    --saveEngine=/workspace/${ENGINE_PATH} \
    --minShapes=input_ids:1x128,attention_mask:1x128 \
    --optShapes=input_ids:1x128,attention_mask:1x128 \
    --maxShapes=input_ids:1x128,attention_mask:1x128 \
    --verbose
