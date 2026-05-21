#!/bin/bash

ONNX_PATH=$1
ENGINE_PATH=$2

trtexec \
  --onnx=$ONNX_PATH \
  --saveEngine=$ENGINE_PATH
