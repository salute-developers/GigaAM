#!/bin/bash

# Script to convert ONNX models to TensorRT (TRT) format
# Usage: bash run_convert_trt.sh [ctc|rnnt]
# Requires: TensorRT, trtexec in PATH

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if trtexec is available
if ! command -v trtexec &> /dev/null; then
    echo -e "${RED}Error: trtexec not found in PATH${NC}"
    echo "Please install TensorRT and ensure trtexec is in your PATH"
    exit 1
fi

# Parse model type argument
MODEL_TYPE=${1:-""}

if [ -z "$MODEL_TYPE" ]; then
    echo -e "${RED}Error: Model type not specified${NC}"
    echo "Usage: bash run_convert_trt.sh [ctc|rnnt]"
    exit 1
fi

if [ "$MODEL_TYPE" != "ctc" ] && [ "$MODEL_TYPE" != "rnnt" ]; then
    echo -e "${RED}Error: Invalid model type: $MODEL_TYPE${NC}"
    echo "Usage: bash run_convert_trt.sh [ctc|rnnt]"
    exit 1
fi

# Convert CTC encoder
if [ "$MODEL_TYPE" = "ctc" ]; then
    echo -e "${GREEN}=== Converting CTC Encoder ===${NC}"
    ctc_onnx="repos/ctc_encoder_onnx/1/model.onnx"
    ctc_trt_dir="repos/ctc_encoder_trt/1"
    ctc_trt_path="$ctc_trt_dir/model.plan"
    
    if [ -f "$ctc_onnx" ]; then
        mkdir -p "$ctc_trt_dir"
        trtexec \
            --onnx="$ctc_onnx" \
            --saveEngine="$ctc_trt_path" \
            --fp16 \
            --memPoolSize=workspace:8192 \
            --minShapes=features:1x64x1,feature_lengths:1 \
            --optShapes=features:8x64x1000,feature_lengths:8 \
            --maxShapes=features:32x64x5000,feature_lengths:32 \
            --verbose \
            --noTF32
        
        if [ $? -eq 0 ]; then
            echo -e "${GREEN}✓ CTC encoder converted${NC}"
        else
            echo -e "${RED}✗ Failed to convert CTC encoder${NC}"
            exit 1
        fi
    else
        echo -e "${RED}Error: CTC encoder ONNX not found: $ctc_onnx${NC}"
        echo "  Run run_convert_onnx.py v3_ctc first to generate ONNX model"
        exit 1
    fi
fi

# Convert RNNT encoder
if [ "$MODEL_TYPE" = "rnnt" ]; then
    echo -e "${GREEN}=== Converting RNNT Encoder ===${NC}"
    rnnt_onnx="repos/gigaam_encoder_onnx/1/model.onnx"
    rnnt_trt_dir="repos/gigaam_encoder_trt/1"
    rnnt_trt_path="$rnnt_trt_dir/model.plan"
    
    if [ -f "$rnnt_onnx" ]; then
        mkdir -p "$rnnt_trt_dir"
        trtexec \
            --onnx="$rnnt_onnx" \
            --saveEngine="$rnnt_trt_path" \
            --fp16 \
            --memPoolSize=workspace:8192 \
            --minShapes=audio_signal:1x64x1,length:1 \
            --optShapes=audio_signal:8x64x1000,length:8 \
            --maxShapes=audio_signal:32x64x5000,length:32 \
            --verbose \
            --noTF32
        
        if [ $? -eq 0 ]; then
            echo -e "${GREEN}✓ RNNT encoder converted${NC}"
        else
            echo -e "${RED}✗ Failed to convert RNNT encoder${NC}"
            exit 1
        fi
    else
        echo -e "${RED}Error: RNNT encoder ONNX not found: $rnnt_onnx${NC}"
        echo "  Run run_convert_onnx.py v3_e2e_rnnt first to generate ONNX model"
        exit 1
    fi
fi

echo -e "${GREEN}=== Conversion Complete ===${NC}"
