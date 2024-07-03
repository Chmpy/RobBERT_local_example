#!/bin/bash

# Array of models to convert
models=(
  "pdelobelle/robbert-v2-dutch-base"
  "DTAI-KULeuven/robbert-2022-dutch-base"
  "DTAI-KULeuven/robbert-2023-dutch-base"
  "DTAI-KULeuven/robbert-2023-dutch-large"
)

# Function to convert a model
convert_model() {
  local model=$1
  local model_name=$(basename $model)
  echo "Converting $model"

  optimum-cli export tflite \
    -m $model \
    --task fill-mask \
    --sequence_length 128 \
    ${model_name}_tflite

  echo "Conversion complete for $model"
  echo "------------------------"
}

# Main execution
echo "Starting model conversions..."
for model in "${models[@]}"; do
  convert_model $model
done

echo "All conversions completed!"