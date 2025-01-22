MODEL="unsloth/Llama-3.3-70B-Instruct-bnb-4bit"
MAX_MODEL_LEN=128
LOAD_FORMAT="bitsandbytes"
QUANTIZATION="bitsandbytes"

vllm serve "$MODEL" \
  --port 41401 \
  --max-model-len "$MAX_MODEL_LEN" \
  --load-format "$LOAD_FORMAT" \
  --quantization "$QUANTIZATION" \
  --disable-sliding-window \
  --trust-remote-code
# --no-enable-prefix-caching \
