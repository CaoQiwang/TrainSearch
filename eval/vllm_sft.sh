python -m vllm.entrypoints.openai.api_server \
    --model /root/autodl-tmp/Qwen/sft_lora_ckp-300 \
    --served-model-name Qwen3-4B \
    --host 127.0.0.1 \
    --port 8000