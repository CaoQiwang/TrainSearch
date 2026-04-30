python -m vllm.entrypoints.openai.api_server \
    --model /workspace/Qwen/Qwen3-4B \
    --served-model-name Qwen3-4B \
    --host 127.0.0.1 \
    --port 8000