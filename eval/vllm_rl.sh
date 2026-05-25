python -m vllm.entrypoints.openai.api_server \
    --model /root/autodl-tmp/output/TrainSearch-RL/global_step_25/merged \
    --served-model-name Qwen3-4B \
    --host 127.0.0.1 \
    --port 8000