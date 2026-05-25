
set -e
set -x

base_url=127.0.0.1:8000
model_name=Qwen3-4B
dataset=test_data_100.jsonl



python3 get_trajectory.py \
  --dataset $dataset \
  --base_url "$base_url" \
  --model_name "$model_name" \
  --concurrent 36