set -x

export VERIFIER_SERVER=https://api.deepseek.com/v1
export VERIFIER_API_KEY=sk-d4faa9a59fb944fa86c5e879d34be6a8
export VERIFIER_PATH=deepseek-v4-flash

base_url=127.0.0.1:8000
model_name=Qwen3-4B
dataset=test_data.jsonl


python3 get_response.py \
  --dataset $dataset \
  --base_url $base_url \
  --model_name $model_name \
  --concurrent 32

python3 get_eval.py \
  --dataset $dataset \
  --model_name $model_name \
  --concurrent 32

python3 print_acc.py \
  --dataset $dataset \
  --model_name $model_name