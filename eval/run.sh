set -x

export VERIFIER_SERVER=https://api.deepseek.com/v1
export VERIFIER_API_KEY=sk-d03c7a4e986345f08c3705c5803f1867
export VERIFIER_PATH=deepseek-v4-flash

base_url=127.0.0.1:8000
model_name=Qwen3-4B
dataset=test_data.jsonl


python3 get_response.py \
  --dataset $dataset \
  --base_url $base_url \
  --model_name $model_name \
  --concurrent 128

python3 get_eval.py \
  --dataset $dataset \
  --model_name $model_name \
  --concurrent 32

python3 print_acc.py \
  --dataset $dataset \
  --model_name $model_name