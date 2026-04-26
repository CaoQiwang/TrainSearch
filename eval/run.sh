set -x

base_url=127.0.0.1:8000
model_name=SearchShortQA-RL-step20-ckpt
dataset=test_data.jsonl


python3 get_response.py \
  --dataset $dataset \
  --base_url $base_url \
  --model_name $model_name \
  --concurrent 128

python3 get_eval.py \
  --dataset $dataset \
  --model_name $model_name \
  --concurrent 512

python3 print_acc.py \
  --dataset $dataset \
  --model_name $model_name