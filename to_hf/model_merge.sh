set -x

step=35

local_dir="/root/output/SearchShortQA-RL/global_step_${step}/actor"  # 切换为对应的文件夹
hf_path="/root/output/SearchShortQA-RL/global_step_${step}/actor/huggingface"

output_path="/root/output/SearchShortQA-RL-ckpt/SearchShortQA-RL-step${step}-ckpt"

python3 legacy_model_merger.py merge \
    --backend=fsdp \
    --local_dir=$local_dir \
    --hf_model_path=$hf_path \
    --target_dir=$output_path