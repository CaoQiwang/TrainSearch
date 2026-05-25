# 合并 FSDP checkpoint 为 HuggingFace 格式
python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir /root/autodl-tmp/output/TrainSearch-RL/global_step_15/actor \
    --target_dir /root/autodl-tmp/output/TrainSearch-RL/global_step_15/merged 