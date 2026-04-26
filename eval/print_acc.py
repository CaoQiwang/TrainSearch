import argparse
import re
import json


if __name__ == "__main__":
    # print(res)
    parser = argparse.ArgumentParser(description="agent loop response")
    # 添加参数
    parser.add_argument("--model_name", type=str, default="")
    parser.add_argument("--dataset", type=str, default="test_data.jsonl", help="输入数据集, 需要在dataset目录下")

    args = parser.parse_args()

    total_cnt = 0
    cnt = 0
    with open(f"./eval_result/{args.model_name}.{args.dataset}", "r") as fin:
        for line in fin:
            total_cnt += 1
            data = json.loads(line)
            pred = data["eval_result"]
            if pred == "A":
                cnt += 1
    
    print(f"Acc: {cnt}/{total_cnt} = {cnt/total_cnt}")

        