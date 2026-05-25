import argparse
import re
import json
from prompts import GRADER_en_TEMPLATE
import concurrent.futures
from tqdm import tqdm
import openai
import time
import os


VERIFIER_SERVER = os.getenv("VERIFIER_SERVER", "http://127.0.0.1:8000/v1")
VERIFIER_API_KEY = os.getenv("VERIFIER_API_KEY", "None")
VERIFIER_PATH = os.getenv("VERIFIER_PATH", "Qwen2.5-14B-Instruct")

def request_model(prompt):
    client = openai.Client(base_url=VERIFIER_SERVER, api_key=VERIFIER_API_KEY)
    messages = [{"role": "user", "content": prompt}]
    for _ in range(5):
        try:
            response = client.chat.completions.create(
                model=VERIFIER_PATH,
                messages=messages,
                temperature=0.1,
                top_p=1.0,
                max_tokens=128,
                extra_body={"thinking": {"type": "disabled"}},
            )
            return response.choices[0].message.content.strip()
        except:
            time.sleep(1)
            continue
    
    return ""


def extract_answer(response: str) -> str:
    """
    从响应中提取回复内容，去除<cite>标签及之后的部分
    
    Args:
        response: 包含可能带有<cite>标签的响应字符串
        
    Returns:
        提取后的纯回复内容
    """
    # 使用正则表达式找到第一个<cite>标签的位置
    match = re.search(r'<cite[^>]*>', response)
    
    if match:
        # 提取<cite>标签之前的内容
        return response[:match.start()].rstrip()
    else:
        # 如果没有<cite>标签，返回原响应
        return response.strip()


if __name__ == "__main__":
    # print(res)
    parser = argparse.ArgumentParser(description="agent loop response")
    # 添加参数
    parser.add_argument("--model_name", type=str, default="")
    parser.add_argument("--dataset", type=str, default="sft_train_data.jsonl", help="输入数据集, 需要在dataset目录下")
    parser.add_argument("--concurrent", type=int, default=256, help="")

    args = parser.parse_args()

    # 第一次抓取
    total_cnt = 0
    cnt = 0
    with open(f"./output/{args.model_name}.{args.dataset}", "r") as fin:
        with open(f"./eval_result/{args.model_name}.{args.dataset}", "w") as fout:
            datas = []
            for line in fin:
                data = json.loads(line)
                datas.append(data)

            with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrent) as executor:
                futures = []
                for data in datas:
                    response = data["response"]
                    if args.model_name == "Qwen3-8B":
                        answer_match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
                        if not answer_match:
                            content = response
                        else:
                            content = answer_match.group(1).strip()
                    else:
                        content = response
                    # 提取答案
                    pred = extract_answer(content)
                    prompt = GRADER_en_TEMPLATE.format(question=data["query"], target=data["answer"], predicted_answer=pred)
                    futures.append(executor.submit(request_model, prompt))
                    
                pairs = [(data, future) for data, future in zip(datas, futures)]
                print(f"开始测评回答, 数量{len(datas)}, 并发数{args.concurrent}")
                for data, future in tqdm(pairs):
                    res = future.result()
                    data["eval_result"] = res
                    fout.write(json.dumps(data, ensure_ascii=False) + "\n")

                

        