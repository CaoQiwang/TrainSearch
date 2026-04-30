import argparse
import concurrent.futures
import json
import re
from pathlib import Path

import numpy as np
from tqdm import tqdm

from prompts import system_prompt, user_prompt
from tool_utils.tool_parser import CustomToolParser


SCRIPT_DIR = Path(__file__).resolve().parent


def get_length(text):
    return len(text.split())


def truncate_at_call_tool(text):
    if not text:
        return text

    tag = "</google_search>"
    index = text.find(tag)
    if index != -1:
        return text[: index + len(tag)]
    return text


def extract_answer(text):
    match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()


class TrajectoryRunner:
    def __init__(self, args):
        self.args = args
        self.tool_parser = CustomToolParser()
        from tool_utils.apis import request_model
        from tool_utils.tools import search

        self.request_model = request_model
        self.search = search

    def build_messages(self, data):
        query = user_prompt.format(query=data["query"])
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ]

    def run_one(self, data):
        messages = self.build_messages(data)
        trajectory = []
        assistant_turns = 0
        user_turns = 0
        total_response_length = 0

        while True:
            if assistant_turns >= self.args.max_assistant_turns:
                break
            if user_turns > self.args.max_user_turns:
                break
            if total_response_length >= self.args.max_response_length:
                break

            output = self.request_model(self.args.base_url, self.args.model_name, messages)
            output = truncate_at_call_tool(output)
            assistant_turns += 1
            total_response_length += get_length(output)
            messages.append({"role": "assistant", "content": output})

            _, tool_calls = self.tool_parser.extract_tool_calls(output)
            turn_record = {
                "turn": assistant_turns,
                "assistant": output,
                "tool_calls": tool_calls,
                "tool_responses": [],
            }

            if not tool_calls:
                trajectory.append(turn_record)
                break

            for tool_call in tool_calls:
                tool_response = self.search(
                    tool_call.get("arguments", {}).get("query_list", []),
                    top_k=self.args.top_k,
                )
                turn_record["tool_responses"].append(
                    {
                        "tool_call": tool_call,
                        "tool_response": tool_response,
                    }
                )
                total_response_length += get_length(tool_response)
                messages.append({"role": "tool", "content": tool_response})

                if total_response_length >= self.args.max_response_length:
                    break

            trajectory.append(turn_record)
            user_turns += 1

        full_response = "\n".join(
            message["content"] for message in messages[2:] if message.get("content")
        )
        return {
            **data,
            "response": extract_answer(full_response),
            "full_response": full_response,
            "trajectory": trajectory,
        }


def read_jsonl(path):
    data = []
    with path.open("r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def main():
    parser = argparse.ArgumentParser(description="Run tool-use trajectories and save tool calls/responses.")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--base_url", type=str, default="")
    parser.add_argument("--model_name", type=str, default="")
    parser.add_argument("--dataset", type=str, default="test_data_10.jsonl")
    parser.add_argument("--dataset_dir", type=Path, default=SCRIPT_DIR / "dataset")
    parser.add_argument("--output_dir", type=Path, default=SCRIPT_DIR / "trajectory_output")
    parser.add_argument("--max_response_length", type=int, default=np.inf)
    parser.add_argument("--max_assistant_turns", type=int, default=10)
    parser.add_argument("--max_user_turns", type=int, default=10)
    parser.add_argument("--concurrent", type=int, default=2)
    args = parser.parse_args()

    dataset_path = args.dataset_dir / args.dataset
    output_path = args.output_dir / f"{args.model_name}.{args.dataset}"

    args.output_dir.mkdir(parents=True, exist_ok=True)
    datas = read_jsonl(dataset_path)
    runner = TrajectoryRunner(args)

    print(f"Start trajectory generation, data={len(datas)}, concurrent={args.concurrent}")
    with output_path.open("w", encoding="utf-8") as fout:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrent) as executor:
            futures = [executor.submit(runner.run_one, data) for data in datas]
            for future in tqdm(futures):
                result = future.result()
                fout.write(json.dumps(result, ensure_ascii=False) + "\n")
                fout.flush()

    print(f"Wrote trajectories to {output_path}")


if __name__ == "__main__":
    main()
