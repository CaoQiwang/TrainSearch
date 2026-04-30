import argparse
import concurrent.futures
import json
import logging
import os
import re
from enum import Enum

from tqdm import tqdm

from prompts import system_prompt, user_prompt
from tool_utils.apis import request_model
from tool_utils.tool_parser import CustomToolParser
from tool_utils.tools import search


logger = logging.getLogger(__file__)


class AgentState(Enum):
    GENERATING = "generating"
    PROCESSING_TOOLS = "processing_tools"
    TERMINATED = "terminated"


class TraceAgentData:
    def __init__(self, messages):
        self.messages = messages
        self.user_turns = 0
        self.assistant_turns = 0
        self.total_response_length = 0
        self.tool_calls = []
        self.init_messages_length = len(messages)
        self.trajectory = []


def truncate_at_call_tool(text):
    if not text:
        return text

    tag = "</google_search>"
    index = text.find(tag)
    if index != -1:
        return text[: index + len(tag)]

    return text


def get_query_and_messages(data):
    query = user_prompt.format(query=data["query"])
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query},
    ]
    return query, messages


def get_qwen_response(messages):
    text = ""
    for message in messages:
        text += message["content"]
        text += "\n"
    return text


class TracingToolAgentLoop:
    def __init__(self, args):
        self.args = args
        self.tool_parser = CustomToolParser()

    def get_length(self, text):
        return len(text.split()) if text else 0

    def _handle_generating_state(self, agent_data):
        output = request_model(self.args.base_url, self.args.model_name, agent_data.messages)

        truncate_flag = True
        if self.args.max_assistant_turns and agent_data.assistant_turns + 1 >= self.args.max_assistant_turns:
            _, tool_calls = self.tool_parser.extract_tool_calls(output)
            if len(tool_calls) > 0:
                output += "\n<answer>Cannot determine an answer based on the available information.</answer>"
                truncate_flag = False

        if truncate_flag:
            output = truncate_at_call_tool(output)

        if agent_data.total_response_length + self.get_length(output) >= self.args.max_response_length:
            return AgentState.TERMINATED
        if agent_data.assistant_turns + 1 > self.args.max_assistant_turns:
            return AgentState.TERMINATED
        if agent_data.user_turns > self.args.max_user_turns:
            return AgentState.TERMINATED

        content, tool_calls = self.tool_parser.extract_tool_calls(output)
        agent_data.assistant_turns += 1
        agent_data.total_response_length += self.get_length(output)
        agent_data.messages.append({"role": "assistant", "content": output})
        agent_data.tool_calls = tool_calls
        agent_data.trajectory.append(
            {
                "turn": len(agent_data.trajectory),
                "type": "assistant",
                "assistant_turn": agent_data.assistant_turns,
                "content_before_tool": content,
                "raw_output": output,
                "tool_calls": tool_calls,
            }
        )

        if tool_calls:
            return AgentState.PROCESSING_TOOLS
        return AgentState.TERMINATED

    def _call_tool(self, tool_call):
        kwargs = tool_call["arguments"]
        return search(kwargs.get("query_list", []), top_k=self.args.top_k)

    def _handle_processing_tools_state(self, agent_data):
        tool_responses = []
        for tool_call in agent_data.tool_calls:
            response = self._call_tool(tool_call)
            tool_responses.append({"tool_call": tool_call, "tool_response": response})
            agent_data.total_response_length += self.get_length(response)
            if agent_data.total_response_length >= self.args.max_response_length:
                break

        for item in tool_responses:
            agent_data.messages.append({"role": "tool", "content": item["tool_response"]})

        agent_data.trajectory.append(
            {
                "turn": len(agent_data.trajectory),
                "type": "tool",
                "user_turn": agent_data.user_turns + 1,
                "tool_responses": tool_responses,
            }
        )
        agent_data.user_turns += 1

        if agent_data.total_response_length >= self.args.max_response_length:
            return AgentState.TERMINATED
        return AgentState.GENERATING

    def run(self, data):
        _, messages = get_query_and_messages(data)
        agent_data = TraceAgentData(messages)

        state = AgentState.GENERATING
        while state != AgentState.TERMINATED:
            if state == AgentState.GENERATING:
                state = self._handle_generating_state(agent_data)
            elif state == AgentState.PROCESSING_TOOLS:
                state = self._handle_processing_tools_state(agent_data)
            else:
                logger.error("Invalid state: %s", state)
                state = AgentState.TERMINATED

        response_messages = agent_data.messages[agent_data.init_messages_length :]
        full_response = get_qwen_response(response_messages)
        answer_match = re.search(r"<answer>(.*?)</answer>", full_response, re.DOTALL)
        response = answer_match.group(1).strip() if answer_match else full_response

        tool_call_count = sum(
            len(step.get("tool_calls", []))
            for step in agent_data.trajectory
            if step.get("type") == "assistant"
        )

        return {
            "response": response,
            "full_response": full_response,
            "tool_used": tool_call_count > 0,
            "tool_call_count": tool_call_count,
            "trajectory": agent_data.trajectory,
        }


def load_dataset(path, limit=None):
    data = []
    with open(path, "r", encoding="utf-8-sig") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
            if limit is not None and len(data) >= limit:
                break
    return data


def process_one(args, data):
    result = TracingToolAgentLoop(args).run(data)
    output = dict(data)
    output.update(result)
    return output


def main():
    parser = argparse.ArgumentParser(description="Run agent loop and save tool-call trajectory.")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--base_url", type=str, default="")
    parser.add_argument("--model_name", type=str, default="")
    parser.add_argument("--dataset", type=str, default="test_data.jsonl")
    parser.add_argument("--max_response_length", type=float, default=float("inf"))
    parser.add_argument("--max_assistant_turns", type=int, default=10)
    parser.add_argument("--max_user_turns", type=int, default=10)
    parser.add_argument("--concurrent", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default="./trajectory")
    args = parser.parse_args()

    dataset_path = os.path.join("./dataset", args.dataset)
    output_path = os.path.join(args.output_dir, f"{args.model_name}.{args.dataset}")
    os.makedirs(args.output_dir, exist_ok=True)

    datas = load_dataset(dataset_path, limit=args.limit)
    print(f"Start trajectory collection, count={len(datas)}, concurrent={args.concurrent}")
    print(f"Output path: {output_path}")

    with open(output_path, "w", encoding="utf-8") as fout:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrent) as executor:
            futures = [executor.submit(process_one, args, data) for data in datas]
            for future in tqdm(futures):
                try:
                    result = future.result()
                except Exception as exc:
                    result = {"error": repr(exc)}
                fout.write(json.dumps(result, ensure_ascii=False) + "\n")
                fout.flush()


if __name__ == "__main__":
    main()
