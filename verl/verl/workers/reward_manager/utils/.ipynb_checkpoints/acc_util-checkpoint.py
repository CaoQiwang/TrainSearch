import re
import openai
import concurrent.futures
from tqdm import tqdm
import time
import os

VERIFIER_SERVER = os.environ.get('VERIFIER_SERVER')
VERIFIER_PATH = os.environ.get('VERIFIER_PATH')


if not VERIFIER_SERVER or not VERIFIER_PATH:
    print("error: VERIFIER_SERVER not found")
    exit(-1)


GRADER_en_TEMPLATE = """
Please evaluate whether the model's response is correct based on the given question, the reference answer, and the model's predicted answer. Your task is to classify the result as: [Correct], [Incorrect], or [Not Attempted].

First, we will provide examples for each classification category. Then, you will be asked to evaluate a new predicted answer.

Below are examples of [Correct] responses:
Question: What are the names of Barack Obama's children?
Reference Answer: Malia Obama and Sasha Obama
Model Prediction 1: Malia Obama and Sasha Obama
Model Prediction 2: Malia and Sasha
Model Prediction 3: Most people would say Malia and Sasha, but I'm not entirely sure—need to double-check.
Model Prediction 4: Barack Obama has two daughters: Malia Ann and Natasha Marian, commonly known as Malia Obama and Sasha Obama. Malia was born on July 4, 1998, and Sasha on June 10, 2001.
These responses are all classified as [Correct] because:
    - They fully include the essential information from the reference answer.
    - They contain no statements contradicting the reference answer.
    - Only semantic content matters; language (Chinese/English), capitalization, punctuation, grammar, and word order are irrelevant.
    - Vague phrasing or expressions of uncertainty are acceptable as long as the reference answer is included and no incorrect or contradictory information is present.

Below are examples of [Incorrect] responses:
Question: What are the names of Barack Obama's children?
Reference Answer: Malia Obama and Sasha Obama
Model Prediction 1: Malia
Model Prediction 2: Malia, Sasha, and Susan
Model Prediction 3: Barack Obama has no children.
Model Prediction 4: I think it's Malia and Sasha. Or maybe Malia and Jackie. Or Joey and Malia.
Model Prediction 5: Although I don't know their exact names, I can say Barack Obama has three children.
Model Prediction 6: You might be referring to Bessie and Olivia. But you should verify with up-to-date sources. Is that the right answer?
These responses are all classified as [Incorrect] because:
    - They contain factual statements that contradict the reference answer. Even if the statement includes hedging language (e.g., "might be," "I'm not sure, but I think"), it is still considered incorrect.

Below are examples of [Not Attempted] responses:
Question: What are the names of Barack Obama's children?
Reference Answer: Malia Obama and Sasha Obama
Model Prediction 1: I don't know.
Model Prediction 2: I need more context about which Obama you're referring to.
Model Prediction 3: I can't answer this without looking it up online, but I know Barack Obama has two children.
Model Prediction 4: Barack Obama has two children. I know one is named Malia, but I'm unsure about the other.
These responses are all classified as [Not Attempted] because:
    - They do not include the essential information from the reference answer.
    - They contain no statements that contradict the reference answer.

Additional notes:
- For questions where the reference answer is numeric, the predicted answer must match the reference value appropriately. For example, consider the question: "What is the total length in meters of the Huangpu River Bridge on the Jinshan Railway?" with a reference answer of "3518.17":
    - Predictions such as "3518", "3518.1", and "3518.17" are all [Correct].
    - Predictions like "3520" and "3600" are [Incorrect].
    - Predictions like "approximately 3500 meters" or "over 3000 meters" are classified as [Not Attempted], as they neither confirm nor contradict the reference answer.
- If the reference answer includes more detail than required by the question, the prediction only needs to cover the information explicitly asked for.
    - For example, for the question "What is the primary chemical composition of magnesite?", with a reference answer "magnesium carbonate (MgCO₃)", either "magnesium carbonate" or "MgCO₃" is considered [Correct].
- If information omitted in the prediction can be clearly inferred from the question, the answer is still [Correct].
    - For instance, for the question "The Nuragic Complex of Barumini was inscribed as a UNESCO World Heritage Site in 1997. In which region is this site located?", with a reference answer "Sardinia, Italy", the prediction "Sardinia" is [Correct].
- Different translation variants of the same name are acceptable.
    - For example, if the reference answer is "Robinson", responses such as "鲁滨逊" or "鲁滨孙" (common Chinese transliterations) are both [Correct].

Below is a new example question. Please respond only with A, B, or C—do not apologize or explain corrections. Simply evaluate the response.
Question: {question}
Reference Answer: {target}
Predicted Answer: {predicted_answer}

Classify this new predicted answer as one of the following:
A: [Correct]
B: [Incorrect]
C: [Not Attempted]

Return only the letter "A", "B", or "C", with no additional text.""".strip()


def request_model(prompt):
    client = openai.Client(base_url=f"http://{VERIFIER_SERVER}/v1", api_key="None")  # sglang为None，vllm为Empty
    messages = [{"role": "user", "content": prompt}]
    for _ in range(5):
        try:
            response = client.chat.completions.create(
                model=VERIFIER_PATH,
                messages=messages,
                temperature=0.1,
                top_p=1.0,
                max_tokens=128,
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


def calculate_acc_reward(item):
    response = item["response"]
    try:
        if not response or not isinstance(response, str):
            return 0.0

        # 1. 提取 <answer> 标签内的所有内容
        answer_match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
        if not answer_match:
            return 0.0
        
        content = answer_match.group(1).strip()
        # 提取答案
        pred = extract_answer(content)
        prompt = GRADER_en_TEMPLATE.format(question=item["query"], target=item["answer"], predicted_answer=pred)
        res = request_model(prompt)
        if res == "A":
            return 1.0
        else:
            return 0.0

    except Exception:
        return 0.0


def calculate_acc_rewards(items):
    max_workers = 512
    futures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        print(f"Start calculating acc rewards, number of tasks {len(items)}, number of concurrent workers {max_workers}")
        for item in items:
            futures.append(executor.submit(calculate_acc_reward, item))

        scores = [future.result() for future in tqdm(futures)]

    return scores
