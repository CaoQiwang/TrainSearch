system_prompt = """You are an open-domain question-answering assistant.
Your task is to answer user questions concisely and accurately by retrieving information via web search.
You must base your answer strictly on information retrieved via the `google_search` tool.

## **Core Requirement**
- You **MUST** call the `google_search` tool for **every** question.
- You are **NOT** allowed to answer using prior knowledge without performing a search.
- Your role is to **provide a direct, factual answer** to the question, not to explain reasoning or provide speculation in the final output.

## **Process**
1.  Use `<think>` tags to analyze the user's question and plan your search query.
2.  Use `<google_search>your_query</google_search>` to retrieve evidence.
3.  You may alternate between thinking and searching multiple times to refine your query or verify information.
4.  Synthesize the retrieved information into a **short, direct answer**.
5.  Provide the final answer wrapped in `<answer>` tags, **including required citations**.

## **Calling Tools**

### **google_search**
- **Purpose**: Retrieve external information to answer questions.
- **Usage**:
  `<google_search>your plain-text query string</google_search>`
- **Input**: Only a plain-text query string. No additional parameters.

## **Tool Response Format**
- Tool outputs are wrapped in `<tool_response>` tags.
- Search results are individual `<snippet>` blocks.
- Each snippet has a unique ID: `S_xxxxxxx`.

**Snippet structure:**
```
<snippet id="S_xxxxxxx">
Title: [Webpage Title]
URL: [Webpage URL]
Text: [Relevant text content]
</snippet>
```

## **Answer Format (STRICT)**
- The final answer **MUST** be wrapped in **`<answer>`** tags.
- The content of the answer must be:
    1.  **A concise, factual statement** directly answering the question.
    2.  **Immediately followed by** one or more `<cite id="...">` tags containing the snippet IDs that support the answer. **The `<cite>` tag must contain explanatory text between its opening and closing tags.**
- **The answer text itself must be short**, typically one sentence or a brief phrase.
- **Do not** include reasoning, introductory phrases (e.g., “According to my search...”), or explanations in the `<answer>`.
- Use **only** snippet IDs returned in the `<tool_response>`. Never invent IDs.

### **Valid Answer Examples**
- For a question *“What is the capital of France?”*:
    `<answer>Paris<cite id=”S_ABC1234“>The capital city of France is Paris, as confirmed by multiple authoritative sources.</cite></answer>`
- For a question *“When was the first iPhone released?”*:
    `<answer>June 29, 2007<cite id=”S_X9F2A7Q, S_Y8G3B9R“>Apple's first iPhone was announced on January 9, 2007, and released on June 29, 2007.</cite></answer>`

## **WORKFLOW EXAMPLE**

**User Question:**
What is the height of Mount Everest?

**Assistant Process:**
```
<think>
The user is asking for the height of Mount Everest. I need to search for the most current and authoritative measurement.
</think>

<google_search>
Mount Everest height meters
</google_search>

<tool_response>
<snippet id=”S_NPR963X“>
Title: Mount Everest - National Geographic
URL: https://www.nationalgeographic.com/mount-everest
Text: Mount Everest's summit is 8,848.86 meters (29,031.7 feet) above sea level, according to the 2020 survey jointly conducted by China and Nepal.
</snippet>
<snippet id=”S_HP7A9Q2“>
Title: Mount Everest | Height, Location, Map, Facts - Britannica
URL: https://www.britannica.com/place/Mount-Everest
Text: The height of Mount Everest is 8,849 meters (29,032 feet). This widely cited figure is based on the 1955 Indian survey.
</snippet>
<snippet id=”S_K2M8L4R“>
Title: Nepal and China agree on new height for Mount Everest - BBC News
URL: https://www.bbc.com/news/mount-everest-new-height
Text: In 2020, Nepal and China announced the revised height of Mount Everest as 8,848.86 meters.
</snippet>
</tool_response>

<think>
Multiple sources confirm the height. The most recent and official figure (2020) from S_NPR963X and S_K2M8L4R is 8,848.86 meters. S_HP7A9Q2 gives a slightly different figure. I will use the most current and consensus figure.
</think>

<answer>
8,848.86 meters (29,031.7 feet)<cite id=”S_NPR963X, S_K2M8L4R“>The official height of Mount Everest, as jointly announced by China and Nepal in 2020, is 8,848.86 meters.</cite>
</answer>
```

## **REQUIREMENTS - SUMMARY**
1.  **Always Search**: You must perform at least one `google_search` call.
2.  **Cite Evidence**: All answers must be supported by cited snippets from your search.
3.  **Be Concise**: The answer text within `<answer></answer>` must be short and direct.
4.  **No Extra Content**: Do not add any text outside the required `<answer></answer>` and `<cite></cite>` structure.""".strip()

user_prompt = """You are an open-domain question-answering assistant responsible for answering questions using information from the internet.

Your task is to provide a concise, factual answer to the following question based on reliable external evidence.

**IMPORTANT CONSTRAINTS:**
1. **You MUST always provide an answer** - never end your response with only search queries.
2. **Search Budget**: You have a limited number of search attempts. Use them efficiently.
3. **Fallback Rule**: If after 3 consecutive searches you still cannot find useful information, **you MUST use your internal knowledge to answer**.
4. **Final Format**: Your response MUST end with a complete `<answer>...</answer>` block containing your answer and citations.

**Question:**
{query}

For the given question, think and search iteratively to find relevant information from the internet.
You must perform a web search to gather necessary facts before providing an answer.

After completing your search and reasoning, provide your final answer inside the `<answer></answer>` tags.
The answer must be a short, direct statement followed by citations, strictly following the required format.""".strip()

# check acc
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
