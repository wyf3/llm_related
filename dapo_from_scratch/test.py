from openai import OpenAI
client = OpenAI(api_key='ww', base_url='http://10.250.2.24:8036/v1')

SYSTEM_PROMPT = """
按照如下格式回答问题：
<think>
你的思考过程
</think>
<answer>
你的回答
</answer>
"""

completion = client.chat.completions.create(
model = 'qwen1.5b',

temperature=0.0,
logprobs = True,
messages=[
    {
        "role": "system", 
        "content": SYSTEM_PROMPT},
    {
        "role": "user",
        "content": "天上五只鸟，地上五只鸡，一共几只鸭",
    }
],
)
print(completion.choices[0].message.content)
