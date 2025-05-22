import re
def extract_answer(text):
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()

def mark_num(text):
    reward = 0
    if text.count("<think>\n") == 1:
        reward += 0.125
        
    if text.count("</think>\n") == 1:
        reward += 0.125
        
    if text.count("<answer>\n") == 1:
        reward += 0.125
        
    if text.count("</answer>\n") == 1:
        reward += 0.125
    return reward

# 生成答案是否正确的奖励
def correctness_reward(prompts, responses, answers):
    
    extracted_responses = [extract_answer(r) for r in responses]
    print(f"问题:\n{prompts[0]}", f"\n答案:\n{answers[0]}", f"\n模型输出:\n{responses[0]}", f"\n提取后的答案:\n{extracted_responses[0]}")
    return [2.0 if response == str(ans) else 0.0 for response, ans in zip(extracted_responses, answers)]

# 生成答案是否是数字的奖励（单纯依赖结果是否正确进行奖励，条件很苛刻，会导致奖励比较稀疏，模型难以收敛，所以加上答案是否是数字的奖励，虽然答案错误，但是至少生成的是数字（对于数学问题），也要给予适当奖励）
def digit_reward(prompts, responses, answers):
    extracted_responses = [extract_answer(r) for r in responses]
    return [0.5 if response.isdigit() else 0.0 for response in extracted_responses]

# 格式奖励
def hard_format_reward(prompts, responses, answers):
    pattern = r"^<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>\n$"
    matches = [re.match(pattern, response) for response in responses]
    return [0.5 if match else 0.0 for match in matches]

# 标记奖励（改善格式奖励稀疏问题）
def mark_reward(prompts, responses, answers):
    return [mark_num(response) for response in responses]