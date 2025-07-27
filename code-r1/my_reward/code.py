import re
import random
from openai import OpenAI

client = OpenAI(base_url='http://***/v1', api_key='**')
def get_llm_output(prompt):
    
    messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]

    completion = client.chat.completions.create(
        model = 'qwen3-32b',
        temperature=0.0,
        messages=messages,
        stream=False,
    )
    output = completion.choices[0].message.content
    return output


def extract_answer(text):
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()


def is_valid_sequence(content):
    
    
    
    # Check for balanced tags
    tags_to_check = ["think", "code", "observation", "answer"]
    for tag in tags_to_check:
        opening_count = len(re.findall(f"<{tag}>", content))
        closing_count = len(re.findall(f"</{tag}>", content))
        if opening_count != closing_count:
            return False, f"Mismatch in {tag} tags: {opening_count} opening vs {closing_count} closing tags"
    
    
    # 1. First split the content by any tags we recognize
    split_pattern = r"(</?(?:think|code|observation|answer)>)"
    parts = re.split(split_pattern, content)
    
    # 2. Keep track of the current position in the expected sequence
    state = "start"  # start -> think -> search -> information -> think -> ... -> answer -> end
    
    # 3. Check each part
    for i, part in enumerate(parts):
        # Skip empty parts
        if not part.strip():
            continue
            
        # Check if this is a tag
        if re.match(r"</?(?:think|code|observation|answer)>", part):
            # This is a tag, check if it's valid in the current state
            if part == "<think>" and state in ["start", "observation"]:
                state = "in_think"
            elif part == "</think>" and state == "in_think":
                state = "after_think"
            elif part == "<code>" and state == "after_think":
                state = "in_code"
            elif part == "</code>" and state == "in_code":
                state = "after_code"
            elif part == "<observation>" and state == "after_code":
                state = "in_observation"
            elif part == "</observation>" and state == "in_observation":
                state = "observation"
            elif part == "<answer>" and state == "after_think":
                state = "in_answer"
            elif part == "</answer>" and state == "in_answer":
                state = "end"
            else:
                return False, f"Unexpected tag {part} in state {state}"
        else:
            # This is content, check if it's valid in the current state
            if state in ["in_think", "in_code", "in_observation", "in_answer"]:
                # Content is allowed inside tags
                pass
            elif state in ["start", "after_think", "after_code", "observation"]:
                # Only whitespace is allowed between tags
                if part.strip():
                    return False, f"Unexpected content '{part.strip()}' between tags (state: {state})"
            else:
                return False, f"Unexpected content in state {state}"
    
    # Check final state
    if state != "end":
        return False, f"Incomplete sequence, ended in state {state}"
        
    return True, "Valid sequence format"

def answer_reward(user_message, answer):
    prompt = '''
    ## 任务目标
    判断答案是否满足问题要求
    
    ## 任务要求
    - 只输出是或否，不要输出多余内容
    
    ## 问题
    {}
    
    ## 答案
    {}'''
    
    result = get_llm_output(prompt.format(user_message, answer))
    if result == '是':
        return 1
    else:
        return 0
    

def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    
    is_valid, _ = is_valid_sequence(solution_str)
    if is_valid:
        answer = extract_answer(solution_str)
        print(answer)
        user_message = extra_info['user_message']
        print(user_message)
        answer_score = answer_reward(user_message, answer)
        format_score = 0.5
        return answer_score + format_score
    else:
        
        format_score = 0
        
        if solution_str.startswith('<think>'):
            format_score += 0.1
            
        if '</think><answer>' in solution_str.replace('\n', ''):
            format_score += 0.1
        
        if '<think>' in solution_str and '</think>' in solution_str:
            format_score += 0.02
        
        if '<code>' in solution_str and '</code>' in solution_str:
            format_score += 0.02
        
        if '<answer>' in solution_str and '</answer>' in solution_str:
            format_score += 0.02
            
        return format_score
