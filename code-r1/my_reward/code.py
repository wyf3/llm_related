import re
import random
from openai import OpenAI

client = OpenAI(base_url='http://***/v1', api_key='***')
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
    根据执行过程判断是否成功解决问题
    
    ## 任务要求
    - 认真审视执行过程，做出正确的判断
    - 只输出是或否，不要输出多余内容
    
    ## 问题
    {}
    
    ## 执行过程
    {}'''
    
    result = get_llm_output(prompt.format(user_message, answer))
    if result == '是':
        return 1
    else:
        return -1



def exec_code(code: str) -> str:
    import requests
    
    url = 'http://10.250.2.24:8090/run_code'
    headers = {
        'Content-Type': 'application/json'
    }
    data = {
        'code': code,
        'language': 'python'
    }

    response = requests.post(url, json=data, headers=headers)
    stdout = response.json()['run_result']['stdout']
    stderr = response.json()['run_result']['stderr']
    print(stdout, stderr)
    return stdout[:1000], stderr[:1000]


def extract_code(text: str)-> str:
        
    code_block_pattern = re.compile(r'<code>(.*?)</code>', re.DOTALL)

    # Find all matches in the text
    code_blocks = code_block_pattern.findall(text)

    # If no code blocks are found, try to find indented code blocks
    if not code_blocks:
        return []
    return code_blocks
    
def code_result(solution_str):
    code_blocks = extract_code(solution_str)
    if not code_blocks:
        return '', ''
    code = code_blocks[-1]
    stdout, stderr = exec_code(code)
    return stdout, stderr
    
    

def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    
    is_valid, _ = is_valid_sequence(solution_str)
    if is_valid:
        score = 0.5
        stdout, stderr = code_result(solution_str)
        if 'error' in stderr.lower() or 'traceback' in stderr.lower():
            score -= 0.5
        else:
            score += 0.5
            user_message = extra_info['user_message']
            score += answer_reward(user_message, solution_str)
        print('+++++++++++++++++++++++++++++')
        print(solution_str)
        print('+++++++++++++++++++++++++++++')
        return score
    else:
        format_score = 0
        
        if solution_str.startswith('<think>'):
            format_score += 0.1
        if solution_str.endswith('</answer>'):
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

