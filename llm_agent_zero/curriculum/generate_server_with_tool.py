from prompts import executor_prompt, curriculum_prompt
import argparse
from transformers import AutoTokenizer
import re
import vllm
import requests
import stopit
from mathruler.grader import extract_boxed_content, grade_answer
import json
import tqdm
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Any
import uvicorn
app = FastAPI()


class Request(BaseModel):
    questions: List[str]
    answers: List[str]
    
class Response(BaseModel):
    code: int
    message: str
    data: Any




@stopit.threading_timeoutable(default='TIMED_OUT')
def grade_answer_with_timeout(res1, res2):
    return grade_answer(res1, res2)


def code_exec(code):
    
    
    try:

        url = args.sandbox_url

        payload = json.dumps({
        "code": code,
        "language": "python"
        })
        headers = {
        'Content-Type': 'application/json'
        }

        response = requests.request("POST", url, headers=headers, data=payload).json()
        
        if response.get("status") == "Success" and response.get("run_result"):
            run_info = response["run_result"]
            if run_info.get("status") == "Finished":
                stdout = run_info.get("stdout", "")
                return stdout if stdout else "[No output]"
            else:
                stderr = run_info.get('stderr', '')
                return f"Execution failed with status: {run_info.get('status')}\nStderr: {stderr}"
        else:
            return f"API Error: {response}"
    
    except Exception as e:
        return f"Execution Error: {e}"



def generate_with_tool_use(question: str, num_candidates: int = 10, max_turns: int = 4):
    
    messages = executor_prompt + [{'role': 'user', 'content': question}]
    
    conversations = [messages for _ in range(num_candidates)]
    final_assistant_messages = [""] * num_candidates
    
    # 标识是否生成完毕
    active_indices = list(range(num_candidates))

    # 最多调用四次工具
    tool_count = 0
    for turn in range(max_turns):
        if not active_indices:
            break

        
        prompts = [tokenizer.apply_chat_template(conversations[i], tokenize=False, add_generation_prompt=True) for i in active_indices]
      
        responses = model.generate(prompts, sampling_params_single_turn, use_tqdm=False)
    

        next_active_indices = []
        for i, response in enumerate(responses):
            original_index = active_indices[i]
            model_output = response.outputs[0].text.strip()
            
            
            
            # 取第一个代码块（提示词里面写了生成代码块之后立即停止，但是模型不一定完全遵从指令，有可能生成自己编造执行结果，然后再次生成）
            code_block_start_tag = "```python"
            code_block_end_tag = "```"
            start_index = model_output.find(code_block_start_tag)
            if start_index != -1:
                end_index = model_output.find(code_block_end_tag, start_index + len(code_block_start_tag))
                if end_index != -1:
                    model_output = model_output[:end_index + len(code_block_end_tag)]
            
            # 将生成的代码块添加到对话中（assistant）
            conversations[original_index].append({'role': 'assistant', 'content': model_output})

            code_match = re.search(r"```python\n(.*?)\n```", model_output, re.DOTALL)
            
            has_boxed = r'\boxed' in model_output

            # 有代码块，且没有最终答案，执行代码
            if code_match and not has_boxed:
                
                code_to_run = (code_match.group(1) or "").strip()
                if code_to_run:
                    exec_result = code_exec(code_to_run)
                    tool_count += 1
                    
                    tool_feedback = f"Code execution result: {exec_result}"
                    conversations[original_index].append({'role': 'user', 'content': tool_feedback})
                    next_active_indices.append(original_index)
                    
                else:
                    pass
            # 有最终答案
            elif has_boxed:
                
                final_assistant_messages[original_index] = model_output
            else:
                
                pass

        for i, response in enumerate(responses):
            original_index = active_indices[i]
            
            # 如果有最终答案，该分支结束
            if final_assistant_messages[original_index]:
                continue
            
            
            else:
                next_active_indices.append(original_index)
        
        active_indices = next_active_indices

    
    for i in range(num_candidates):
        if not final_assistant_messages[i]:
            for msg in reversed(conversations[i]):
                if msg['role'] == 'assistant':
                    final_assistant_messages[i] = msg['content']
                    break
    
    return final_assistant_messages, tool_count / num_candidates





def consolidate_and_grade(question, golden_answer, assistant_messages, tool_count):
    
    results = [extract_boxed_content(msg) for msg in assistant_messages]
    
    answer_counts = {}
    for res in results:
        if not res: continue
        matched = False
        
        for exist_ans in list(answer_counts.keys()):
            
            # 如果新答案 res 和以前的某个 exist_ans 完全相等，直接计数+1
            # 或者两个答案里都包含 "no "（比如都是否定回答），也当成同一类
            if res == exist_ans or ('no ' in res.lower() and 'no ' in exist_ans.lower()):
                answer_counts[exist_ans] += 1
                matched = True
                break
            
            try:
                is_match = False
                match_result_1 = grade_answer_with_timeout(res, exist_ans, timeout=20)
                if match_result_1 and match_result_1 != 'TIMED_OUT':
                    is_match = True

                if not is_match:
                    match_result_2 = grade_answer_with_timeout(exist_ans, res, timeout=20)
                    if match_result_2 and match_result_2 != 'TIMED_OUT':
                        is_match = True
                
                if is_match:
                    answer_counts[exist_ans] += 1
                    matched = True
                    break

            except Exception:
                continue
        
        if not matched:
            answer_counts[res] = 1

    if not answer_counts:
        majority_ans, max_count = '', 0
    else:
        majority_ans = max(answer_counts, key=answer_counts.get)
        max_count = answer_counts[majority_ans]

    # 多数答案占的比例
    score = max_count / len(assistant_messages) if assistant_messages else 0.0

    return {
        'question': question,
        'answer':   majority_ans,
        'score':    score if grade_answer(majority_ans, golden_answer) and score > 0.1 else 0, # 用标准答案再过滤一次（是否真正正确）
        'all_outputs':  assistant_messages,
        'extracted_results': results,
        'tool_count': tool_count
    }




@app.post("/generate", response_model=Response)
async def generate(request: Request):
    
    results_all = []
    
    questions = request.questions
    answers = request.answers

    for question, answer in tqdm.tqdm(zip(questions, answers), total=len(questions)):
    
        try:
            if question and answer:
                
                final_assistant_messages, tool_count = generate_with_tool_use(question, max_turns=4)
                
                
                item = consolidate_and_grade(question, answer, final_assistant_messages, tool_count)
                print(item)
                results_all.append(item)
            else:
                results_all.append({'question': question, 'answer': answer, 'score': -1, 'all_outputs': [], 'extracted_results': [], 'tool_count': 0})
        except Exception as e:
            
            print(f'\n[server] Error processing question: {str(e)}')
            results_all.append({
                'question': question, 'answer': answer, 'score': -1, 'error': f'unhandled exception: {str(e)}', 'tool_count': 0
            })
    
    return Response(code=200, message='success', data=results_all)



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='/home/user/Downloads/Qwen2.5-3B-Instruct/')
    parser.add_argument('--gpu_memory_utilization', type=float, default=0.7,
                        help='The maximum GPU memory utilization fraction for vLLM.')
    parser.add_argument('--sandbox_url', type=str, default='http://0.0.0.0:8085/run_code')
    parser.add_argument('--port', type=int, default=8015)
    args = parser.parse_args()
    
    print('[init] Loading model...')
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = vllm.LLM(
        model=args.model_path,
        tokenizer=args.model_path,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

    sampling_params_single_turn = vllm.SamplingParams(
        max_tokens=2048,
        temperature=0.7,
        top_p=0.9,
        n=1,
        stop_token_ids=[tokenizer.eos_token_id]
    )
    
  

    uvicorn.run(app, host="0.0.0.0", port=args.port)
    
  