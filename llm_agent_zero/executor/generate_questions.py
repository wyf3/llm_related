import vllm
from transformers import AutoTokenizer
import argparse
from typing import List
from vllm.outputs import RequestOutput
import json
import regex as re
from prompts import curriculum_prompt
import os

from mathruler.grader import extract_boxed_content, grade_answer

def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = vllm.LLM(
        model=args.model,
        tokenizer=args.model,
        gpu_memory_utilization=0.8
    )
    

    prompt = tokenizer.apply_chat_template(
        curriculum_prompt, 
        tokenize=False,
        add_generation_prompt=True, 
        add_special_tokens=True
    )
    
    sample_params = vllm.SamplingParams(
        max_tokens=4096,
        temperature=1.0,
        top_p=0.95,
        n=1,
        stop_token_ids=[tokenizer.eos_token_id],
    )

    completions = model.generate([prompt]*args.num_samples, sampling_params=sample_params)
    results=[]
    for completion in completions:
        response = completion.outputs[0].text
        try:
            questions = re.findall(r"<question>(.*?)</question>", response, re.DOTALL)
            answer = extract_boxed_content(response)
            if answer == "None":
                answer = ""
            answers = [answer]

            if questions and answers:
                question = questions[-1].strip()
                answer = answers[-1].strip()
                results.append({"question": question, "answer": answer, "score": 0})
            else:
                results.append({"question": response, "answer": "", "score": -1})
        except:
            results.append({"question": response, "answer": "", "score": -1})
    with open(f"data/generated_questions.json", "w") as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-4B")
    parser.add_argument("--num_samples", type=int, default=1250, help="Number of samples to generate")

    args = parser.parse_args()

    main(args) 