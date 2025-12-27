import regex as re
from typing import Dict, List
import json
from mathruler.grader import extract_boxed_content, grade_answer
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from collections import Counter
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import requests


# 计算bleu相似度
def bleu_distance_matrix(sentences):
    n = len(sentences)
    dist = np.zeros((n, n))
    smoother = SmoothingFunction().method1
    for i in tqdm(range(n), desc="  - Calculating BLEU distance matrix", leave=False):
        for j in range(i, n):
            if i == j:
                score = 1.0
            else:
                ref = [sentences[j].split()]
                hyp = sentences[i].split()
                score = sentence_bleu(ref, hyp, smoothing_function=smoother)
            dist[i, j] = dist[j, i] = 1 - score
    return dist



def cluster_share_per_problem(
        problems,
        distance_threshold: float = 0.5,
        linkage: str = "average"):
    if not problems:
        return []
    print('start clustering')
    start_time = time.time()
    dist_mat = bleu_distance_matrix(problems)

    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=distance_threshold,
        metric="precomputed",
        linkage=linkage
    )
    labels = clustering.fit_predict(dist_mat)
    print(f'end clustering, time: {time.time() - start_time}')
    total = len(problems)
    cluster_size = Counter(labels)
    cluster_ratio = {lab: sz / total for lab, sz in cluster_size.items()}

    proportions = [cluster_ratio[lab] for lab in labels]
    return proportions


def compute_score(data_sources, solution_strs, ground_truths, extra_infos=None) -> List[float]:
    
    results = []
    for solution_str in solution_strs:
    
        questions = re.findall(r"<question>(.*?)</question>", solution_str, re.DOTALL)
        answer = extract_boxed_content(solution_str)
        
        if answer == 'None':
            answer = ''
            
        answers = [answer]
        
        if questions and answers:
            try:
                question = questions[-1].strip()
                answer = answers[-1].strip()
                results.append({"question": question, "answer": answer})
            except:
                results.append({"question": "", "answer": ""})
        else:
            results.append({"question": "", "answer": ""})
        
    

    url = "http://0.0.0.0:8015/generate"
    headers = {
        "accept": "application/json",
        "Content-Type": "application/json"
    }
    data = {
        "questions": [result["question"] for result in results],
        "answers": [result["answer"] for result in results]
    }

    final_results = requests.post(url, headers=headers, json=data).json()['data']
    
    penalty = cluster_share_per_problem([result['question'] for result in final_results], distance_threshold=0.5)

   
    final_scores = []
    for i in tqdm(range(len(final_results)), desc=" - Calculating final scores"):
        
        # 出的题不能太难，也不能太简单
        difficulty_score = min(final_results[i]["score"],1-final_results[i]["score"]) if final_results[i]['question'] else -1
        
        # 鼓励出调用工具的题
        tool_score = min(final_results[i]['tool_count'], 4) * 0.05
        
        # 重复性惩罚
        penalty_score = penalty[i]
        
        final_score =  difficulty_score  + tool_score - penalty_score
        final_scores.append(final_score)
    
    print("Final scores:", final_scores)
    return final_scores