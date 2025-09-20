from sentence_transformers import SentenceTransformer
from sentence_transformers.evaluation import RerankingEvaluator
from datasets import load_dataset


def get_eval_data(path):
    
    samples = []
    eval_dataset = load_dataset(path)['validation']
    for i in eval_dataset:
        samples.append({'query': i['query'], 'positive': i['positive'], 'negative': i['negative']})
        
    return samples
    

samples = get_eval_data('origin_data')
evaluator = RerankingEvaluator(samples, show_progress_bar=True)




# print('Loading original model...')
# original_model = SentenceTransformer('/home/user/Downloads/Qwen3-Embedding-0.6B')
# original_model.cuda()
# original_model.eval()
print('Loading distillation model...')
distillation_model = SentenceTransformer('merged_model/Qwen3-Embedding-0.6B')
distillation_model.cuda()
distillation_model.eval()
# print('Loading teacher model...')
# teacher_model = SentenceTransformer('/home/user/Downloads/Qwen3-Embedding-8B')
# teacher_model.cuda()
# teacher_model.eval()

# print('Evaluating original model...')
# original_result = evaluator(original_model)
# print(original_result)
print('Evaluating distillation model...')
distillation_result = evaluator(distillation_model)
print(distillation_result)
# print('Evaluating teacher model...')
# teacher_result = evaluator(teacher_model)
# print(teacher_result)

    
    

