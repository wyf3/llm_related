from sentence_transformers.evaluation import RerankingEvaluator
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer
from peft import PeftModel



model = AutoModel.from_pretrained("Qwen3-Embedding-0.6B")
tokenizer = AutoTokenizer.from_pretrained("Qwen3-Embedding-0.6B")
lora_path = 'saves_lora_8b_negative_1'
model = PeftModel.from_pretrained(model, lora_path)
model = model.merge_and_unload()
model.save_pretrained("merged_model/Qwen3-Embedding-0.6B")
tokenizer.save_pretrained("merged_model/Qwen3-Embedding-0.6B")