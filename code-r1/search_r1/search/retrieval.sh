
DATA_NAME=nq

DATASET_PATH="/home/peterjin/mnt/data/$DATA_NAME"

SPLIT='test'
TOPK=3

INDEX_PATH=/home/peterjin/mnt/index/wiki-18
CORPUS_PATH=/home/peterjin/mnt/data/retrieval-corpus/wiki-18.jsonl
SAVE_NAME=e5_${TOPK}_wiki18.json

# INDEX_PATH=/home/peterjin/rm_retrieval_corpus/index/wiki-21
# CORPUS_PATH=/home/peterjin/rm_retrieval_corpus/corpora/wiki/enwiki-dec2021/text-list-100-sec.jsonl
# SAVE_NAME=e5_${TOPK}_wiki21.json

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python retrieval.py --retrieval_method e5 \
                    --retrieval_topk $TOPK \
                    --index_path $INDEX_PATH \
                    --corpus_path $CORPUS_PATH \
                    --dataset_path $DATASET_PATH \
                    --data_split $SPLIT \
                    --retrieval_model_path "intfloat/e5-base-v2" \
                    --retrieval_pooling_method "mean" \
                    --retrieval_batch_size 512 \
