## 训练
### 直接运行
预训练:\
python pretrain.py\
SFT:\
python sft_train.py
### torchrun
预训练:\
torchrun --nproc_per_node=2 pretrain.py\
SFT:\
torchrun --nproc_per_node=2 sft_train.py
### deepspeed
预训练:\
deepspeed --include 'localhost:0,1' pretrain.py\
SFT:\
deepspeed --include 'localhost:0,1' sft_train.py

## 测试
test_moe.ipynb