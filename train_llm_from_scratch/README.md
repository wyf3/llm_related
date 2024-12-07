# 使用方法

## 下载数据

https://github.com/jingyaogong/minimind
![image](.\screenshot-20241207-093824.png)

## 开始训练
### 直接运行
预训练:\
python train.py\
SFT:\
python sft_train.py
### torchrun
预训练:\
torchrun --nproc_per_node=2 train.py
SFT:\
torchrun --nproc_per_node=2 sft_train.py
### deepspeed
预训练:\
deepspeed --include 'localhost:0,1' train.py\
SFT:\
deepspeed --include 'localhost:0,1' sft_train.py

## 测试
test_llm.ipynb