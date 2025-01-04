# 使用方法

## 下载数据

https://github.com/jingyaogong/minimind
![image](.\screenshot-20241207-093824.png)

## 开始训练
### 直接运行
预训练:\
python moe_train.py\
SFT:\
python moe_sft_train.py
### torchrun
预训练:\
torchrun --nproc_per_node=2 moe_train.py
SFT:\
torchrun --nproc_per_node=2 moe_sft_train.py
### deepspeed
预训练:\
deepspeed --include 'localhost:0,1' moe_train.py\
SFT:\
deepspeed --include 'localhost:0,1' moe_sft_train.py

## 测试
python moe_test.py