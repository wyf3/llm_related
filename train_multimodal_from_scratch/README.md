# 使用方法

## 下载模型及数据
### 下载qwen2.5-0.5b和siglip
qwen2.5-0.5b: \
https://hf-mirror.com/Qwen/Qwen2.5-0.5B-Instruct \
siglip: \
此处使用的是如下版本的siglip（模型小，但是效果可能没那么好，训练更快，显存要求更低）：\
https://hf-mirror.com/google/siglip-base-patch16-224

也可以使用效果更好的版本，但是模型会更大（注意，使用这个版本可能需要修改image_pad_num这个参数，这个版本的模型输出的图片特征为（b,729,dim），在图片压缩的时候是reshape成（b,729/9,dim*9））：\
https://hf-mirror.com/google/siglip-so400m-patch14-384

### 下载数据集
1、预训练数据：\
图片数据：\
https://hf-mirror.com/datasets/liuhaotian/LLaVA-CC3M-Pretrain-595K \
中文文本数据：\
https://hf-mirror.com/datasets/LinkSoul/Chinese-LLaVA-Vision-Instructions \
2、SFT数据:\
图片数据:\
https://hf-mirror.com/datasets/jingyaogong/minimind-v_dataset \
中文文本数据:\
https://hf-mirror.com/datasets/LinkSoul/Chinese-LLaVA-Vision-Instructions

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
python test.py
