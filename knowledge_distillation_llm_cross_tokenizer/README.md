## 训练
### 直接运行
python train.py
### torchrun
torchrun --nproc_per_node=2 train.py
## 部分python包的版本
- transformers==4.45.2
- torch==2.6.0
- torchaudio==2.6.0
- torchvision==0.21.0
- tokenizers==0.20.3
