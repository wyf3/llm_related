from transformers import TrainingArguments, Trainer, default_data_collator
from model import SiglipModel, SiglipConfig
from dataset import SiglipDataset, MyDataCollator
from transformers import AutoTokenizer, AutoProcessor
from transformers import ViTImageProcessor, ViTForImageClassification

def train():
    
    config = SiglipConfig(vision_model_name_or_path='/home/user/wyf/train_siglip_from_scratch/vit-base-patch16-224',
                          text_model_name_or_path='/home/user/wyf/chinese-roberta-wwm-ext')
    
    model = SiglipModel(config)
    tokenizer = AutoTokenizer.from_pretrained(config.text_model_name_or_path)
    processor = AutoProcessor.from_pretrained(config.vision_model_name_or_path)
  
    args = TrainingArguments(
        output_dir='./outputs',
        do_train=True,
        per_device_train_batch_size=32,
        learning_rate=1e-4,
        num_train_epochs=40,
        save_steps=2000,
        save_total_limit=5,
        fp16=True,
        gradient_accumulation_steps=8,
        logging_steps=100,
        report_to='none',
        dataloader_pin_memory=True,
        dataloader_num_workers=1,
    )
    dataset = SiglipDataset(text_data_path='/home/user/wyf/train_siglip_from_scratch/MUGE/all_texts.jsonl',
                            image_data_path='/home/user/wyf/train_siglip_from_scratch/MUGE/all_imgs.tsv',
                            tokenizer=tokenizer,
                            processor=processor,
                            max_seq_length=64)
    
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
        data_collator=MyDataCollator(tokenizer)
    )
    trainer.train(resume_from_checkpoint=True)
    trainer.save_model()
    trainer.save_state()
    
if __name__ == '__main__':
    train()