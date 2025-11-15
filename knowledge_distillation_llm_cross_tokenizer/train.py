from transformers import AutoModelForCausalLM, AutoTokenizer, DefaultDataCollator
from peft import LoraConfig, get_peft_model, TaskType
from peft import PeftModel
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from transformers import Trainer, TrainingArguments
from dataset import SFTDataset,MyDataCollator


class ULDLoss(nn.Module):
   
    def __init__(self, student_tokenizer=None, teacher_tokenizer=None, crossentropy_weight=0.0, distillation_weight=1.0, temperature=1, skip_eos=False):
        super().__init__()
        
        self.student_tokenizer = student_tokenizer
        self.teacher_tokenizer = teacher_tokenizer
        self.crossentropy_weight = crossentropy_weight
        self.distillation_weight = distillation_weight
        self.temperature = temperature
        self.skip_eos = skip_eos

        vocab_mapping, teacher_matched_ids, student_matched_ids = self.init_vocab_mapping()
        
        self.vocab_mapping = vocab_mapping
        self.teacher_matched_ids = teacher_matched_ids
        self.student_matched_ids = student_matched_ids

    def __call__(
        self, student_logits, teacher_logits, student_labels, teacher_labels, student_input_ids, teacher_input_ids
    ):
        
        
        if self.crossentropy_weight > 0:
            shift_logits = student_logits[..., :-1, :].contiguous()
            shift_labels = student_labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss(ignore_index=self.student_tokenizer.pad_token_id)
            crossentropy_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            crossentropy_loss = self.crossentropy_weight * crossentropy_loss
        else:
            crossentropy_loss = 0.0

        distillation_loss = self.compute_distillation_loss(
            student_logits, teacher_logits, student_labels, teacher_labels, student_input_ids, teacher_input_ids
        )

        return crossentropy_loss + distillation_loss * self.distillation_weight

    def init_vocab_mapping(self):

        student_vocab = self.student_tokenizer.get_vocab()
        teacher_vocab = self.teacher_tokenizer.get_vocab()
        
        student_token_to_id = dict(student_vocab.items())
        vocab_mapping = {}
        
        teacher_matched_ids = set()
        student_matched_ids = set()

        for token_str, teacher_token_id in teacher_vocab.items():
            if token_str in student_token_to_id:
                student_token_id = student_token_to_id[token_str]
                vocab_mapping[teacher_token_id] = student_token_id
                teacher_matched_ids.add(teacher_token_id)
                student_matched_ids.add(student_token_id)

        return vocab_mapping, teacher_matched_ids, student_matched_ids
    
    
    def get_start_and_size_answers(self, answers, tokenizer):
        answers_index = []
        answers_size = []

        for answer in answers:
            answer_mask = answer.ne(tokenizer.pad_token_id)
            if not answer_mask.any():
                answers_index.append(0)
                answers_size.append(0)
                continue

            indices = answer_mask.nonzero(as_tuple=True)[0]
            answers_index.append(int(indices[0].item()))
            answers_size.append(int(answer_mask.sum().item()))
        return answers_index, answers_size
    
    
    def compute_distillation_loss(
        self, student_logits, teacher_logits, student_labels, teacher_labels, student_input_ids, teacher_input_ids
    ):
        
        student_answer_index, student_answer_size = self.get_start_and_size_answers(student_labels, self.student_tokenizer)
        teacher_answer_index, teacher_answer_size = self.get_start_and_size_answers(teacher_labels, self.teacher_tokenizer)

        
        if self.student_tokenizer.eos_token_id != self.student_tokenizer.pad_token_id:
            
            if self.skip_eos:
                student_answer_size = [size - 1 for size in student_answer_size]
        
        else:
            if not self.skip_eos:
                student_answer_size = [size + 1 for size in student_answer_size]
        
        if self.teacher_tokenizer.eos_token_id != self.teacher_tokenizer.pad_token_id:
            if self.skip_eos:
                teacher_answer_size = [size - 1 for size in teacher_answer_size]
        
        else:
            if not self.skip_eos:
                teacher_answer_size = [size + 1 for size in teacher_answer_size]
        

        batch_size = student_logits.size(0)
        distillation_losses = []

        for i in range(batch_size):
            student_start = student_answer_index[i]
            student_size = student_answer_size[i]
            teacher_start = teacher_answer_index[i]
            teacher_size = teacher_answer_size[i]

        
            student_answer_logits = student_logits[i, student_start : student_start + student_size] # [student_len, student_vocab_size]
            teacher_answer_logits = teacher_logits[i, teacher_start : teacher_start + teacher_size] # [teacher_len, teacher_vocab_size]

            student_probs = F.softmax(student_answer_logits / self.temperature, dim=-1)
            teacher_probs = F.softmax(teacher_answer_logits / self.temperature, dim=-1)

            student_token_ids = student_input_ids[i, student_start : student_start + student_size].tolist()  # [student_len]
            teacher_token_ids = teacher_input_ids[i, teacher_start : teacher_start + teacher_size].tolist()  # [teacher_len]

    
            # 针对tokenizer分词后长度不一致的问题，进行对齐处理
            # 方案一（截断）
            # min_length = min(len(student_token_ids), len(teacher_token_ids))
            # student_aligned = student_probs[:min_length, :]
            # teacher_aligned = teacher_probs[:min_length, :]
            
            # 方案二（对齐文本）
            if self.skip_eos:
                student_alignment_groups, teacher_alignment_groups = self.get_alignment_groups_from_ids(student_token_ids, teacher_token_ids)

                student_aligned = self.merge_prob_with_alignment_groups(student_probs, student_alignment_groups)
            
                teacher_aligned = self.merge_prob_with_alignment_groups(teacher_probs, teacher_alignment_groups)
            
            else:
                student_alignment_groups, teacher_alignment_groups = self.get_alignment_groups_from_ids(student_token_ids[:-1], teacher_token_ids[:-1])

                student_aligned = self.merge_prob_with_alignment_groups(student_probs[:-1, :], student_alignment_groups)
            
                teacher_aligned = self.merge_prob_with_alignment_groups(teacher_probs[:-1, :], teacher_alignment_groups)
            
                student_aligned = torch.cat([student_aligned, student_probs[-1:, :]], dim=0)
                teacher_aligned = torch.cat([teacher_aligned, teacher_probs[-1:, :]], dim=0)
        

            # 针对vocal size不一致的问题，进行对齐处理
            # 方案一（不区分匹配和不匹配的token，统一处理：sort+pad）
            # student_sorted = student_aligned.sort(dim=-1, descending=True).values
            # teacher_sorted = teacher_aligned.sort(dim=-1, descending=True).values

            # student_vocab_size = student_sorted.size(-1)
            # teacher_vocab_size = teacher_sorted.size(-1)
            # max_vocab_size = max(student_vocab_size, teacher_vocab_size)

            # if student_vocab_size < max_vocab_size:
            #     student_sorted = F.pad(student_sorted, (0, max_vocab_size - student_vocab_size))
            # if teacher_vocab_size < max_vocab_size:
            #     teacher_sorted = F.pad(teacher_sorted, (0, max_vocab_size - teacher_vocab_size))

            # # Compute L1 distance (ULD approach)
            # aligned_loss = F.l1_loss(student_sorted, teacher_sorted, reduction="sum")
            # aligned_loss /= student_aligned.size(0) 
            
            # 方案二（区分匹配和不匹配的token，分别处理：matched计算kl散度，unmatched排序后计算l1距离）
            aligned_loss = self.compute_hybrid_uld_loss(student_aligned, teacher_aligned)

            distillation_losses.append(aligned_loss)

        distillation_loss = torch.stack(distillation_losses).mean()
        return distillation_loss

    def get_alignment_groups_from_ids(self, student_token_ids, teacher_token_ids):
    
        def to_canonical_pieces(tok, ids):
            pieces = []
            prev = ""
            for k in range(len(ids)):
                cur = tok.decode(ids[: k + 1], skip_special_tokens=False, clean_up_tokenization_spaces=False)
                pieces.append(cur[len(prev) :])
                prev = cur
            return pieces

        s_pieces = to_canonical_pieces(self.student_tokenizer, student_token_ids)
        t_pieces = to_canonical_pieces(self.teacher_tokenizer, teacher_token_ids)

        i = j = 0
        s_buf = t_buf = ""
        s_group = []
        t_group = []
        s_groups = []
        t_groups = []

        def flush():
            if s_group and t_group:
                s_groups.append(s_group.copy())
                t_groups.append(t_group.copy())

        while i < len(s_pieces) or j < len(t_pieces):
            if s_buf == t_buf and s_buf != "":
                flush()
                s_buf = t_buf = ""
                s_group = []
                t_group = []
                continue

            if s_buf == "" and i < len(s_pieces):
                s_buf += s_pieces[i]
                s_group.append(i)
                i += 1
                continue
            if t_buf == "" and j < len(t_pieces):
                t_buf += t_pieces[j]
                t_group.append(j)
                j += 1
                continue

            if len(s_buf) <= len(t_buf):
                if i < len(s_pieces):
                    s_buf += s_pieces[i]
                    s_group.append(i)
                    i += 1
                elif j < len(t_pieces):
                    t_buf += t_pieces[j]
                    t_group.append(j)
                    j += 1
            else:
                if j < len(t_pieces):
                    t_buf += t_pieces[j]
                    t_group.append(j)
                    j += 1
                elif i < len(s_pieces):
                    s_buf += s_pieces[i]
                    s_group.append(i)
                    i += 1

        if s_buf == t_buf and s_group and t_group:
            flush()
        elif s_group or t_group:
        
            if s_group or t_group:
                if not s_group:
                    s_group = []
                if not t_group:
                    t_group = []
                if s_group or t_group:
                    s_groups.append(s_group.copy() if s_group else [])
                    t_groups.append(t_group.copy() if t_group else [])

        return s_groups, t_groups

    def merge_prob_with_alignment_groups(self, probs, alignment_groups):
     
       
        if not alignment_groups:
            return probs

        vocab_size = probs.size(-1)
        target_len = len(alignment_groups)
        aligned_probs = torch.zeros(target_len, vocab_size, device=probs.device)

    
        for group_idx, group in enumerate(alignment_groups):
            if len(group) > 1:
                eps = 1e-8
                logp = torch.log(probs[group[0]].clamp_min(eps))
                for idx in group[1:]:
                    if idx < probs.size(0):
                        logp = logp + torch.log(probs[idx].clamp_min(eps))
                aligned_probs[group_idx] = torch.softmax(logp, dim=-1)
            elif len(group) == 1:
                aligned_probs[group_idx] = probs[group[0]]
            else:
                aligned_probs[group_idx] = torch.zeros_like(probs[0])

        return aligned_probs

    def compute_hybrid_uld_loss(self, student_aligned, teacher_aligned):
        
        device = student_aligned.device
        student_vocab_size = student_aligned.size(-1)
        teacher_vocab_size = teacher_aligned.size(-1)

        if self.teacher_matched_ids:
            teacher_matched_token_ids = torch.tensor(sorted(self.teacher_matched_ids), dtype=torch.long, device=device)
            student_matched_token_ids = torch.tensor(
                [self.vocab_mapping[token_id.item()] for token_id in teacher_matched_token_ids], dtype=torch.long, device=device
            )
        else:
            teacher_matched_token_ids = torch.tensor([], dtype=torch.long, device=device)
            student_matched_token_ids = torch.tensor([], dtype=torch.long, device=device)

        teacher_matched_mask = torch.zeros(teacher_vocab_size, dtype=torch.bool, device=device)
        student_matched_mask = torch.zeros(student_vocab_size, dtype=torch.bool, device=device)

        if len(teacher_matched_token_ids) > 0:
            teacher_matched_mask[teacher_matched_token_ids] = True
            student_matched_mask[student_matched_token_ids] = True

        matched_loss = torch.tensor(0.0, device=device)
        matched_token_count = 0
        if len(teacher_matched_token_ids) > 0:
    
            teacher_matched_probs = teacher_aligned[:, teacher_matched_token_ids]  # [seq_len, num_matched]
            student_matched_probs = student_aligned[:, student_matched_token_ids]  # [seq_len, num_matched]
            matched_token_count = teacher_matched_probs.size(-1)
            matched_loss = self.compute_kl_loss(student_matched_probs, teacher_matched_probs)

        teacher_unmatched_mask = ~teacher_matched_mask
        student_unmatched_mask = ~student_matched_mask

        teacher_unmatched_probs = teacher_aligned[:, teacher_unmatched_mask]  # [seq_len, num_teacher_unmatched]
        student_unmatched_probs = student_aligned[:, student_unmatched_mask]  # [seq_len, num_student_unmatched]

        unmatched_loss = torch.tensor(0.0, device=device)
        if teacher_unmatched_probs.size(-1) > 0 and student_unmatched_probs.size(-1) > 0:
         
            teacher_unmatched_sorted = teacher_unmatched_probs.sort(dim=-1, descending=True).values
            student_unmatched_sorted = student_unmatched_probs.sort(dim=-1, descending=True).values

            teacher_unmatched_size = teacher_unmatched_sorted.size(-1)
            student_unmatched_size = student_unmatched_sorted.size(-1)
            max_unmatched_size = max(teacher_unmatched_size, student_unmatched_size)

            if teacher_unmatched_size < max_unmatched_size:
                teacher_unmatched_sorted = F.pad(
                    teacher_unmatched_sorted, (0, max_unmatched_size - teacher_unmatched_size)
                )
            if student_unmatched_size < max_unmatched_size:
                student_unmatched_sorted = F.pad(
                    student_unmatched_sorted, (0, max_unmatched_size - student_unmatched_size)
                )

            unmatched_loss = F.l1_loss(student_unmatched_sorted, teacher_unmatched_sorted, reduction="sum")
            unmatched_loss /= student_aligned.size(0)  
        
 
        matched_weight = matched_token_count / max(1, teacher_vocab_size)
        unmatched_weight = 1.0 - matched_weight
 
        total_loss = matched_weight * matched_loss + unmatched_weight * unmatched_loss

        return total_loss

    def compute_kl_loss(self, student_logits, teacher_logits):
        
        
        batch_seq_len, num_matched = student_logits.shape

        student_logits = student_logits.view(-1, num_matched)
        teacher_logits = teacher_logits.view(-1, num_matched)
 
        student_logits = student_logits / self.temperature
        teacher_logits = teacher_logits / self.temperature
        
        student_log_probs = F.log_softmax(student_logits, dim=-1)
        teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)

    
        kl_loss = F.kl_div(student_log_probs, teacher_log_probs, reduction="none", log_target=True)

        return kl_loss.mean()



class KGTrainer(Trainer):
    
    def __init__(
        self,
        model = None,
        teacher_model = None,
        args = None,
        data_collator = None, 
        train_dataset = None,
        eval_dataset = None,
        tokenizer = None,
        teacher_tokenizer = None,
        model_init = None, 
        compute_metrics = None, 
        callbacks = None,
        optimizers = (None, None), 
        preprocess_logits_for_metrics = None,
    ):
        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics
        )
        self.teacher_model = teacher_model
        self.teacher_tokenizer = teacher_tokenizer
        self.uld_loss_fn = ULDLoss(student_tokenizer=tokenizer, teacher_tokenizer=teacher_tokenizer)
    
    
    def get_inputs_from_texts(self, tokenizer, prompt_texts: list[str], answer_texts: list[str]):
        
        sequences = []
        labels_list = []
        attention_masks = []
        
        for prompt_text, answer_text in zip(prompt_texts, answer_texts):
            messages = [{'role': 'user', 'content': prompt_text}]
            prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            prompt_ids = tokenizer.encode(prompt)
            answer_ids = tokenizer.encode(answer_text, add_special_tokens=False) + [tokenizer.eos_token_id]
            sequence = prompt_ids + answer_ids
            attention_mask = [1] * len(sequence)
            labels = [tokenizer.pad_token_id] * len(prompt_ids) + answer_ids

            sequences.append(torch.tensor(sequence))
            labels_list.append(torch.tensor(labels))
            attention_masks.append(torch.tensor(attention_mask))
        
        input_ids = pad_sequence(sequences, batch_first=True, padding_value=tokenizer.pad_token_id)
        labels = pad_sequence(labels_list, batch_first=True, padding_value=tokenizer.pad_token_id)
        attention_mask = pad_sequence(attention_masks, batch_first=True, padding_value=0)

        return input_ids, labels, attention_mask
        
    
    def compute_loss(self, model, inputs, return_outputs=False,num_items_in_batch=None):
        
        
        input_ids = inputs['input_ids']
        labels = inputs['labels']
        prompt_ids = [input_id[:len(input_id) - len(label)] for input_id, label in zip(input_ids, labels)]
        
        prompt_texts = self.tokenizer.batch_decode(prompt_ids, skip_special_tokens=True)
        answer_texts = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        student_input_ids, student_labels, student_attention_mask = self.get_inputs_from_texts(self.tokenizer, prompt_texts, answer_texts)
        teacher_input_ids, teacher_labels, teacher_attention_mask = self.get_inputs_from_texts(self.teacher_tokenizer, prompt_texts, answer_texts)
        
        student_input_ids = student_input_ids.to(self.model.device)
        student_labels = student_labels.to(self.model.device)
        student_attention_mask = student_attention_mask.to(self.model.device)

        teacher_input_ids = teacher_input_ids.to(self.teacher_model.device)
        teacher_labels = teacher_labels.to(self.teacher_model.device)
        teacher_attention_mask = teacher_attention_mask.to(self.teacher_model.device)
        
        student_outputs = model(input_ids=student_input_ids, attention_mask=student_attention_mask)
        self.teacher_model.eval()
        with torch.no_grad():
            teacher_outputs = self.teacher_model(input_ids=teacher_input_ids, attention_mask=teacher_attention_mask)
            
        student_logits = student_outputs.logits
        teacher_logits = teacher_outputs.logits
        
        loss = self.uld_loss_fn(student_logits, teacher_logits, student_labels, teacher_labels, student_input_ids, teacher_input_ids)
        
        return (loss, student_outputs) if return_outputs else loss
        

if __name__ == '__main__':
    
    import os
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    model = AutoModelForCausalLM.from_pretrained("Qwen2.5-0.5B-Instruct",trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained("Qwen2.5-0.5B-Instruct",trust_remote_code=True)
    
    lora_config = LoraConfig(
    r=8,  
    lora_alpha=256,  
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.1, 
    task_type=TaskType.CAUSAL_LM)
 
    model = get_peft_model(model, lora_config)
    model.cuda()
    model.floating_point_ops = lambda s: 0
    print(model.print_trainable_parameters())
    
    teacher_tokenizer = AutoTokenizer.from_pretrained("glm-4-9b-chat",trust_remote_code=True)
    teacher_model = AutoModelForCausalLM.from_pretrained("glm-4-9b-chat",trust_remote_code=True)
    teacher_model.eval()
    teacher_model.cuda()
  
    
    args = TrainingArguments(output_dir='./results', 
                            num_train_epochs=1, 
                            do_train=True, 
                            per_device_train_batch_size=8,
                            gradient_accumulation_steps=1,
                            logging_steps=1,
                            report_to='tensorboard',
                            save_strategy='steps',
                            save_total_limit=3,
                            save_steps=100,
                            bf16=True,
                            learning_rate=0.00001,
                            lr_scheduler_type='cosine',
                            dataloader_num_workers=8,
                            dataloader_pin_memory=True)
    data_collator = MyDataCollator()
    train_dataset = SFTDataset('example.json', tokenizer=tokenizer)
    
    trainer = KGTrainer(model=model,
                        teacher_model=teacher_model, 
                        args=args, 
                        train_dataset=train_dataset, 
                        tokenizer=tokenizer, 
                        teacher_tokenizer=teacher_tokenizer,
                        data_collator=data_collator)

    trainer.train(resume_from_checkpoint=False)
    trainer.save_model('./saves')
    trainer.save_state()
    
      
    