from trl import RLOOTrainer, RLOOConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import time
from peft import get_peft_model, LoraConfig, TaskType
import pandas as pd
import datasets
import copy
import torch
from typing import List, Dict, Any
import math
from dataclasses import field, dataclass
import gc
import math
import os
import re
import textwrap
import time
from collections import defaultdict
from typing import Callable, Optional, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from accelerate import Accelerator
from accelerate.utils import broadcast, gather_object
from datasets import Dataset
from torch.utils.data import DataLoader

from transformers import (
    BaseImageProcessor,
    DataCollatorWithPadding,
    FeatureExtractionMixin,
    GenerationConfig,
    PreTrainedTokenizerBase,
    ProcessorMixin,
    Trainer,
    TrainerCallback,
    TrainerControl,
    is_wandb_available,
)

from transformers.integrations import get_reporting_integration_callbacks
from transformers.trainer import DEFAULT_CALLBACKS, DEFAULT_PROGRESS_CALLBACK
from transformers.trainer_callback import CallbackHandler, ExportableState, PrinterCallback

from trl.trainer.utils import (OnlineTrainerState,
    batch_generation,
    disable_dropout_in_model,
    exact_div,
    first_true_indices,
    forward,
    get_reward,
    prepare_deepspeed,
    print_rich_table,
    selective_log_softmax,
    truncate_response,
    log_table_to_comet_experiment)

from trl.models.utils import unwrap_model_for_generation
INVALID_LOGPROB = 1.0

lora_rank = 32


def extract_answer(text):
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()

def correctness_reward(prompts=None, completions=None, answer=None, **kwargs):
    responses = [extract_answer(completion) for completion in completions]
    print(f"模型输出：{completions[0]}")
    return [1 if str(response)==str(ans) else -1 for response, ans in zip(responses, answer)]


class DataCollator:
    
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch = {"prompt":[feature['prompt'] for feature in features], "answer":[feature['answer'] for feature in features]}
        return batch




@dataclass
class ReinForcePlusPlusConfig(RLOOConfig):
    use_baseline: bool = False # REINFORCE++-baseline

class ReinForcePlusPlusTrainer(RLOOTrainer):
    
      
    def train(self):
        args = self.args
        accelerator = self.accelerator
        optimizer = self.optimizer
        model = self.model
        self.model_wrapped = self.model
        ref_policy = self.ref_policy
        reward_model = self.reward_model
        processing_class = self.processing_class
        dataloader = self.dataloader
        device = accelerator.device

        def repeat_generator():
            while True:
                yield from dataloader

        iter_dataloader = iter(repeat_generator())

        generation_config = GenerationConfig(
            max_new_tokens=args.response_length,
            temperature=(args.temperature + 1e-7),
            top_k=0.0,
            top_p=1.0,
            do_sample=True
        )

        accelerator.print("===training policy===")
        start_time = time.time()
        stats_shape = (args.num_ppo_epochs, args.num_mini_batches, args.gradient_accumulation_steps)
        approxkl_stats = torch.zeros(stats_shape, device=device)
        pg_clipfrac_stats = torch.zeros(stats_shape, device=device)
        pg_loss_stats = torch.zeros(stats_shape, device=device)
        vf_clipfrac_stats = torch.zeros(stats_shape, device=device)
        entropy_stats = torch.zeros(stats_shape, device=device)
        ratio_stats = torch.zeros(stats_shape, device=device)
        model.train()

        # trainer state initialization
        self.state.global_step = 0
        self.state.episode = 0
        self.state.max_steps = (args.num_total_batches * args.num_mini_batches) // 2
        self.state.num_train_epochs = args.total_episodes / self.train_dataset_len
        # Compute absolute values for logging, eval, and save if given as ratio
        if args.logging_steps is not None:
            if args.logging_steps < 1:
                self.state.logging_steps = math.ceil(self.state.max_steps * args.logging_steps)
            else:
                self.state.logging_steps = args.logging_steps
        if args.eval_steps is not None:
            if args.eval_steps < 1:
                self.state.eval_steps = math.ceil(self.state.max_steps * args.eval_steps)
            else:
                self.state.eval_steps = args.eval_steps
        if args.save_steps is not None:
            if args.save_steps < 1:
                self.state.save_steps = math.ceil(self.state.max_steps * args.save_steps)
            else:
                self.state.save_steps = args.save_steps
        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

        for update in range(1, args.num_total_batches + 1):
            self.state.episode += 1 * args.batch_size
            data = next(iter_dataloader)
            with torch.no_grad():
                queries = data["prompt"]
                answers = data["answer"]
                processing_class.padding_side = "left"
                queries = processing_class.apply_chat_template(queries, tokenize=True, add_generation_prompt=True, return_tensors='pt', padding=True)
                queries = queries.to(device)
                
                
                if args.use_baseline:
                    queries = queries.repeat(args.rloo_k, 1)
                    answers = answers * args.rloo_k
                
                context_length = queries.shape[1]
                responses = []
                postprocessed_responses = []
                logprobs = []
                ref_logprobs = []
                scores = []
                sequence_lengths = []

                # Generate responses and compute logprobs
                with unwrap_model_for_generation(
                    self.model, self.accelerator, gather_deepspeed3_params=self.args.ds3_gather_for_generation
                ) as unwrapped_model:
                    query_responses, logitss = batch_generation(
                        unwrapped_model,
                        queries,
                        args.local_rollout_forward_batch_size,
                        processing_class.pad_token_id,
                        generation_config,
                    )

                # Process responses in batches
                for i in range(0, queries.shape[0], args.local_rollout_forward_batch_size):
                  
                    query = queries[i : i + args.local_rollout_forward_batch_size]
                    answer = answers[i : i + args.local_rollout_forward_batch_size]
                    query_response = query_responses[i : i + args.local_rollout_forward_batch_size]
                    response = query_response[:, context_length:]
                    logits = logitss[i : i + args.local_rollout_forward_batch_size]
                    logprob = selective_log_softmax(logits, response)
                    del logits
                    torch.cuda.empty_cache()

                    ref_output = forward(ref_policy, query_response, processing_class.pad_token_id)
                    ref_logits = ref_output.logits[:, context_length - 1 : -1]
                    ref_logits /= args.temperature + 1e-7
                    ref_logprob = selective_log_softmax(ref_logits, response)
                    del ref_output, ref_logits
                    torch.cuda.empty_cache()

                    # Response Processing 1. truncate response after the first occurrence of `stop_token_id`
                    postprocessed_response = response
       
                    if args.stop_token_id is not None:  # handle the edge case when stop_token_id exists but is 0
                        postprocessed_response = truncate_response(
                            args.stop_token_id, processing_class.pad_token_id, response
                        )
                    
                    # Response Processing 2. run reward model on the truncated responses
                    postprocessed_query_response = torch.cat((query, postprocessed_response), 1)
                    sequence_length = first_true_indices(postprocessed_response == processing_class.pad_token_id) - 1

                    if isinstance(reward_model, nn.Module):
                        _, score, _ = get_reward(
                            reward_model, postprocessed_query_response, processing_class.pad_token_id, context_length
                        )
                    
                    elif isinstance(reward_model, list):
                        scores_ = torch.zeros((query.shape[0], len(reward_model)))
                        
                        for i, rm in enumerate(reward_model):
                            if isinstance(rm, nn.Module):
                                _, score, _ = get_reward(
                            rm, postprocessed_query_response, processing_class.pad_token_id, context_length
                        )
                                scores_[:, i] = score
                            else:
                                response_text = processing_class.batch_decode(postprocessed_response, skip_special_tokens=True)
                                
                                scores_[:, i] = torch.tensor(rm(completions=response_text, answer=answer))
                                
                    
                        score = scores_.sum(dim=1).to(device)
                    
                                

                    # Store batch results
                    responses.append(response)
                    postprocessed_responses.append(postprocessed_response)
                    logprobs.append(logprob)
                    ref_logprobs.append(ref_logprob)
                    sequence_lengths.append(sequence_length)
                    scores.append(score)

                # Concatenate all batched results
                responses = torch.cat(responses, 0)
                postprocessed_responses = torch.cat(postprocessed_responses, 0)
                logprobs = torch.cat(logprobs, 0)
                ref_logprobs = torch.cat(ref_logprobs, 0)
                sequence_lengths = torch.cat(sequence_lengths, 0)
                scores = torch.cat(scores, 0)
                
                del (logprob, ref_logprob, score)
                torch.cuda.empty_cache()
                gc.collect()

                # Response Processing 3. filter response. Ensure that the sample contains stop_token_id
                # responses not passing that filter will receive a low (fixed) score
                # only query humans on responses that pass that filter
                contain_eos_token = torch.any(postprocessed_responses == processing_class.eos_token_id, dim=-1)
                if args.missing_eos_penalty is not None:
                    scores[~contain_eos_token] -= self.args.missing_eos_penalty
                # accelerator.print(f"{scores=}, {(contain_eos_token.sum() / len(contain_eos_token))=}")

                # be very careful with `padding_mask_p1`; see https://excalidraw.com/#json=LWnzG4w2k5DjF_EOL_xPt,e2w3a-hFJ_gX5vOfeyXGTw
                response_idxs = torch.arange(responses.shape[1], device=responses.device).repeat(responses.shape[0], 1)
                padding_mask = response_idxs > sequence_lengths.unsqueeze(1)
                logprobs = torch.masked_fill(logprobs, padding_mask, INVALID_LOGPROB)
                ref_logprobs = torch.masked_fill(ref_logprobs, padding_mask, INVALID_LOGPROB)

                # 4. compute rewards
                # Compute KL divergence
                kl = logprobs - ref_logprobs

                # reinforce++
                if args.normalize_reward:
                    scores = (scores - scores.mean()) / (scores.std() + 1e-8)
                    scores = torch.clamp(scores, -args.reward_clip_range, args.reward_clip_range)

                # reinforce++使用token粒度的kl散度
                if args.token_level_kl:
                    # Token-level KL penalty: apply KL penalty per token
                    kl_reward = -args.kl_coef * kl

                    # Get the index of the last non-padded token for each sequence
                    eos_indices = padding_mask.size(1) - 1 - padding_mask.long().fliplr().argmax(dim=1, keepdim=True)
                    last_reward = torch.zeros_like(kl)
                    # Ensure scores has correct shape and type
                    scores_shaped = scores.reshape(-1, 1).to(kl.dtype)
                    last_reward.scatter_(dim=1, index=eos_indices, src=scores_shaped)
                    # ppo采用的方法，将奖励归因于最后一个token，最后一个token之前只有kl散度作为奖励
                    non_score_reward = kl_reward.sum(1)  # Keep this for logging
                    reward = last_reward + kl_reward
                    rlhf_reward = reward.sum(1)  # Sum across sequence length
                else:
                    # Sequence-level KL penalty: sum KL across tokens first
                    sequence_kl = kl.sum(1)
                    non_score_reward = -args.kl_coef * sequence_kl
                    rlhf_reward = non_score_reward + scores

                
                
                # 类似于grpo，使用一个组的平均奖励计算baseline，同时去掉了std
                if args.use_baseline:
                
                    rlhf_reward = rlhf_reward.reshape(args.rloo_k, -1)
                    rlhf_reward = rlhf_reward - rlhf_reward.mean(dim=0, keepdim=True)
                
                advantages = rlhf_reward
                advantages = advantages.flatten()
                

                # reinforce++会使用一个batch内的数据对优势进行归一化
                if args.normalize_advantage:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                torch.cuda.empty_cache()

            # Do multiple epochs of PPO training, with a fresh random shuffle in each epoch
            for ppo_epoch_idx in range(args.num_ppo_epochs):
                b_inds = np.random.permutation(args.local_batch_size)
                minibatch_idx = 0
                for mini_batch_start in range(0, args.local_batch_size, args.local_mini_batch_size):
                    mini_batch_end = mini_batch_start + args.local_mini_batch_size
                    mini_batch_inds = b_inds[mini_batch_start:mini_batch_end]
                    gradient_accumulation_idx = 0
                    for micro_batch_start in range(0, args.local_mini_batch_size, args.per_device_train_batch_size):
                        with accelerator.accumulate(model):
                            micro_batch_end = micro_batch_start + args.per_device_train_batch_size
                            micro_batch_inds = mini_batch_inds[micro_batch_start:micro_batch_end]

                            # Get batch data
                            mb_advantage = advantages[micro_batch_inds]
                            mb_responses = responses[micro_batch_inds]
                            mb_query_responses = query_responses[micro_batch_inds]
                            mb_logprobs = logprobs[micro_batch_inds]

                            # Forward pass
                            output = forward(model, mb_query_responses, processing_class.pad_token_id)
                            logits = output.logits[:, context_length - 1 : -1]
                            logits /= args.temperature + 1e-7

                            # 计算当前策略模型的logprobs
                            new_logprobs = selective_log_softmax(logits, mb_responses)
                            new_logprobs = torch.masked_fill(
                                new_logprobs, padding_mask[micro_batch_inds], INVALID_LOGPROB
                            )

                                
                            
                            # reinforce++ 使用token粒度的重要性权重
                            new_ratio = (new_logprobs - mb_logprobs).exp()
                            logprobs_diff = new_logprobs - mb_logprobs
                            ratio = torch.exp(logprobs_diff)

                            # reinforce++ 使用ppo loss更新模型
                            pg_losses = -mb_advantage * ratio
                            pg_losses2 = -mb_advantage * torch.clamp(ratio, 1.0 - args.cliprange, 1.0 + args.cliprange)
                            pg_loss_max = torch.max(pg_losses, pg_losses2)
                            pg_loss = pg_loss_max.mean()
                            
                            

                            loss = pg_loss

                            # Optimization step
                            accelerator.backward(loss)
                            optimizer.step()
                            optimizer.zero_grad()

                            with torch.no_grad():
                                pg_clipfrac = (pg_losses2 > pg_losses).float().mean()
                                prob_dist = torch.nn.functional.softmax(logits, dim=-1)
                                entropy = torch.logsumexp(logits, dim=-1) - torch.sum(prob_dist * logits, dim=-1)
                                approxkl = 0.5 * (logprobs_diff**2).mean()
                                approxkl_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = approxkl
                                pg_clipfrac_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = (
                                    pg_clipfrac
                                )
                                pg_loss_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = pg_loss
                                entropy_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = entropy.mean()
                                ratio_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = new_ratio.mean()
                        gradient_accumulation_idx += 1
                    minibatch_idx += 1

                    # del everything and empty cache
                    # fmt: off
                    del (
                        output, logits, new_logprobs, logprobs_diff, ratio, pg_losses,
                        pg_losses2, pg_loss, loss, pg_clipfrac, prob_dist, entropy, approxkl,
                        mb_advantage, mb_responses, mb_query_responses, mb_logprobs,
                    )
                    # fmt: on
                    torch.cuda.empty_cache()

            # Compute metrics
            with torch.no_grad():
                mean_kl = kl.sum(1).mean()
                mean_entropy = (-logprobs).sum(1).mean()
                mean_non_score_reward = non_score_reward.mean()
                eps = int(self.state.episode / (time.time() - start_time))
                metrics = {}
                metrics["eps"] = eps
                metrics["objective/kl"] = self.accelerator.gather_for_metrics(mean_kl).mean().item()
                metrics["objective/entropy"] = self.accelerator.gather_for_metrics(mean_entropy).mean().item()
                metrics["objective/non_score_reward"] = (
                    self.accelerator.gather_for_metrics(mean_non_score_reward).mean().item()
                )
                metrics["objective/rlhf_reward"] = self.accelerator.gather_for_metrics(rlhf_reward).mean().item()
                metrics["objective/scores"] = self.accelerator.gather_for_metrics(scores.mean()).mean().item()
                metrics["policy/approxkl_avg"] = self.accelerator.gather_for_metrics(approxkl_stats).mean().item()
                metrics["policy/clipfrac_avg"] = self.accelerator.gather_for_metrics(pg_clipfrac_stats).mean().item()
                metrics["loss/policy_avg"] = self.accelerator.gather_for_metrics(pg_loss_stats).mean().item()
                metrics["val/clipfrac_avg"] = self.accelerator.gather_for_metrics(vf_clipfrac_stats).mean().item()
                metrics["policy/entropy_avg"] = self.accelerator.gather_for_metrics(entropy_stats).mean().item()
                metrics["val/ratio"] = self.accelerator.gather_for_metrics(ratio_stats).mean().item()
                metrics["val/ratio_var"] = self.accelerator.gather_for_metrics(ratio_stats).var().item()
                metrics["val/num_eos_tokens"] = (responses == processing_class.eos_token_id).sum().item()
                metrics["lr"] = self.lr_scheduler.get_last_lr()[0]
                metrics["episode"] = self.state.episode
                self.state.epoch = self.state.episode / (args.rloo_k * self.train_dataset_len)  # used by self.log
                self.log(metrics)
            del kl, mean_kl, mean_entropy, scores

            self.lr_scheduler.step()
            self.state.global_step += 1
            self.control = self.callback_handler.on_step_end(args, self.state, self.control)
            if self.control.should_save:
                self._save_checkpoint(model, trial=None)
                self.control = self.callback_handler.on_save(self.args, self.state, self.control)
            torch.cuda.empty_cache()
            gc.collect()
            
            
            if self.eval_dataset and args.num_sample_generations > 0 and (update - 1) % self.sample_generations_freq == 0:
                self.generate_completions(sampling=True)
            
        

        # HF trainer specifics
        self.control = self.callback_handler.on_train_end(args, self.state, self.control)
        if self.control.should_save:
            self._save_checkpoint(model, trial=None, metrics=None)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)


    def generate_completions(self, sampling: bool = False):
        args = self.args
        processing_class = self.processing_class
        generation_config = GenerationConfig(
            max_new_tokens=self.args.response_length,
            temperature=(0.01 + 1e-7),
            top_k=0.0,
            top_p=1.0,
            do_sample=True,
        )

        table = defaultdict(list)
        with unwrap_model_for_generation(
            self.model, self.accelerator, gather_deepspeed3_params=self.args.ds3_gather_for_generation
        ) as unwrapped_model:
            for batch in self.eval_dataloader:
                query = batch["prompt"]
                answer = batch["answer"]
                processing_class.padding_side = "left"
                query = processing_class.apply_chat_template(query, tokenize=True, add_generation_prompt=True, return_tensors='pt', padding=True)
                with torch.no_grad():
                    context_length = query.shape[1]
                    query_response, _ = batch_generation(
                        unwrapped_model,
                        query,
                        query.shape[0],
                        processing_class.pad_token_id,
                        generation_config,
                    )
                    response = query_response[:, context_length:]
                    postprocessed_response = response
                    if args.stop_token_id is not None:  # handle the edge case when stop_token_id exists but is 0
                        postprocessed_response = truncate_response(
                            args.stop_token_id, processing_class.pad_token_id, response
                        )
                    table["query"].extend(
                        gather_object(processing_class.batch_decode(query, skip_special_tokens=True))
                    )
                    table["model response"].extend(
                        gather_object(processing_class.batch_decode(postprocessed_response))
                    )

                    postprocessed_query_response = torch.cat((query, postprocessed_response), 1)

                    if isinstance(self.reward_model, nn.Module):
                        _, score, _ = get_reward(
                            self.reward_model,
                            postprocessed_query_response,
                            processing_class.pad_token_id,
                            context_length,
                        )
                        
                    elif isinstance(self.reward_model, list):
                        scores_ = torch.zeros((query.shape[0], len(self.reward_model)))
                        
                        for i, rm in enumerate(self.reward_model):
                            if isinstance(rm, nn.Module):
                                _, score, _ = get_reward(
                            rm, postprocessed_query_response, processing_class.pad_token_id, context_length
                        )
                                scores_[:, i] = score
                            else:
                                response_text = processing_class.batch_decode(postprocessed_response, skip_special_tokens=True)
                                
                                scores_[:, i] = torch.tensor(rm(completions=response_text, answer=answer))
                                
                    
                        score = scores_.sum(dim=1).to(postprocessed_query_response.device)

                    table["score"].extend(self.accelerator.gather_for_metrics(score).float().cpu().numpy())

                if sampling:
                    break
        df = pd.DataFrame(table)

        if self.accelerator.is_main_process:
            print_rich_table(df.iloc[0 : 0 + 5])
            if "wandb" in args.report_to:
                import wandb

                if wandb.run is not None:
                    wandb.log({"completions": wandb.Table(dataframe=df)})

            if "comet_ml" in args.report_to:
                log_table_to_comet_experiment(
                    name="completions.csv",
                    table=df,
                )



if __name__ == "__main__":
 
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen2.5-7B-Instruct",
        torch_dtype=torch.bfloat16,  # 或者 torch.bfloat16
        trust_remote_code=True,
        # 如果需要4bit量化，可以添加以下参数：
        # load_in_4bit=True,
        # bnb_4bit_use_double_quant=True,
        # bnb_4bit_quant_type="nf4",
        # bnb_4bit_compute_dtype=torch.float16,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen2.5-7B-Instruct",
        trust_remote_code=True
    )

    # 配置LoRA参数
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_rank,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )

    # 应用LoRA
    model = get_peft_model(model, lora_config)

    # 打印模型信息
    model.print_trainable_parameters()


    training_args = ReinForcePlusPlusConfig(
        output_dir="output_reinforce++",
        do_train=True,
        learning_rate = 5e-6,
        adam_beta1 = 0.9,
        adam_beta2 = 0.99,
        weight_decay = 0.1,
        warmup_ratio = 0.1,
        lr_scheduler_type = "cosine",
        optim = "paged_adamw_32bit",# 可使用paged_adamw_8bit进一步降低显存
        logging_steps = 1,
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 8,
        bf16=True,
        length_column_name='prompt',
        response_length = 200,
        num_train_epochs=10,
        save_steps = 100,
        save_total_limit=5,
        temperature=1.0,
        stop_token_id=tokenizer.eos_token_id,
        rloo_k=1,
        report_to = "tensorboard",
        token_level_kl=True,
        normalize_advantage=True,
        normalize_reward=False,
        use_baseline=False,
    )
    dataset = datasets.load_dataset("data")['train']
 
    trainer = ReinForcePlusPlusTrainer(
        config = training_args,
        policy = model,
        ref_policy = copy.deepcopy(model),
        processing_class = tokenizer,
        reward_model = [correctness_reward],
        train_dataset = dataset,
        data_collator=DataCollator()
    )
    trainer.train()