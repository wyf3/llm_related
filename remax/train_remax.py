from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, TrainingArguments, Trainer
import time
from peft import get_peft_model, LoraConfig, TaskType
import pandas as pd
import datasets
import copy
import torch
from typing import List, Dict, Any
import math

import gc
import math
import os
import re
import textwrap
import time
from collections import defaultdict
from typing import Callable, Optional, Union
from dataclasses import field, dataclass

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

from trl.trainer.utils import (OnlineTrainerState,OnPolicyConfig,
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
    return [1 if response==ans else -1 for response, ans in zip(responses, answer)]


class DataCollator:
    
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch = {"prompt":[feature['prompt'] for feature in features], "answer":[feature['answer'] for feature in features]}
        return batch



@dataclass
class ReMaxConfig(OnPolicyConfig):
    
    exp_name: str = field(
        default=os.path.basename(__file__)[:-3],
        metadata={"help": "Name of this experiment."},
    )
    num_ppo_epochs: int = field(
        default=1,
        metadata={"help": "Number of epochs to train."},
    )
    kl_coef: float = field(
        default=0.05,
        metadata={"help": "KL coefficient."},
    )
    cliprange: float = field(
        default=0.2,
        metadata={"help": "Clip range."},
    )
    reward_clip_range: float = field(
        default=10.0,
        metadata={"help": "Clip range for rewards"},
    )
    ds3_gather_for_generation: bool = field(
        default=True,
        metadata={
            "help": "This setting applies to DeepSpeed ZeRO-3. If enabled, the policy model weights are gathered for "
            "generation, improving generation speed. However, disabling this option allows training models that "
            "exceed the VRAM capacity of a single GPU, albeit at the cost of slower generation."
        },
    )
    gamma: float = field(
        default=0.95,
        metadata={"help": "Discount factor."},
    )



class ReMaxTrainer(Trainer):
    
    
    def __init__(
        self,
        config: ReMaxConfig,
        processing_class: Optional[
            Union[PreTrainedTokenizerBase, BaseImageProcessor, FeatureExtractionMixin, ProcessorMixin]
        ],
        policy: nn.Module,
        ref_policy: nn.Module,
        reward_model: Union[nn.Module, Callable[[list[str]], list[float]]],
        train_dataset: Dataset,
        data_collator: Optional[DataCollatorWithPadding] = None,
        eval_dataset: Optional[Union[Dataset, dict[str, Dataset]]] = None,
        # less commonly used
        optimizers: tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        callbacks: Optional[list[TrainerCallback]] = None,
    ) -> None:
        if ref_policy is policy:
            raise ValueError(
                "`policy` and `ref_policy` cannot be the same object. If you want `ref_policy` to be the "
                "same as `policy`, you must mass a copy of it, or `None` if you use peft."
            )

        self.args = config
        args = config
        self.processing_class = processing_class
        self.policy = policy


        if data_collator is None:
            data_collator = DataCollatorWithPadding(self.processing_class)

       
        self.ref_policy = ref_policy
        self.reward_model = reward_model
        self.train_dataset = train_dataset
        self.train_dataset_len = len(train_dataset)
        self.data_collator = data_collator
        self.eval_dataset = eval_dataset
        self.optimizer, self.lr_scheduler = optimizers
        self.optimizer_cls_and_kwargs = None  # needed for transformers >= 4.47

        #########
        # calculate various batch sizes
        #########
        if args.total_episodes is None:  # allow the users to define episodes in terms of epochs.
            args.total_episodes = int(args.num_train_epochs * self.train_dataset_len)
        accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps)
        self.accelerator = accelerator
        args.world_size = accelerator.num_processes
        args.local_batch_size = (
            args.per_device_train_batch_size * args.gradient_accumulation_steps * args.num_mini_batches
        )
        args.micro_batch_size = int(args.per_device_train_batch_size * args.world_size)
        args.batch_size = int(args.local_batch_size * args.world_size)
        args.mini_batch_size = exact_div(
            args.batch_size, args.num_mini_batches, "`batch_size` must be a multiple of `num_mini_batches`"
        )
        args.local_mini_batch_size = exact_div(
            args.local_batch_size, args.num_mini_batches, "`local_batch_size` must be a multiple of `num_mini_batches`"
        )
        args.num_total_batches = math.ceil(
            args.total_episodes / args.batch_size
        )  # we may train for more than `total_episodes`
        time_tensor = torch.tensor(int(time.time()), device=accelerator.device)
        time_int = broadcast(time_tensor, 0).item()  # avoid different timestamps across processes
        args.run_name = f"{args.exp_name}__{args.seed}__{time_int}"
        self.local_seed = args.seed + accelerator.process_index * 100003  # Prime
        if args.num_sample_generations > 0:
            self.sample_generations_freq = max(1, args.num_total_batches // args.num_sample_generations)
        

        #########
        # setup model, optimizer, and others
        #########
        for module in [policy, ref_policy, reward_model]:
            if isinstance(module, nn.Module):
                disable_dropout_in_model(module)
        if args.stop_token and args.stop_token == "eos":
            args.stop_token_id = self.processing_class.eos_token_id
        self.model = policy
        self.create_optimizer_and_scheduler(
            num_training_steps=args.num_total_batches
        )  # note that we are calling `self.lr_scheduler.step()` manually only at the batch level

        #########
        ### trainer specifics
        #########
        default_callbacks = DEFAULT_CALLBACKS + get_reporting_integration_callbacks(self.args.report_to)
        self.callbacks = default_callbacks if callbacks is None else default_callbacks + callbacks
        self.callback_handler = CallbackHandler(
            self.callbacks, self.model, self.processing_class, self.optimizer, self.lr_scheduler
        )
        self.add_callback(PrinterCallback if self.args.disable_tqdm else DEFAULT_PROGRESS_CALLBACK)
        self.control = TrainerControl()
        self.state = OnlineTrainerState(
            is_local_process_zero=self.is_local_process_zero(),
            is_world_process_zero=self.is_world_process_zero(),
            stateful_callbacks=[
                cb for cb in self.callback_handler.callbacks + [self.control] if isinstance(cb, ExportableState)
            ],
        )

        self.current_flos = 0
        self.hp_search_backend = None
        self.is_deepspeed_enabled = getattr(self.accelerator.state, "deepspeed_plugin", None) is not None
        self.is_fsdp_enabled = getattr(self.accelerator.state, "fsdp_plugin", None) is not None
        # Create distant repo and output directory if needed
        self.hub_model_id = None
        if self.args.push_to_hub:
            self.init_hf_repo()
        if self.args.should_save:
            os.makedirs(self.args.output_dir, exist_ok=True)
        self.backup_model = None


        #########
        ### setup dataloader
        #########
        self.dataloader = DataLoader(
            self.train_dataset,
            batch_size=args.local_batch_size,
            shuffle=True,
            collate_fn=self.data_collator,
            drop_last=True,  # needed; otherwise the last batch will be of ragged shape
        )
        # sync random states for DataLoader(shuffle=True) before `accelerator.prepare`
        # see https://gist.github.com/vwxyzjn/2581bff1e48e185e0b85b6dfe1def79c
        torch.manual_seed(args.seed)
        self.model, self.optimizer, self.dataloader = accelerator.prepare(self.model, self.optimizer, self.dataloader)
        torch.manual_seed(self.local_seed)  # reset the local seed again

        self.eval_dataloader = DataLoader(
            self.eval_dataset,
            batch_size=args.per_device_eval_batch_size,
            collate_fn=self.data_collator,
            drop_last=True,
        )  # no need to shuffle eval dataset
        self.eval_dataloader = accelerator.prepare(self.eval_dataloader)

        if self.is_deepspeed_enabled:
            if isinstance(self.reward_model, nn.Module):
                self.reward_model = prepare_deepspeed(
                    self.reward_model, args.per_device_train_batch_size, args.fp16, args.bf16
                )
            self.ref_policy = prepare_deepspeed(
                self.ref_policy, args.per_device_train_batch_size, args.fp16, args.bf16
            )
            self.deepspeed = self.model
        else:
            self.ref_policy = self.ref_policy.to(self.accelerator.device)
            if isinstance(self.reward_model, nn.Module):
                self.reward_model = self.reward_model.to(self.accelerator.device)

    def get_train_dataloader(self) -> DataLoader:
        return self.dataloader

    def get_eval_dataloader(self) -> DataLoader:
        return self.eval_dataloader
    
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
        
        # 贪婪采样
        greedy_generation_config = GenerationConfig(
            max_new_tokens=args.response_length,
            do_sample=False
        )
        

        accelerator.print("===training policy===")
        start_time = time.time()
        
        # 训练过程记录的变量
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

                context_length = queries.shape[1]
                responses = []
                postprocessed_responses = []
                logprobs = []
                ref_logprobs = []
                scores = []
                baseline_scores = []
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
                    
                    greedy_query_responses, greedy_logitss = batch_generation(
                        unwrapped_model,
                        queries,
                        args.local_rollout_forward_batch_size,
                        processing_class.pad_token_id,
                        greedy_generation_config,
                    )

                # 将大batch分成若干小batch，生成经验
                for i in range(0, queries.shape[0], args.local_rollout_forward_batch_size):
                  
                    query = queries[i : i + args.local_rollout_forward_batch_size]
                    answer = answers[i : i + args.local_rollout_forward_batch_size]
                    query_response = query_responses[i : i + args.local_rollout_forward_batch_size]
                    greedy_query_response = greedy_query_responses[i : i + args.local_rollout_forward_batch_size]
                    response = query_response[:, context_length:]
                    greedy_response = greedy_query_response[:, context_length:]
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
                        
                        postprocessed_greedy_response = truncate_response(
                            args.stop_token_id, processing_class.pad_token_id, greedy_response
                        )
                    
                    # Response Processing 2. run reward model on the truncated responses
                    postprocessed_query_response = torch.cat((query, postprocessed_response), 1)
                    postprocessed_greedy_query_response = torch.cat((query, postprocessed_greedy_response), 1)
                    sequence_length = first_true_indices(postprocessed_response == processing_class.pad_token_id) - 1

                    if isinstance(reward_model, nn.Module):
                        _, score, _ = get_reward(
                            reward_model, postprocessed_query_response, processing_class.pad_token_id, context_length
                        )
                        _, baseline_score, _ = get_reward(
                            reward_model, postprocessed_greedy_query_response, processing_class.pad_token_id, context_length
                        )
                   
                    elif isinstance(reward_model, list):
                        scores_ = torch.zeros((query.shape[0], len(reward_model)))
                        baseline_scores_ = torch.zeros((query.shape[0], len(reward_model)))
                        
                        for i, rm in enumerate(reward_model):
                            if isinstance(rm, nn.Module):
                                _, score, _ = get_reward(
                            rm, postprocessed_query_response, processing_class.pad_token_id, context_length
                        )
                                _, baseline_score, _ = get_reward(
                            rm, postprocessed_greedy_query_response, processing_class.pad_token_id, context_length
                        )
                                
                                scores_[:, i] = score
                                baseline_scores_[:, i] = baseline_score
                            
                            else:
                                response_text = processing_class.batch_decode(postprocessed_response, skip_special_tokens=True)
                                greedy_response_text = processing_class.batch_decode(postprocessed_greedy_response, skip_special_tokens=True)
                                
                                scores_[:, i] = torch.tensor(rm(completions=response_text, answer=answer))
                                baseline_scores_[:, i] = torch.tensor(rm(completions=greedy_response_text, answer=answer))
                                
                    
                        score = scores_.sum(dim=1).to(device)
                        baseline_score = baseline_scores_.sum(dim=1).to(device)
                    
                                

                    
                    responses.append(response)
                    postprocessed_responses.append(postprocessed_response)
                    logprobs.append(logprob)
                    ref_logprobs.append(ref_logprob)
                    sequence_lengths.append(sequence_length)
                    scores.append(score)
                    baseline_scores.append(baseline_score)
                    

                # 组合所有小batch，还原为大batch
                responses = torch.cat(responses, 0)
                postprocessed_responses = torch.cat(postprocessed_responses, 0)
                logprobs = torch.cat(logprobs, 0)
                ref_logprobs = torch.cat(ref_logprobs, 0)
                sequence_lengths = torch.cat(sequence_lengths, 0)
                scores = torch.cat(scores, 0)
                baseline_scores = torch.cat(baseline_scores, 0)
                
                del (logprob, ref_logprob, score, baseline_score)
                torch.cuda.empty_cache()
                gc.collect()

                response_idxs = torch.arange(responses.shape[1], device=responses.device).repeat(responses.shape[0], 1)
                padding_mask = response_idxs > sequence_lengths.unsqueeze(1)
                logprobs = torch.masked_fill(logprobs, padding_mask, INVALID_LOGPROB)
                ref_logprobs = torch.masked_fill(ref_logprobs, padding_mask, INVALID_LOGPROB)

                # 计算优势
                kl = logprobs - ref_logprobs
                reward_scores = scores - baseline_scores
                reward_scores = torch.clamp(reward_scores, -args.reward_clip_range, args.reward_clip_range)
                kl_reward = -args.kl_coef * kl
                returns = self.compute_returns(kl_reward, reward_scores, sequence_lengths)

                advantages = returns
                torch.cuda.empty_cache()

            # 得到一个大batch的经验数据之后，开始优化模型，num_ppo_epochs为一个大batch数据重复使用多少次
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

                            # Compute new logprobs
                            new_logprobs = selective_log_softmax(logits, mb_responses)
                            new_logprobs = torch.masked_fill(
                                new_logprobs, padding_mask[micro_batch_inds], INVALID_LOGPROB
                            )

                             
                            
                            # Compute probability ratios
                            new_ratio = (new_logprobs - mb_logprobs).exp()
                            logprobs_diff = new_logprobs - mb_logprobs
                            ratio = torch.exp(logprobs_diff)

                            # PPO clipped loss
                            pg_losses = -mb_advantage * ratio
                            pg_losses2 = -mb_advantage * torch.clamp(ratio, 1.0 - args.cliprange, 1.0 + args.cliprange)
                            pg_loss_max = torch.max(pg_losses, pg_losses2)
                            pg_loss = pg_loss_max.mean()
                            
                            
                            # 如果使用reinforce loss    
                            # pg_losses = -new_logprobs * mb_advantage
                            # pg_loss = pg_losses.mean()

                            # Final loss
                            loss = pg_loss

                            # Optimization step
                            accelerator.backward(loss)
                            optimizer.step()
                            optimizer.zero_grad()
                            
                            # 训练过程记录
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
                mean_non_score_reward = kl_reward.mean()
                eps = int(self.state.episode / (time.time() - start_time))
                metrics = {}
                metrics["eps"] = eps
                metrics["objective/kl"] = self.accelerator.gather_for_metrics(mean_kl).mean().item()
                metrics["objective/entropy"] = self.accelerator.gather_for_metrics(mean_entropy).mean().item()
                metrics["objective/non_score_reward"] = (
                    self.accelerator.gather_for_metrics(mean_non_score_reward).mean().item()
                )
                metrics["objective/rlhf_reward"] = self.accelerator.gather_for_metrics(returns).mean().item()
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
                self.state.epoch = self.state.episode / (self.train_dataset_len)
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


    def compute_returns(self, kl, reward_score, sequence_lengths):
        returns = torch.zeros_like(kl)  # (batch_size, sequence_length)
        
        batch_size = kl.shape[0]

        for j in range(batch_size):
            cumulative_reward = reward_score[j]
            cumulative_kl = 0
            for i in reversed(range(sequence_lengths[j])):
                cumulative_kl = kl[j, i]

                cumulative_reward *= self.args.gamma
                returns[j, i] += cumulative_kl + cumulative_reward   
        return returns


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


    training_args = ReMaxConfig(
        output_dir="output_remax",
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
        response_length = 200,
        num_train_epochs=10,
        save_steps = 100,
        save_total_limit=5,
        temperature=1.0,
        stop_token_id=tokenizer.eos_token_id,
        report_to = "tensorboard"
    )
    dataset = datasets.load_dataset("data")['train']
 
    trainer = ReMaxTrainer(
        config = training_args,
        policy = model,
        ref_policy = copy.deepcopy(model),
        processing_class = tokenizer,
        reward_model = [correctness_reward],
        train_dataset = dataset,
        data_collator=DataCollator()
    )
    trainer.train()