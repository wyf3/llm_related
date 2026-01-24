model_path=/home/user/Downloads/Qwen2.5-3B-Instruct
train_data_path='verl/data/math_train.parquet'
test_data_path='verl/data/math_test.parquet'
reward_path='verl/reward_func/math.py'

# verl:0.4.1.dev0
# vllm:0.8.2
# flash-attn:2.7.3
# pip install -e ./verl
HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICES=3 python3 -m verl.trainer.main_ppo \
    --config-path=/home/user/wyf/RL/GDPO/verl/verl/trainer/config \
    --config-name=ppo_trainer.yaml \
    algorithm.adv_estimator=gdpo \
    data.train_files=$train_data_path \
    data.val_files=$test_data_path \
    data.train_batch_size=32 \
    data.max_prompt_length=512 \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=$model_path \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.45 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    reward_model.reward_manager=gdpo \
    custom_reward_function.path=$reward_path \
    trainer.critic_warmup=0 \
    trainer.logger=['tensorboard'] \
    trainer.project_name='gsm8k' \
    trainer.experiment_name='gdpo_test' \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=40 \
    trainer.test_freq=10000 \
    trainer.total_epochs=1