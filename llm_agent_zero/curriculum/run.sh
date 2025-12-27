curriculum_model_path=/home/user/Downloads/Qwen2.5-3B-Instruct
executor_model_path=/home/user/Downloads/Qwen2.5-3B-Instruct
sandbox_url=http://0.0.0.0:8085/run_code
curriculum_train_data_path=data/curriculum_train.parquet
curriculum_reward_path=curriculum_reward.py
port=8015
# 注意curriculum_reward.py中的服务端口需要和port一致

pip install -e ./verl

# 需要先在其他终端启动如下服务
# CUDA_VISIBLE_DEVICES=0 python3 generate_server_with_tool.py --model_path $executor_model_path --sandbox_url $sandbox_url --port $port


echo "Start generate curriculum train data"
python3 generate_curriculum_train_data.py --num_samples 640 --output_path $curriculum_train_data_path

echo "Start training curriculum: $curriculum_model_path"

CUDA_VISIBLE_DEVICES=2 python3 -m verl.trainer.main_ppo \
    --config-path=/home/user/wyf/llm_agent_zero/curriculum/verl/verl/trainer/config \
    --config-name=grpo_trainer.yaml \
    algorithm.adv_estimator=grpo \
    data.train_files=$curriculum_train_data_path \
    data.val_files=$curriculum_train_data_path \
    data.train_batch_size=32 \
    data.max_prompt_length=1024 \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=$curriculum_model_path \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.45 \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    algorithm.use_kl_in_reward=True \
    reward_model.reward_manager=batch \
    trainer.logger=['tensorboard'] \
    trainer.project_name='curriculum_grpo' \
    trainer.experiment_name='curriculum_grpo' \
    custom_reward_function.path=$curriculum_reward_path \
    custom_reward_function.name=compute_score \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=5 \
    trainer.test_freq=10000 \
    trainer.total_epochs=2


# 训练完之后需要合并模型(修改模型路径之后在verl目录下运行)
# python3 scripts/legacy_model_merger.py merge \
# --backend fsdp \
# --local_dir /home/user/wyf/llm_agent_zero/curriculum/verl/checkpoints/curriculum_grpo/curriculum_grpo/global_step_20/actor \
# --target_dir /home/user/wyf/llm_agent_zero/curriculum/models/step_20 \
# --hf_model_path /home/user/Downloads/Qwen2.5-3B-Instruct/