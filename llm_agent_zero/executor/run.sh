curriculum_model_path=/home/user/wyf/llm_agent_zero/curriculum/models/step_20
executor_model_path=/home/user/Downloads/Qwen2.5-3B-Instruct
sandbox_url=http://0.0.0.0:8085/run_code
executor_train_data_path='data/executor_train_data.parquet'
executor_reward_path='executor_reward.py'

echo "加载训练之后的curriculum模型生成问题"
CUDA_VISIBLE_DEVICES=0 python3 generate_questions.py --model $curriculum_model_path --num_samples 1250

pip install -e ./verl

echo "过滤问题"
python3 filter_questions.py --input_file 'data/generated_questions.json' --output_file $executor_train_data_path


echo "训练executor模型"
CUDA_VISIBLE_DEVICES=2 python3 -m verl.trainer.main_ppo \
    --config-path=/home/user/wyf/llm_agent_zero/executor/verl/verl/trainer/config \
    --config-name=agent_python_trainer.yaml \
    algorithm.adv_estimator=grpo \
    data.train_files=$executor_train_data_path \
    data.val_files=$executor_train_data_path \
    data.train_batch_size=32 \
    data.max_prompt_length=1024 \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=$executor_model_path \
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
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    custom_reward_function.path=$executor_reward_path \
    reward_model.sandbox_fusion.url=$sandbox_url \
    trainer.critic_warmup=0 \
    trainer.logger=['tensorboard'] \
    trainer.project_name='executor_grpo' \
    trainer.experiment_name='executor_grpo' \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=5 \
    trainer.test_freq=10000 \
    trainer.total_epochs=2


# 训练完之后需要合并模型(修改模型路径之后在verl目录下运行)
# python3 scripts/legacy_model_merger.py merge \
# --backend fsdp \
# --local_dir /home/user/wyf/llm_agent_zero/executor/checkpoints/executor_grpo/executor_grpo/global_step_75/actor \
# --target_dir /home/user/wyf/llm_agent_zero/executor/models/step_75 \
# --hf_model_path /home/user/Downloads/Qwen2.5-3B-Instruct/

# 合并完之后可使用训练之后的executor模型再次优化curriculum模型

# 注意注意注意：
# 因为修改了verl的源码并且通过pip install -e ./verl重新编译了，
# 可能会对你当前环境之前使用的verl造成影响，所以为了保险起见，后续在使用当前环境的verl训练其他项目时，
# 请进入你使用的verl项目文件夹下重新编译你的verl环境，即pip install -e ./verl


