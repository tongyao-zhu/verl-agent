#!/bin/bash

# SFT Data Collection Script for AlfWorld
# This script demonstrates how to collect SFT data using the existing PPO trainer infrastructure

set -x
ENGINE=${1:-vllm}
export VLLM_ATTENTION_BACKEND=XFORMERS
export VLLM_WORKER_MULTIPROC_METHOD="spawn"
export TOKENIZERS_PARALLELISM=false
# SFT Collection Configuration
export COLLECT_SFT=True                    # Enable SFT data collection
export SFT_SEED=97                         # Set seed for reproducible data
export SFT_REQUIRE_SUCCESS=False           # Set to True to only collect successful trajectories
export PROMPT_TEMPLATE=add_wm2
num_cpus_per_env_worker=0.01

# Use smaller batch sizes for SFT collection to get more diverse episodes
train_data_size=128   # Smaller batch for more episodes
val_data_size=128

echo "🎯 SFT Data Collection Mode Enabled"
echo "   - COLLECT_SFT: $COLLECT_SFT"
echo "   - SFT_SEED: $SFT_SEED"
echo "   - SFT_REQUIRE_SUCCESS: $SFT_REQUIRE_SUCCESS"
echo "   - Data will be collected during validation phase"
echo "   - PROMPT_TEMPLATE: $PROMPT_TEMPLATE"
echo ""

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=gae \
    data.train_files=$HOME/data/verl-agent/text/train.parquet \
    data.val_files=$HOME/data/verl-agent/text/test.parquet \
    data.train_batch_size=$train_data_size \
    data.val_batch_size=$val_data_size \
    data.max_prompt_length=2048 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-1.5B-Instruct \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
    critic.optim.lr=1e-5 \
    critic.model.use_remove_padding=True \
    critic.model.path=Qwen/Qwen2.5-1.5B-Instruct \
    critic.model.enable_gradient_checkpointing=True \
    critic.ppo_micro_batch_size_per_gpu=16 \
    critic.model.fsdp_config.param_offload=False \
    critic.model.fsdp_config.optimizer_offload=False \
    algorithm.use_kl_in_reward=False \
    env.env_name=alfworld/AlfredTWEnv \
    env.seed=0 \
    env.max_steps=50 \
    +env.prompt_template=${PROMPT_TEMPLATE:-default} \
    env.resources_per_worker.num_cpus=$num_cpus_per_env_worker \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='verl_agent_alfworld_sft_collection' \
    trainer.experiment_name='sft_collection_qwen2.5_1.5b' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=5 \
    trainer.total_epochs=150 \
    trainer.val_before_train=True
echo ""
echo "🎉 SFT Data Collection Complete!"
echo "   Check the output directory for:"
echo "   - raw_trajectories.json (raw trajectory data)"
echo "   - train.parquet, val.parquet (SFT training data)"
echo "   - train.csv, val.csv (human-readable format)"
echo ""
echo "💡 Usage Tips:"
echo "   - Set SFT_REQUIRE_SUCCESS=True to only collect successful trajectories"
echo "   - Adjust val_data_size to collect more/fewer episodes"
echo "   - Use different SFT_SEED values for different data collections"
echo "   - Set PROMPT_TEMPLATE=add_wm1 to use world modeling prompts with <prediction> tags"
echo "   - The collected data can be used directly with SFT trainers"

