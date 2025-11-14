set -x
ENGINE=vllm
export VLLM_ATTENTION_BACKEND=XFORMERS

num_cpus_per_env_worker=0.01 # The CPU resource allocated for each environment worker. If you want to use less CPU resources, you can decrease this value.

train_data_size=128 # match GRPO and GiGPO configuration (16 × 8)
val_data_size=128
NUM_GPUS=2

model_name=$1
if [ -z "$model_name" ]; then
    model_name="Qwen/Qwen2.5-1.5B-Instruct"
    experiment_name="ppo_qwen2.5_1.5b_eval"
elif [ "$model_name" == "base2-step10" ]; then
    experiment_name="ppo_qwen2.5_1.5b-base2-step10"
    model_name="/home/aiops/zhuty/verl-agent-checkpoints-2/ppo_qwen2.5_1.5b/global_step_10/hf"
elif [ "$model_name" == "base2-step20" ]; then
    experiment_name="ppo_qwen2.5_1.5b-base2-step20"
    model_name="/home/aiops/zhuty/verl-agent-checkpoints-2/ppo_qwen2.5_1.5b/global_step_20/hf"
elif [ "$model_name" == "base2-step30" ]; then
    experiment_name="ppo_qwen2.5_1.5b-base2-step30"
    model_name="/home/aiops/zhuty/verl-agent-checkpoints-2/ppo_qwen2.5_1.5b/global_step_30/hf"
elif [ "$model_name" == "base2-step40" ]; then
    experiment_name="ppo_qwen2.5_1.5b-base2-step40"
    model_name="/home/aiops/zhuty/verl-agent-checkpoints-2/ppo_qwen2.5_1.5b/global_step_40/hf"
elif [ "$model_name" == "base2-step50" ]; then
    experiment_name="ppo_qwen2.5_1.5b-base2-step50"
    model_name="/home/aiops/zhuty/verl-agent-checkpoints-2/ppo_qwen2.5_1.5b/global_step_50/hf"
elif [ "$model_name" == "base2-step60" ]; then
    experiment_name="ppo_qwen2.5_1.5b-base2-step60"
    model_name="/home/aiops/zhuty/verl-agent-checkpoints-2/ppo_qwen2.5_1.5b/global_step_60/hf"
elif [ "$model_name" == "base2-step70" ]; then
    experiment_name="ppo_qwen2.5_1.5b-base2-step70"
    model_name="/home/aiops/zhuty/verl-agent-checkpoints-2/ppo_qwen2.5_1.5b/global_step_70/hf"
elif [ "$model_name" == "base2-step80" ]; then
    experiment_name="ppo_qwen2.5_1.5b-base2-step80"
    model_name="/home/aiops/zhuty/verl-agent-checkpoints-2/ppo_qwen2.5_1.5b/global_step_80/hf"
elif [ "$model_name" == "base2-step90" ]; then
    experiment_name="ppo_qwen2.5_1.5b-base2-step90"
    model_name="/home/aiops/zhuty/verl-agent-checkpoints-2/ppo_qwen2.5_1.5b/global_step_90/hf"
elif [ "$model_name" == "base2-step100" ]; then
    experiment_name="ppo_qwen2.5_1.5b-base2-step100"
    model_name="/home/aiops/zhuty/verl-agent-checkpoints-2/ppo_qwen2.5_1.5b/global_step_100/hf"
elif [ "$model_name" == "base2-step110" ]; then
    experiment_name="ppo_qwen2.5_1.5b-base2-step110"
    model_name="/home/aiops/zhuty/verl-agent-checkpoints-2/ppo_qwen2.5_1.5b/global_step_110/hf"
elif [ "$model_name" == "base2-step120" ]; then
    experiment_name="ppo_qwen2.5_1.5b-base2-step120"
    model_name="/home/aiops/zhuty/verl-agent-checkpoints-2/ppo_qwen2.5_1.5b/global_step_120/hf"
elif [ "$model_name" == "base2-step130" ]; then
    experiment_name="ppo_qwen2.5_1.5b-base2-step130"
    model_name="/home/aiops/zhuty/verl-agent-checkpoints-2/ppo_qwen2.5_1.5b/global_step_130/hf"
elif [ "$model_name" == "base2-step140" ]; then
    experiment_name="ppo_qwen2.5_1.5b-base2-step140"
    model_name="/home/aiops/zhuty/verl-agent-checkpoints-2/ppo_qwen2.5_1.5b/global_step_140/hf"
elif [ "$model_name" == "base2-step150" ]; then
    experiment_name="ppo_qwen2.5_1.5b-base2-step150"
    model_name="/home/aiops/zhuty/verl-agent-checkpoints-2/ppo_qwen2.5_1.5b/global_step_150/hf"
elif [ "$model_name" == "base2-step160" ]; then
    experiment_name="ppo_qwen2.5_1.5b-base2-step160"
    model_name="/home/aiops/zhuty/verl-agent-checkpoints-2/ppo_qwen2.5_1.5b/global_step_160/hf"
elif [ "$model_name" == "base2-step170" ]; then
    experiment_name="ppo_qwen2.5_1.5b-base2-step170"
    model_name="/home/aiops/zhuty/verl-agent-checkpoints-2/ppo_qwen2.5_1.5b/global_step_170/hf"
elif [ "$model_name" == "base2-step180" ]; then
    experiment_name="ppo_qwen2.5_1.5b-base2-step180"
    model_name="/home/aiops/zhuty/verl-agent-checkpoints-2/ppo_qwen2.5_1.5b/global_step_180/hf"
elif [ "$model_name" == "base2-step190" ]; then
    experiment_name="ppo_qwen2.5_1.5b-base2-step190"
    model_name="/home/aiops/zhuty/verl-agent-checkpoints-2/ppo_qwen2.5_1.5b/global_step_190/hf"
elif [ "$model_name" == "base2-step200" ]; then
    experiment_name="ppo_qwen2.5_1.5b-base2-step200"
    model_name="/home/aiops/zhuty/verl-agent-checkpoints-2/ppo_qwen2.5_1.5b/global_step_200/hf"
elif [ "$model_name" == "base-step25" ]; then
    experiment_name="ppo_qwen2.5_1.5b-base-step25"
    model_name="/home/aiops/zhuty/verl-agent-checkpoints/ppo_qwen2.5_1.5b/global_step_25/hf"
elif [ "$model_name" == "base-step50" ]; then
    experiment_name="ppo_qwen2.5_1.5b-base-step50"
    model_name="/home/aiops/zhuty/verl-agent-checkpoints/ppo_qwen2.5_1.5b/global_step_50/hf"
elif [ "$model_name" == "base-step75" ]; then
    experiment_name="ppo_qwen2.5_1.5b-base-step75"
    model_name="/home/aiops/zhuty/verl-agent-checkpoints/ppo_qwen2.5_1.5b/global_step_75/hf"
elif [ "$model_name" == "base-step100" ]; then
    experiment_name="ppo_qwen2.5_1.5b-base-step100"
    model_name="/home/aiops/zhuty/verl-agent-checkpoints/ppo_qwen2.5_1.5b/global_step_100/hf"
elif [ "$model_name" == "base-step125" ]; then
    experiment_name="ppo_qwen2.5_1.5b-base-step125"
    model_name="/home/aiops/zhuty/verl-agent-checkpoints/ppo_qwen2.5_1.5b/global_step_125/hf"
elif [ "$model_name" == "base-step150" ]; then
    experiment_name="ppo_qwen2.5_1.5b-base-step150"
    model_name="/home/aiops/zhuty/verl-agent-checkpoints/ppo_qwen2.5_1.5b/global_step_150/hf"
elif [ "$model_name" == "" ]; then
    experiment_name="ppo_qwen2.5_1.5b_eval"
    model_name="Qwen/Qwen2.5-1.5B-Instruct"
else
    experiment_name="ppo_${model_name//\//_}"
fi

# if home in model_name, make sure it exists
if [[ "$model_name" == *"/home"* ]]; then
    if [ ! -d "$model_name" ]; then
        echo "Error: Model directory '$model_name' does not exist!"
        exit 1
    fi
fi

# if NUM_GPUS is not set, set it to 8
if [ -z "$NUM_GPUS" ]; then
    NUM_GPUS=8
fi

# python3 -m examples.data_preprocess.prepare \
#     --mode 'text' \
#     --train_data_size $train_data_size \
#     --val_data_size $val_data_size

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
    actor_rollout_ref.model.path=$model_name \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.n=8 \
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
    env.resources_per_worker.num_cpus=$num_cpus_per_env_worker \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='verl_agent_alfworld' \
    trainer.experiment_name=$experiment_name \
    trainer.n_gpus_per_node=$NUM_GPUS \
    trainer.default_local_dir=/home/aiops/zhuty/verl-agent-checkpoints-2/${experiment_name} \
    trainer.max_actor_ckpt_to_keep=100 \
    trainer.max_critic_ckpt_to_keep=1 \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=5 \
    trainer.total_epochs=150 \
    trainer.val_before_train=True \
    trainer.log_val_generations=100 \
    +trainer.val_only=True
