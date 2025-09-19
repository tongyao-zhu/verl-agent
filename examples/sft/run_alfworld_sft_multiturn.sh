#!/bin/bash

# Multi-turn SFT Training Script for AlfWorld Collected Data
# This script trains using multi-turn format (conversation messages)

set -x

# Set environment variables for better performance
export VLLM_WORKER_MULTIPROC_METHOD="spawn"
export TOKENIZERS_PARALLELISM=false


nproc_per_node=4
save_path=/home/aiops/zhuty/verl-agent/sft_models/model_sft1
sft_data_dir=/home/aiops/zhuty/verl-agent/sft_data_20250913_081853


# Validate that the SFT data directory exists
if [ ! -d "$sft_data_dir" ]; then
    echo "Error: SFT data directory '$sft_data_dir' does not exist!"
    echo "Please run the SFT data collection first:"
    echo "  ./examples/ppo_trainer/run_alfworld_sft_collection.sh"
    exit 1
fi

# Check for required data files (use Parquet format to preserve data types)
train_file="$sft_data_dir/train.parquet"
val_file="$sft_data_dir/val.parquet"

if [ ! -f "$train_file" ]; then
    echo "Error: Training data file '$train_file' not found!"
    echo "Make sure you collected SFT data using the collection script first."
    exit 1
fi

if [ ! -f "$val_file" ]; then
    echo "Error: Validation data file '$val_file' not found!"
    echo "Make sure you collected SFT data using the collection script first."
    exit 1
fi

echo "📊 Data files found:"
echo "   - Training: $train_file"
echo "   - Validation: $val_file"

# Create save directory if it doesn't exist
if [ ! -d "$save_path" ]; then
    mkdir -p "$save_path"
fi

echo "🚀 Starting AlfWorld Multi-turn SFT Training"
echo "   - Training data: $train_file"
echo "   - Validation data: $val_file"
echo "   - Save path: $save_path"
echo "   - GPUs: $nproc_per_node"
echo "   - Mode: Multi-turn (conversation messages)"
echo ""

torchrun --standalone --nnodes=1 --nproc_per_node=$nproc_per_node \
     -m verl.trainer.fsdp_sft_trainer \
    data.train_files=$train_file \
    data.val_files=$val_file \
    data.multiturn.enable=true \
    data.multiturn.messages_key=messages \
    data.max_length=2048 \
    optim.lr=1e-4 \
    data.train_batch_size=128 \
    data.micro_batch_size_per_gpu=2 \
    model.partial_pretrain=Qwen/Qwen2.5-1.5B-Instruct \
    trainer.default_local_dir=$save_path \
    trainer.experiment_name=alfworld-sft-multiturn-qwen2.5-1.5b \
    trainer.project_name=alfworld-sft-multiturn \
    trainer.logger=['console','wandb'] \
    trainer.total_epochs=5 \
    trainer.default_hdfs_dir=null \
    model.target_modules=all-linear \
    model.enable_gradient_checkpointing=True $@

echo ""
echo "🎉 Multi-turn SFT Training Complete!"
echo "   Model saved to: $save_path"
echo ""
echo "💡 Next Steps:"
echo "   - Check the saved model in: $save_path"
echo "   - Use the trained model for inference or further training"
echo "   - Monitor training progress in wandb (if enabled)"
