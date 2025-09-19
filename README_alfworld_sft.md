# AlfWorld SFT Trajectory Generation

This directory contains scripts for generating supervised fine-tuning (SFT) trajectories for the AlfWorld environment, adapted from the Sokoban worldmodel script.

## Files

1. **`generate_alfworld_real_trajectories.py`** - Generates trajectories from actual AlfWorld environment interactions
2. **`run_alfworld_real_sft.sh`** - Bash script to run real trajectory generation
3. **`generate_alfworld_sft_trajectories.py`** - Alternative implementation with hydra config support
4. **`config_alfworld.yaml`** - Configuration file for the scripts
5. **`README_alfworld_sft.md`** - This documentation
6. **`USAGE_EXAMPLES.md`** - Detailed usage examples

## Quick Start

Generate **real AlfWorld training data** from actual environment interactions:

```bash
# Make executable
chmod +x run_alfworld_real_sft.sh

# Basic usage - generates REAL trajectories
./run_alfworld_real_sft.sh

# High-quality dataset with success filtering
./run_alfworld_real_sft.sh \
    --num_episodes 100 \
    --max_steps 25 \
    --require_success

# Quick test with real environment
./run_alfworld_real_sft.sh \
    --num_episodes 5 \
    --max_steps 15
```

### Parameters

- `--model_path`: Hugging Face model path for tokenizer (default: "Qwen/Qwen2.5-1.5B-Instruct")
- `--num_episodes`: Number of episodes to run (default: 50)
- `--max_steps`: Maximum steps per episode (default: 30)
- `--env_num`: Number of parallel environments (default: 4)
- `--require_success`: Only keep successful episodes (rejection sampling)
- `--output_dir`: Output directory prefix (default: "alfworld_real_sft_data")
- `--log_dir`: Directory for log files (default: "logs")
- `--config_path`: Path to configuration file (default: "config_alfworld.yaml")

## Output Format

The scripts generate several files:

### Raw Trajectories (`raw_trajectories_*.json`)
```json
[
  {
    "id": 1,
    "messages_list": [
      {"role": "system", "content": "You are an expert agent..."},
      {"role": "user", "content": "Your task is to: put a clean apple in the fridge..."},
      {"role": "assistant", "content": "<think>I need to...</think>\n<action>go to kitchen</action>"},
      ...
    ],
    "rewards": [0.1, 0.2, 0.3],
    "task_description": "put a clean apple in the fridge",
    "success": true
  }
]
```

### Training Data (`alfworld_train.parquet`, `alfworld_val.parquet`)
```
data_source | prompt | response | ability | reward_model | extra_info
alfworld | [{"role": "system", ...}] | "<think>...</think>\n<action>...</action>" | agent | {...} | {...}
```

## AlfWorld Task Types

The script generates trajectories for various AlfWorld tasks:
- Put objects in containers (fridge, microwave, cabinet)
- Clean and place objects
- Heat/cool objects before placing
- Turn on/off appliances
- Multi-step object manipulation

## Trajectory Structure

Each trajectory follows this pattern:

1. **System Message**: Sets up the agent role
2. **Task Introduction**: User provides task description and initial observation
3. **Agent Response**: Assistant provides reasoning (`<think>`) and action (`<action>`)
4. **Environment Feedback**: User provides new observation and available actions
5. **Repeat steps 3-4** until task completion or max steps reached

## Differences from Sokoban Script

Key adaptations made for AlfWorld:

1. **No World State Replacement**: Unlike Sokoban, AlfWorld doesn't require replacing predicted states with real states
2. **Action Extraction**: Actions are extracted from `<action>` tags instead of state tags
3. **Task-Oriented**: Focused on household tasks rather than puzzle solving
4. **Natural Language Actions**: Uses natural language commands instead of movement directions

## Integration with Training Pipeline

The generated parquet files are compatible with the existing VERL training pipeline:

```bash
# Use the generated data in PPO training
python -m verl.trainer.main_ppo \
    data.train_files=alfworld_sft_data_*/alfworld_train.parquet \
    data.val_files=alfworld_sft_data_*/alfworld_val.parquet \
    env.env_name=alfworld/AlfredTWEnv \
    ...
```

## Advanced Usage

### Full Agent System Integration

If you have the full agent system set up, you can use the complete script:

```bash
python generate_alfworld_sft_trajectories.py --config-path=. --config-name=config_alfworld
```

This requires:
- Proper AlfWorld environment setup
- Agent system dependencies
- VLLM or similar inference engine

### Customization

To customize the trajectory generation:

1. **Modify Task Descriptions**: Edit `generate_alfworld_task_descriptions()` in the simple script
2. **Change Action Space**: Update `generate_alfworld_actions()` 
3. **Adjust Conversation Flow**: Modify `create_alfworld_trajectory()`
4. **Custom Reward Schemes**: Update the reward calculation logic

## Troubleshooting

### Common Issues

1. **Import Errors**: If using the full script, ensure all agent system dependencies are installed
2. **Memory Issues**: Reduce `num_trajectories` or `batch_size` if running out of memory
3. **Tokenizer Issues**: Make sure the model path is correct and accessible

### Performance Tips

1. **Start Small**: Begin with 10-20 trajectories to test the pipeline
2. **Batch Processing**: The full script supports batch processing for efficiency
3. **Parallel Generation**: Multiple trajectories can be generated in parallel

## Example Output

Running the simple script with default parameters will generate:
- ~50 trajectories
- ~400-500 training rows (depending on trajectory length)
- Train/validation split (80/20)
- Both CSV and Parquet formats

The generated data can be directly used for supervised fine-tuning of language models on AlfWorld tasks.
