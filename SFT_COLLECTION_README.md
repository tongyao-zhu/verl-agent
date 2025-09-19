# SFT Data Collection using Existing PPO Infrastructure

This document explains how to collect SFT (Supervised Fine-Tuning) data by reusing the existing PPO trainer and environment infrastructure.

## Overview

Instead of creating a separate data collection system, we've modified the existing PPO trainer to enable SFT data collection mode. When `COLLECT_SFT=True` is set, the system:

1. **Enables validation-only mode** (`val_only=True`) - skips training, only runs validation
2. **Sets environment seed** for reproducible data collection
3. **Collects trajectories** during validation phase
4. **Saves data in SFT format** - both raw trajectories and processed training data

## Quick Start

### Basic Usage

```bash
# Set environment variables
export COLLECT_SFT=True
export SFT_SEED=42

# Run the existing PPO trainer - it will automatically switch to SFT collection mode
python3 -m verl.trainer.main_ppo [your normal PPO arguments]
```

### Using the Provided Script

```bash
# Run the pre-configured SFT collection script
./examples/ppo_trainer/run_alfworld_sft_collection.sh
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `COLLECT_SFT` | `False` | Enable SFT data collection mode |
| `SFT_SEED` | `None` | Set environment seed for reproducible data |
| `SFT_REQUIRE_SUCCESS` | `False` | Only save successful trajectories |

## Configuration

### Environment Variables Setup

```bash
export COLLECT_SFT=True                    # Enable SFT collection
export SFT_SEED=42                         # Set seed for reproducibility
export SFT_REQUIRE_SUCCESS=False           # Include both successful and failed trajectories
```

### What Happens When COLLECT_SFT=True

1. **Automatic Configuration Changes:**
   - `trainer.val_only = True` - Only run validation, skip training
   - `trainer.val_before_train = True` - Run validation immediately
   - `env.seed = SFT_SEED` - Set environment seed if provided

2. **Data Collection During Validation:**
   - Each validation batch is processed through the environment
   - Trajectories are collected and stored
   - Success information is extracted and preserved

3. **Data Saving:**
   - Raw trajectories saved as JSON
   - SFT training data saved as Parquet and CSV
   - Automatic train/validation split

## Output Files

The system creates a timestamped directory with:

```
sft_data_20241211_143022/
├── raw_trajectories.json    # Raw trajectory data with full context
├── train.parquet           # Training data in Parquet format
├── val.parquet             # Validation data in Parquet format
├── train.csv               # Training data in CSV format (human-readable)
└── val.csv                 # Validation data in CSV format
```

### Data Format

**Raw Trajectories (`raw_trajectories.json`):**
```json
[
  {
    "messages_list": [
      {"role": "user", "content": "..."},
      {"role": "assistant", "content": "..."}
    ],
    "success": true,
    "rewards": [0.5, 1.0],
    "final_reward": 1.5,
    "task_info": {...},
    "episode_length": 5
  }
]
```

**SFT Training Data (`train.parquet`):**
```
data_source | prompt | response | ability | reward_model | extra_info
agent_sft_collection | [...] | "action response" | agent | {...} | {...}
```

## Examples

### Example 1: Collect Mixed Success/Failure Data

```bash
export COLLECT_SFT=True
export SFT_SEED=42
export SFT_REQUIRE_SUCCESS=False

python3 -m verl.trainer.main_ppo \
    data.val_batch_size=32 \
    env.env_name=alfworld/AlfredTWEnv \
    env.seed=0 \
    [other PPO arguments...]
```

### Example 2: Collect Only Successful Trajectories

```bash
export COLLECT_SFT=True
export SFT_SEED=123
export SFT_REQUIRE_SUCCESS=True

python3 -m verl.trainer.main_ppo \
    data.val_batch_size=16 \
    env.env_name=alfworld/AlfredTWEnv \
    [other PPO arguments...]
```

### Example 3: Multiple Data Collections with Different Seeds

```bash
# Collection 1
export SFT_SEED=42
export COLLECT_SFT=True
python3 -m verl.trainer.main_ppo [args...] 

# Collection 2  
export SFT_SEED=123
python3 -m verl.trainer.main_ppo [args...]

# Collection 3
export SFT_SEED=456
python3 -m verl.trainer.main_ppo [args...]
```

## Integration with Existing Workflows

### Using with Different Environments

The SFT collection works with any environment supported by the PPO trainer:

```bash
# AlfWorld
export COLLECT_SFT=True
python3 -m verl.trainer.main_ppo env.env_name=alfworld/AlfredTWEnv [args...]

# WebShop  
export COLLECT_SFT=True
python3 -m verl.trainer.main_ppo env.env_name=webshop/WebShopEnv [args...]
```

### Using with Different Models

```bash
export COLLECT_SFT=True

# Qwen 1.5B
python3 -m verl.trainer.main_ppo actor_rollout_ref.model.path=Qwen/Qwen2.5-1.5B-Instruct [args...]

# Qwen 7B
python3 -m verl.trainer.main_ppo actor_rollout_ref.model.path=Qwen/Qwen2.5-7B-Instruct [args...]
```

## Tips and Best Practices

### 1. Batch Size Considerations
- Use smaller `val_batch_size` for more diverse episodes
- Larger batch sizes process faster but may be less diverse

### 2. Success Filtering
- Set `SFT_REQUIRE_SUCCESS=True` for high-quality data
- Set `SFT_REQUIRE_SUCCESS=False` to include learning examples

### 3. Reproducibility
- Always set `SFT_SEED` for reproducible data collection
- Use different seeds for different data collection runs

### 4. Data Quality
- Monitor success rates in the output summary
- Adjust environment parameters if success rate is too low/high

### 5. Storage
- Raw trajectories contain full context - useful for analysis
- Parquet files are optimized for training - use these for SFT

## Troubleshooting

### Common Issues

1. **No trajectories collected:**
   - Check that `COLLECT_SFT=True` is set
   - Verify validation data is available
   - Check environment setup

2. **Low success rates:**
   - Adjust environment difficulty
   - Check model capability
   - Consider using `SFT_REQUIRE_SUCCESS=False`

3. **Memory issues:**
   - Reduce `val_batch_size`
   - Use smaller models for collection
   - Process in multiple smaller runs

### Debug Information

The system provides detailed logging:
```
🎯 COLLECT_SFT mode enabled - will collect trajectories for SFT training
🎲 Set environment seed to 42 for reproducible SFT data
🎯 SFT data collection enabled - trajectories will be saved for SFT training
💾 Saving SFT data from X collected trajectories...
✅ SFT data collection completed! Data saved to: sft_data_20241211_143022
```

## Comparison with Existing Scripts

| Feature | `generate_alfworld_real_trajectories.py` | **New SFT Collection** |
|---------|------------------------------------------|----------------------|
| Infrastructure | Separate script | Reuses PPO trainer |
| Environment | Custom setup | Uses existing env setup |
| Model Integration | Manual setup | Uses existing model config |
| Configuration | Command line args | Hydra config + env vars |
| Maintenance | Separate codebase | Integrated with main code |
| Flexibility | AlfWorld specific | Works with any environment |

## Future Enhancements

Potential improvements:
1. **Streaming data collection** - Save data incrementally
2. **Custom filtering** - More sophisticated trajectory filtering
3. **Data augmentation** - Generate variations of collected trajectories
4. **Quality scoring** - Automatic quality assessment of trajectories
5. **Multi-environment** - Collect from multiple environments simultaneously

## Summary

This SFT collection feature provides:
- ✅ **Reuse existing infrastructure** - No duplicate code
- ✅ **Easy integration** - Just set environment variables
- ✅ **Flexible configuration** - Works with any PPO setup
- ✅ **Reproducible data** - Seed control for consistency
- ✅ **Multiple formats** - Raw and processed data
- ✅ **Quality control** - Success filtering options

The approach is much simpler than maintaining separate trajectory collection scripts while being more flexible and maintainable.
