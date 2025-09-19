# AlfWorld SFT Training Guide

This guide explains how to collect and train SFT data from AlfWorld environments using the integrated system.

## 📊 **Data Format Compatibility**

The SFT data collector now creates data compatible with **both** single-turn and multi-turn SFT trainers:

### **Generated Data Structure:**
```
{
  "data_source": "agent_sft_collection",
  "messages": [                           # For multi-turn SFT
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ],
  "prompt_data": {"content": "user: ..."},  # For single-turn SFT
  "response_data": {"content": "..."},      # For single-turn SFT
  "ability": "agent",
  "reward_model": "...",
  "extra_info": "..."
}
```

## 🚀 **Step-by-Step Process**

### **Step 1: Collect SFT Data**

**Basic collection (original prompts):**
```bash
./examples/ppo_trainer/run_alfworld_sft_collection.sh
```

**World modeling collection (with `<prediction>` tags):**
```bash
PROMPT_TEMPLATE=add_wm1 ./examples/ppo_trainer/run_alfworld_sft_collection.sh
```

**Output:** Creates `sft_data_YYYYMMDD_HHMMSS/` directory with:
- `train.parquet` (90% of data)
- `val.parquet` (10% of data)
- `train.csv`, `val.csv` (human-readable)
- `raw_trajectories.json` (full trajectory data)

### **Step 2: Choose Training Mode**

#### **Option A: Single-turn SFT (Recommended for simplicity)**
```bash
./examples/sft/run_alfworld_sft_single.sh 4 ./sft_models sft_data_20241211_143022
```

#### **Option B: Multi-turn SFT (Better for conversation understanding)**
```bash
./examples/sft/run_alfworld_sft_multiturn.sh 4 ./sft_models sft_data_20241211_143022
```

## ⚙️ **Configuration Details**

### **Single-turn SFT Configuration:**
```bash
data.prompt_key=prompt_data
data.response_key=response_data
data.prompt_dict_keys=['content']
data.response_dict_keys=['content']
```

### **Multi-turn SFT Configuration:**
```bash
data.multiturn.enable=true
data.multiturn.messages_key=messages
```

## 📈 **Training Parameters**

Both scripts use optimized parameters:
- **Model**: Qwen/Qwen2.5-1.5B-Instruct
- **LoRA**: rank=32, alpha=16 (efficient fine-tuning)
- **Batch size**: 128 global, 2 per GPU
- **Learning rate**: 1e-4
- **Epochs**: 3 (prevents overfitting)
- **Max length**: 2048 tokens
- **Validation split**: 10%

## 🎯 **Data Quality Features**

### **World Modeling Prompts** (`PROMPT_TEMPLATE=add_wm1`):
- Includes `<think>` tags for reasoning
- Includes `<prediction>` tags for next-state prediction
- Includes `<action>` tags for actions
- Better for training predictive capabilities

### **Standard Prompts** (default):
- Includes `<think>` tags for reasoning
- Includes `<action>` tags for actions
- Good for basic reasoning and action selection

## 🔧 **Customization Options**

### **Modify Collection Parameters:**
```bash
# Collect only successful trajectories
export SFT_REQUIRE_SUCCESS=True
./examples/ppo_trainer/run_alfworld_sft_collection.sh

# Use different seed for reproducibility
export SFT_SEED=123
./examples/ppo_trainer/run_alfworld_sft_collection.sh

# Use world modeling prompts
export PROMPT_TEMPLATE=add_wm1
./examples/ppo_trainer/run_alfworld_sft_collection.sh
```

### **Modify Training Parameters:**
```bash
# Increase learning rate
./examples/sft/run_alfworld_sft_single.sh 4 ./models data_dir optim.lr=5e-4

# More epochs
./examples/sft/run_alfworld_sft_single.sh 4 ./models data_dir trainer.total_epochs=5

# Larger LoRA rank
./examples/sft/run_alfworld_sft_single.sh 4 ./models data_dir model.lora_rank=64
```

## 🎉 **Expected Results**

### **Data Collection:**
- **Success rate**: Varies based on model capability (typically 20-80%)
- **Episodes**: Depends on validation batch size
- **Quality**: High-quality environment interactions

### **SFT Training:**
- **Training time**: ~1-3 hours for 3 epochs (depends on data size and hardware)
- **Model size**: ~3GB with LoRA (much smaller than full fine-tuning)
- **Performance**: Improved reasoning and action selection in AlfWorld tasks

## 🚨 **Troubleshooting**

### **Common Issues:**

1. **"KeyError: 'question'"**
   - **Solution**: Data format mismatch. Re-collect data with updated collector.

2. **"UndefinedError: dict object has no element 0"**
   - **Solution**: Use single-turn mode or ensure messages format is correct.

3. **"No such file or directory"**
   - **Solution**: Check data directory path and ensure collection completed.

### **Validation:**
```bash
# Check data format
head -2 sft_data_*/train.csv

# Check file sizes
ls -lh sft_data_*/

# Verify training works with small test
./examples/sft/run_alfworld_sft_single.sh 1 ./test_model data_dir trainer.total_epochs=1
```

## 💡 **Best Practices**

1. **Start with single-turn** - easier to debug and faster training
2. **Use world modeling prompts** - better data quality for agent training
3. **Monitor success rates** - adjust environment/model if too low/high
4. **Use different seeds** - collect diverse datasets
5. **Validate on small batches first** - test before full training
6. **Save multiple checkpoints** - for model comparison

## 🔄 **Workflow Summary**

```bash
# 1. Collect SFT data with world modeling
PROMPT_TEMPLATE=add_wm1 ./examples/ppo_trainer/run_alfworld_sft_collection.sh

# 2. Train with single-turn (recommended)
./examples/sft/run_alfworld_sft_single.sh 4 ./sft_models sft_data_20241211_143022

# 3. Use trained model for inference or further training
```

This integrated approach ensures high-quality SFT data that's fully compatible with the existing training infrastructure!
