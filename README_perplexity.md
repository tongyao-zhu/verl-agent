# WebShop Observation Perplexity Calculator

This script calculates the perplexity of observation content from WebShop SFT (Supervised Fine-Tuning) data using a base language model.

## Overview

The script extracts the observation content that appears between:
- `"current observation is:"`
- `"Your admissible actions of the current situation are"`

It then calculates the perplexity of this extracted content using a specified language model (default: GPT-2).

## Installation

Install the required dependencies:

```bash
pip install -r requirements_perplexity.txt
```

## Usage

### Basic Usage

```bash
python calculate_observation_perplexity.py --data_file your_data.json
```

### Command Line Arguments

- `--data_file`: Path to your SFT data file (required). Supports JSON, JSONL, and Parquet formats.
- `--model_name`: Language model to use for perplexity calculation (default: 'gpt2')
- `--device`: Device to use - 'cuda' or 'cpu' (default: auto-detect)
- `--output_file`: Optional CSV file to save detailed results
- `--prompt_field`: Field name containing the prompt in your data (default: 'prompt')

### Examples

1. **Basic perplexity calculation with GPT-2:**
   ```bash
   python calculate_observation_perplexity.py --data_file webshop_sft_data.json
   ```
   
   For parquet files:
   ```bash
   python calculate_observation_perplexity.py --data_file webshop_sft_data.parquet
   ```

2. **Using GPT-2 medium model with GPU:**
   ```bash
   python calculate_observation_perplexity.py --data_file webshop_sft_data.json --model_name gpt2-medium --device cuda
   ```

3. **Save detailed results to CSV:**
   ```bash
   python calculate_observation_perplexity.py --data_file webshop_sft_data.json --output_file results.csv
   ```

4. **Run the example:**
   ```bash
   python example_usage_perplexity.py
   python calculate_observation_perplexity.py --data_file sample_webshop_data.json --output_file perplexity_results.csv
   ```

## Output

The script provides:

1. **Console output** with statistics:
   - Total records processed
   - Number of successful/failed extractions
   - Perplexity statistics (mean, median, std dev, min, max, percentiles)

2. **Optional CSV file** (if `--output_file` is specified) containing:
   - Index of each record
   - Extracted observation (truncated for display)
   - Observation length
   - Calculated perplexity

3. **Statistics JSON file** (if `--output_file` is specified) with all computed statistics

## Data Format

The script expects data in JSON, JSONL, or Parquet format where each record contains a prompt field (configurable via `--prompt_field`). The prompt should follow the WebShop format with the observation content between the specified markers.

Example data structure:
```json
[
  {
    "prompt": "You are an expert autonomous agent...current observation is: [observation content here]...Your admissible actions of the current situation are..."
  }
]
```

## Notes

- The script uses regex patterns to extract observations and can handle variations in formatting
- Perplexity is calculated as `exp(loss)` where loss is the cross-entropy loss from the language model
- Lower perplexity values indicate that the model finds the text more predictable/likely
- The script handles multi-line observations and various formatting variations
