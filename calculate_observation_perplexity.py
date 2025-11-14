#!/usr/bin/env python3
"""
Calculate perplexity on observation content from WebShop SFT data.

This script extracts the observation content between:
- "current observation is:" 
- "Your admissible actions of the current situation are"

Then calculates the perplexity of this content using a base language model.
"""

import json
import re
import argparse
from typing import List, Dict, Tuple, Optional
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm
import pandas as pd


def extract_observation(text: str) -> Optional[str]:
    """
    Extract observation content from the prompt text.
    
    Args:
        text: The full prompt text
        
    Returns:
        The extracted observation content or None if not found
    """
    # Pattern to match content between the markers
    # Using case-insensitive search and handling potential variations
    pattern = r'current observation is:\s*(.*?)\s*Your admissible actions of the current situation'
    
    # Try with DOTALL flag to handle multi-line observations
    match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
    
    if match:
        observation = match.group(1).strip()
        return observation
    raise ValueError(f"No observation found in the text: {text}")


def calculate_perplexity(text: str, model, tokenizer, device: str = 'cuda') -> float:
    """
    Calculate perplexity of a text using a language model.
    
    Args:
        text: The text to calculate perplexity for
        model: The language model
        tokenizer: The tokenizer
        device: Device to use ('cuda' or 'cpu')
        
    Returns:
        Perplexity value
    """
    # Tokenize the text
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=2048)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Calculate loss
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs['input_ids'])
        loss = outputs.loss
        
    # Perplexity is exp(loss)
    perplexity = torch.exp(loss).item()
    
    return perplexity


def load_data(file_path: str) -> List[Dict]:
    """
    Load SFT data from file. Supports JSON, JSONL, and Parquet formats.
    
    Args:
        file_path: Path to the data file
        
    Returns:
        List of data records
    """
    data = []
    
    if file_path.endswith('.parquet'):
        # Load parquet file using pandas
        df = pd.read_parquet(file_path)
        # Convert to list of dictionaries
        data = df.to_dict('records')
    elif file_path.endswith('.jsonl'):
        with open(file_path, 'r') as f:
            for line in f:
                data.append(json.loads(line.strip()))
    else:
        with open(file_path, 'r') as f:
            content = json.load(f)
            if isinstance(content, list):
                data = content
            else:
                data = [content]
    
    return data


def main():
    parser = argparse.ArgumentParser(description='Calculate perplexity on WebShop observations')
    parser.add_argument('--data_file', type=str, required=True, help='Path to SFT data file')
    parser.add_argument('--model_name', type=str, default='gpt2', help='Model to use for perplexity calculation')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', 
                        help='Device to use (cuda/cpu)')
    parser.add_argument('--output_file', type=str, help='Optional output file for detailed results')
    parser.add_argument('--prompt_field', type=str, default='prompt', 
                        help='Field name containing the prompt in the data')
    
    args = parser.parse_args()
    
    print(f"Loading model {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    model = model.to(args.device)
    model.eval()
    
    # Add padding token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Loading data from {args.data_file}...")
    data = load_data(args.data_file)
    print(f"Loaded {len(data)} records")
    
    results = []
    failed_extractions = 0
    
    print("Processing data...")
    for i, record in enumerate(tqdm(data)):
        # Get the prompt text
        if isinstance(record, dict):
            prompt_text = record.get(args.prompt_field, '')
        else:
            prompt_text = str(record)
        prompt_text = prompt_text['content']
        # print(f"Prompt text: {prompt_text}")
        
        # Extract observation
        observation = extract_observation(prompt_text)
        # if 'Welcome to TextWorld, ALFRED!' not in observation:
        #     continue
        if len(observation) < 50:
            continue
        print(f"Observation: {observation}")
        
        if observation:
            # Calculate perplexity
            perplexity = calculate_perplexity(observation, model, tokenizer, args.device)
            print(f"Perplexity: {perplexity}")
            
            results.append({
                'index': i,
                'observation': observation[:100] + '...' if len(observation) > 100 else observation,
                'observation_length': len(observation),
                'perplexity': perplexity
            })
        else:
            failed_extractions += 1
            results.append({
                'index': i,
                'observation': None,
                'observation_length': 0,
                'perplexity': None
            })
    
    # Calculate statistics
    valid_perplexities = [r['perplexity'] for r in results if r['perplexity'] is not None]
    
    if valid_perplexities:
        stats = {
            'total_records': len(data),
            'successful_extractions': len(valid_perplexities),
            'failed_extractions': failed_extractions,
            'mean_perplexity': np.mean(valid_perplexities),
            'median_perplexity': np.median(valid_perplexities),
            'std_perplexity': np.std(valid_perplexities),
            'min_perplexity': np.min(valid_perplexities),
            'max_perplexity': np.max(valid_perplexities),
            'percentile_25': np.percentile(valid_perplexities, 25),
            'percentile_75': np.percentile(valid_perplexities, 75)
        }
        
        print("\n=== Perplexity Statistics ===")
        print(f"Total records: {stats['total_records']}")
        print(f"Successful extractions: {stats['successful_extractions']}")
        print(f"Failed extractions: {stats['failed_extractions']}")
        print(f"\nPerplexity statistics:")
        print(f"  Mean: {stats['mean_perplexity']:.2f}")
        print(f"  Median: {stats['median_perplexity']:.2f}")
        print(f"  Std Dev: {stats['std_perplexity']:.2f}")
        print(f"  Min: {stats['min_perplexity']:.2f}")
        print(f"  Max: {stats['max_perplexity']:.2f}")
        print(f"  25th percentile: {stats['percentile_25']:.2f}")
        print(f"  75th percentile: {stats['percentile_75']:.2f}")
        
        # Save detailed results if requested
        if args.output_file:
            df = pd.DataFrame(results)
            df.to_csv(args.output_file, index=False)
            print(f"\nDetailed results saved to {args.output_file}")
            
            # Also save statistics
            stats_file = args.output_file.replace('.csv', '_stats.json')
            with open(stats_file, 'w') as f:
                json.dump(stats, f, indent=2)
            print(f"Statistics saved to {stats_file}")
    else:
        print("\nNo valid observations found to calculate perplexity!")


if __name__ == "__main__":
    main()


# Alfworld: 1.5b instruct
# === Perplexity Statistics ===
# Total records: 5528
# Successful extractions: 115
# Failed extractions: 0

# Perplexity statistics:
#   Mean: 6.04
#   Median: 5.28
#   Std Dev: 3.28
#   Min: 2.19
#   Max: 15.22
#   25th percentile: 3.25
#   75th percentile: 7.60

# webshop 
Perplexity statistics:
#   Mean: 11.66
#   Median: 8.23
#   Std Dev: 9.69
#   Min: 2.60
#   Max: 66.53
#   25th percentile: 5.46
#   75th percentile: 14.24