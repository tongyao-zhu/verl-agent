# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Utility functions for collecting and saving SFT data from validation trajectories.
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any
from sklearn.model_selection import train_test_split
import torch
from verl import DataProto


class SFTDataCollector:
    """Collects and formats trajectory data for SFT training."""
    
    def __init__(self, tokenizer, output_dir=None):
        """
        Initialize SFT data collector.
        
        Args:
            tokenizer: The tokenizer used for decoding
            output_dir: Directory to save SFT data (if None, uses default)
        """
        self.tokenizer = tokenizer
        self.collected_trajectories = []
        self.output_dir = output_dir or f"sft_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
    def add_validation_batch(self, batch_output: DataProto, success_info: Dict = None):
        """
        Add a validation batch to the SFT collection.
        
        Args:
            batch_output: The batch output from validation containing trajectories
            success_info: Optional success information for filtering
        """
        if 'input_ids' not in batch_output.batch:
            raise ValueError("❌ Missing 'input_ids' in batch_output.batch")
        
        batch_size = len(batch_output.batch['input_ids'])
        
        for i in range(batch_size):
            trajectory = self._extract_trajectory_from_batch(batch_output, i, success_info)
            self.collected_trajectories.append(trajectory)
            
        print(f"✅ Successfully added {batch_size} trajectories to SFT collection (total: {len(self.collected_trajectories)})")
    
    def _extract_trajectory_from_batch(self, batch_output: DataProto, index: int, success_info: Dict = None) -> Dict:
        """Extract a single trajectory from batch output."""
        # Strict validation - no fallbacks
        if not hasattr(batch_output, 'non_tensor_batch'):
            raise ValueError(f"❌ Batch output missing non_tensor_batch at index {index}")
        
        if 'raw_prompt' not in batch_output.non_tensor_batch:
            raise ValueError(f"❌ Missing 'raw_prompt' in non_tensor_batch at index {index}")
        
        raw_prompt = batch_output.non_tensor_batch['raw_prompt'][index]
        
        # Handle numpy array conversion
        if isinstance(raw_prompt, np.ndarray):
            if raw_prompt.size == 0:
                raise ValueError(f"❌ raw_prompt[{index}] is empty numpy array")
            # Convert numpy array to list - it should contain message dicts
            raw_prompt = raw_prompt.tolist()
        
        if not isinstance(raw_prompt, list) or len(raw_prompt) == 0:
            raise ValueError(f"❌ raw_prompt[{index}] is not a valid messages list: {type(raw_prompt)}, len={len(raw_prompt) if hasattr(raw_prompt, '__len__') else 'N/A'}")
        
        # Validate message structure
        for i, msg in enumerate(raw_prompt):
            if not isinstance(msg, dict):
                raise ValueError(f"❌ raw_prompt[{index}][{i}] is not a dict: {type(msg)}")
            if 'role' not in msg or 'content' not in msg:
                raise ValueError(f"❌ raw_prompt[{index}][{i}] missing 'role' or 'content': {msg.keys()}")
        
        original_messages = raw_prompt
        
        # Extract response - strict validation
        response_text = ""
        if 'responses' in batch_output.batch:
            response_ids = batch_output.batch['responses'][index]
            response_text = self.tokenizer.decode(response_ids, skip_special_tokens=True)
        elif 'response_ids' in batch_output.batch:
            response_ids = batch_output.batch['response_ids'][index]
            response_text = self.tokenizer.decode(response_ids, skip_special_tokens=True)
        else:
            raise ValueError(f"❌ Missing 'responses' or 'response_ids' in batch at index {index}")
        
        # Extract additional information from non_tensor_batch
        success = False
        rewards = []
        task_info = {}
        
        if batch_output.non_tensor_batch:
            # Extract success information
            if 'success' in batch_output.non_tensor_batch:
                success = batch_output.non_tensor_batch['success'][index]
            elif success_info and 'success' in success_info:
                success = success_info['success'][index] if isinstance(success_info['success'], list) else success_info['success']
            
            # Extract rewards
            if 'rewards' in batch_output.non_tensor_batch:
                rewards = batch_output.non_tensor_batch['rewards'][index]
                # Ensure rewards is always a list
                if not isinstance(rewards, (list, tuple, np.ndarray)):
                    rewards = [rewards]
                elif isinstance(rewards, np.ndarray):
                    rewards = rewards.tolist()
            elif 'reward' in batch_output.non_tensor_batch:
                reward_val = batch_output.non_tensor_batch['reward'][index]
                rewards = [reward_val]
            
            # Extract task information
            if 'data_source' in batch_output.non_tensor_batch:
                task_info['data_source'] = batch_output.non_tensor_batch['data_source'][index]
            
            # Create complete messages list by adding the response to original messages
            complete_messages = original_messages.copy()
            if response_text.strip():
                # Add the assistant response
                complete_messages.append({
                    'role': 'assistant',
                    'content': response_text.strip()
                })
            
            trajectory = {
                'messages_list': complete_messages,
                'success': success,
                'rewards': rewards if rewards else [0.0],
                'final_reward': sum(rewards) if rewards else 0.0,
                'task_info': task_info,
                'episode_length': len(complete_messages) // 2 if complete_messages else 0
            }
            
        return trajectory
    
    
    def save_sft_data(self, require_success: bool = False, test_size: float = 0.1) -> str:
        """
        Save collected trajectories as SFT training data.
        
        Args:
            require_success: If True, only save successful trajectories
            test_size: Fraction of data to use for validation
            
        Returns:
            Path to the output directory
        """
        if not self.collected_trajectories:
            print("Warning: No trajectories collected for SFT data")
            return self.output_dir
        
        print(f"💾 Saving SFT data from {len(self.collected_trajectories)} collected trajectories...")
        
        # Filter trajectories if needed
        trajectories_to_save = self.collected_trajectories
        if require_success:
            trajectories_to_save = [t for t in self.collected_trajectories if t.get('success', False)]
            print(f"🎯 Filtered to {len(trajectories_to_save)} successful trajectories")
        
        if not trajectories_to_save:
            print("Warning: No trajectories to save after filtering")
            return self.output_dir
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Save raw trajectories
        raw_file = os.path.join(self.output_dir, 'raw_trajectories.json')
        with open(raw_file, 'w') as f:
            json.dump(trajectories_to_save, f, indent=2, default=str)
        
        # Convert to SFT format
        sft_data = self._convert_to_sft_format(trajectories_to_save)
        
        # Create training rows
        training_rows = self._create_training_rows(sft_data)
        
        if not training_rows:
            print("Warning: No training rows created")
            return self.output_dir
        
        # Split into train/val
        if len(training_rows) > 1:
            train_rows, val_rows = train_test_split(training_rows, test_size=test_size, random_state=42)
        else:
            train_rows = training_rows
            val_rows = []
        
        # Save as CSV and Parquet
        train_df = pd.DataFrame(train_rows)
        val_df = pd.DataFrame(val_rows) if val_rows else pd.DataFrame(columns=train_df.columns)
        
        # Save files
        train_csv = os.path.join(self.output_dir, 'train.csv')
        val_csv = os.path.join(self.output_dir, 'val.csv')
        train_parquet = os.path.join(self.output_dir, 'train.parquet')
        val_parquet = os.path.join(self.output_dir, 'val.parquet')
        
        train_df.to_csv(train_csv, index=False)
        val_df.to_csv(val_csv, index=False)
        train_df.to_parquet(train_parquet, index=False)
        val_df.to_parquet(val_parquet, index=False)
        
        # Calculate statistics
        successful_count = sum(1 for t in trajectories_to_save if t.get('success', False))
        success_rate = successful_count / len(trajectories_to_save) if trajectories_to_save else 0
        avg_reward = sum(t.get('final_reward', 0) for t in trajectories_to_save) / len(trajectories_to_save) if trajectories_to_save else 0
        
        # Print summary
        print(f"\n{'='*60}")
        print(f"SFT DATA COLLECTION SUMMARY")
        print(f"{'='*60}")
        print(f"Total Trajectories Collected: {len(self.collected_trajectories)}")
        print(f"Trajectories Used for SFT: {len(trajectories_to_save)}")
        print(f"Successful Trajectories: {successful_count}")
        print(f"Success Rate: {success_rate:.1%}")
        print(f"Average Final Reward: {avg_reward:.3f}")
        print(f"")
        print(f"Training Data:")
        print(f"  Training Rows: {len(train_rows)}")
        print(f"  Validation Rows: {len(val_rows)}")
        print(f"  Total Samples: {len(training_rows)}")
        print(f"")
        print(f"Output Directory: {self.output_dir}")
        print(f"Files Created:")
        print(f"  - raw_trajectories.json")
        print(f"  - train.csv, train.parquet")
        print(f"  - val.csv, val.parquet")
        print(f"{'='*60}")
        
        return self.output_dir
    
    def _convert_to_sft_format(self, trajectories: List[Dict]) -> List[Dict]:
        """Convert trajectories to SFT format."""
        sft_data = []
        
        for idx, traj in enumerate(trajectories):
            messages = traj.get('messages_list', [])
            
            if not messages:
                continue
            
            sft_data.append({
                'id': idx + 1,
                'messages_list': messages,
                'success': traj.get('success', False),
                'final_reward': traj.get('final_reward', 0.0),
                'episode_length': traj.get('episode_length', 0),
                'task_info': traj.get('task_info', {})
            })
        
        return sft_data
    
    def _create_training_rows(self, sft_data: List[Dict]) -> List[Dict]:
        """Convert SFT data to training rows format compatible with both single-turn and multi-turn SFT."""
        rows = []
        DEFAULT_DATA_SOURCE = "agent_sft_collection"
        DEFAULT_ABILITY = "agent"
        DEFAULT_REWARD_MODEL = "{'ground_truth': {'numbers': [], 'target': 0}, 'style': 'rule'}"
        DEFAULT_EXTRA_INFO = "{'index': 0, 'split': 'train'}"
        
        for sample_idx, sample in enumerate(sft_data):
            messages = sample.get("messages_list", [])
            
            for i, msg in enumerate(messages):
                if msg["role"] == "assistant":
                    # Get all messages up to this point (including current assistant message)
                    conversation_messages = messages[:i+1]
                    
                    # For multi-turn compatibility: store the full conversation as messages
                    # For single-turn compatibility: create prompt/response structure
                    prompt_messages = messages[:i]  # Messages before assistant response
                    
                    rows.append({
                        'data_source': DEFAULT_DATA_SOURCE,
                        'messages': conversation_messages,  # For multi-turn SFT
                        'prompt_data': {'content': self._extract_raw_prompt_content(prompt_messages)},  # For single-turn SFT
                        'response_data': {'content': msg['content']},  # For single-turn SFT
                        'ability': DEFAULT_ABILITY,
                        'reward_model': DEFAULT_REWARD_MODEL,
                        'extra_info': DEFAULT_EXTRA_INFO,
                    })
        
        return rows
    
    def _extract_raw_prompt_content(self, messages: List[Dict]) -> str:
        """Extract raw content for single-turn SFT (let the trainer apply chat template)."""
        if not messages:
            return ""
        
        # For single-turn, we want the last user message content
        # The SFT trainer will wrap this in proper chat template format
        for msg in reversed(messages):
            if msg.get('role') == 'user':
                content = msg.get('content', '')
                # Clean up any existing chat template formatting
                content = content.strip()
                return content
        
        # Fallback: concatenate all user messages
        user_contents = [msg.get('content', '') for msg in messages if msg.get('role') == 'user']
        return " ".join(user_contents) if user_contents else ""
    
    def _messages_to_prompt_string(self, messages: List[Dict]) -> str:
        """Convert messages list to a formatted prompt string (for debugging/reference)."""
        prompt_parts = []
        for msg in messages:
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            prompt_parts.append(f"{role}: {content}")
        return "\n\n".join(prompt_parts)
