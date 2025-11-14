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
import re
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
        
        # Check if we have trajectory UIDs to reconstruct multi-step trajectories
        if (hasattr(batch_output, 'non_tensor_batch') and 
            batch_output.non_tensor_batch and 
            'traj_uid' in batch_output.non_tensor_batch):
            
            print("🔄 Reconstructing multi-step trajectories from flattened batch...")
            trajectories = self._reconstruct_trajectories_from_batch(batch_output, success_info)
            self.collected_trajectories.extend(trajectories)
            print(f"✅ Successfully reconstructed and added {len(trajectories)} trajectories to SFT collection (total: {len(self.collected_trajectories)})")
        else:
            # Fallback to single-step extraction
            batch_size = len(batch_output.batch['input_ids'])
            for i in range(batch_size):
                trajectory = self._extract_trajectory_from_batch(batch_output, i, success_info)
                self.collected_trajectories.append(trajectory)
            print(f"✅ Successfully added {batch_size} single-step trajectories to SFT collection (total: {len(self.collected_trajectories)})")
    
    def _reconstruct_trajectories_from_batch(self, batch_output: DataProto, success_info: Dict = None) -> List[Dict]:
        """
        Reconstruct complete trajectories from flattened multi-step batch.
        
        The batch contains all steps from all trajectories flattened together.
        We need to group by traj_uid and reconstruct the observation sequences.
        
        Args:
            batch_output: Flattened batch containing all trajectory steps
            success_info: Optional success information for filtering
            
        Returns:
            List of complete trajectory dictionaries with full observation sequences
        """
        batch_size = len(batch_output.batch['input_ids'])
        traj_uids = batch_output.non_tensor_batch['traj_uid']
        
        print(f"🔍 Processing {batch_size} steps from flattened batch...")
        
        # Group steps by trajectory UID
        trajectory_steps = {}
        for i in range(batch_size):
            traj_uid = traj_uids[i]
            if traj_uid not in trajectory_steps:
                trajectory_steps[traj_uid] = []
            trajectory_steps[traj_uid].append(i)
        
        print(f"🔍 Found {len(trajectory_steps)} unique trajectories")
        
        # Reconstruct each trajectory
        reconstructed_trajectories = []
        for traj_uid, step_indices in trajectory_steps.items():
            print(f"🔄 Reconstructing trajectory {traj_uid} with {len(step_indices)} steps")
            
            # Sort steps by chronological order (assuming they're already in order, but let's be safe)
            step_indices.sort()
            
            # Extract the complete trajectory
            trajectory = self._reconstruct_single_trajectory(
                batch_output, step_indices, traj_uid, success_info
            )
            
            if trajectory:  # Only add if reconstruction was successful
                reconstructed_trajectories.append(trajectory)
        
        return reconstructed_trajectories
    
    def _reconstruct_single_trajectory(self, batch_output: DataProto, step_indices: List[int], 
                                     traj_uid: int, success_info: Dict = None) -> Dict:
        """
        Reconstruct a single complete trajectory from its step indices.
        
        Args:
            batch_output: The flattened batch
            step_indices: List of indices for this trajectory's steps
            traj_uid: Trajectory unique identifier
            success_info: Optional success information
            
        Returns:
            Complete trajectory dictionary with observation sequence
        """
        messages_list = []
        all_rewards = []
        
        # First, extract observation sequence from the first step (it contains the full sequence)
        if not step_indices:
            raise ValueError(f"❌ No step indices for trajectory {traj_uid}")
        
        first_step_idx = step_indices[0]
        
        # Extract the full observation sequence
        if 'observation_sequence' not in batch_output.non_tensor_batch:
            raise ValueError(f"❌ Missing 'observation_sequence' in non_tensor_batch for trajectory {traj_uid}")
        
        real_states = batch_output.non_tensor_batch['observation_sequence'][first_step_idx]
        if real_states is None:
            raise ValueError(f"❌ observation_sequence is None for trajectory {traj_uid}")
        
        if not isinstance(real_states, list):
            raise ValueError(f"❌ observation_sequence is not a list for trajectory {traj_uid}, got {type(real_states)}")
        
        # Now collect messages and rewards from all steps
        for step_idx in step_indices:
            # Extract messages for this step
            if 'raw_prompt' in batch_output.non_tensor_batch:
                raw_prompt = batch_output.non_tensor_batch['raw_prompt'][step_idx]
                if isinstance(raw_prompt, np.ndarray):
                    raw_prompt = raw_prompt.tolist()
                
                # Add user message
                if isinstance(raw_prompt, list) and len(raw_prompt) > 0:
                    messages_list.extend(raw_prompt)
                
                # Add assistant response
                if 'responses' in batch_output.batch:
                    response_ids = batch_output.batch['responses'][step_idx]
                    response_text = self.tokenizer.decode(response_ids, skip_special_tokens=True)
                    if response_text.strip():
                        messages_list.append({
                            'role': 'assistant',
                            'content': response_text.strip()
                        })
            
            # Extract rewards if available
            if 'rewards' in batch_output.non_tensor_batch:
                reward = batch_output.non_tensor_batch['rewards'][step_idx]
                if reward is not None:
                    all_rewards.append(float(reward))
        
        # Extract success information
        success = False
        if success_info:
            # Try to find success info for this trajectory
            for key, values in success_info.items():
                if isinstance(values, (list, np.ndarray)) and len(values) > step_indices[0]:
                    success = bool(values[step_indices[0]])  # Use first step's success info
                    break
        
        # Extract task information from first step
        task_info = {}
        if step_indices and 'data_source' in batch_output.non_tensor_batch:
            first_step_idx = step_indices[0]
            if first_step_idx < len(batch_output.non_tensor_batch['data_source']):
                task_info['data_source'] = batch_output.non_tensor_batch['data_source'][first_step_idx]
        
        # Strict invariant check: num_observations = num_assistant_messages + 1
        num_assistant_messages = sum(1 for msg in messages_list if msg.get('role') == 'assistant')
        if len(real_states) != num_assistant_messages + 1:
            raise ValueError(f"❌ Invariant violation for trajectory {traj_uid}: "
                            f"Expected {num_assistant_messages + 1} observations for {num_assistant_messages} assistant messages, "
                            f"but got {len(real_states)} observations")
        
        print(f"✅ Trajectory {traj_uid}: {len(real_states)} observations, {len(messages_list)} messages, "
                f"{num_assistant_messages} assistant turns, success={success}")
        
        return {
            'messages_list': messages_list,
            'real_states': real_states,  # This is the key fix - full observation sequence!
            'success': success,
            'rewards': all_rewards if all_rewards else [0.0],
            'final_reward': sum(all_rewards) if all_rewards else 0.0,
            'task_info': task_info,
            'episode_length': len(step_indices),
            'traj_uid': traj_uid
        }

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
        real_states = None
        
        if batch_output.non_tensor_batch:
            # Debug: Print available keys in non_tensor_batch
            print(f"🐛 DEBUG: Available keys in non_tensor_batch for trajectory {index}: {list(batch_output.non_tensor_batch.keys())}")
            
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
            
            # Extract real states (observations) for world model training
            if 'observation_sequence' not in batch_output.non_tensor_batch:
                raise ValueError(f"❌ Missing 'observation_sequence' in non_tensor_batch for trajectory {index}. "
                               f"Available keys: {list(batch_output.non_tensor_batch.keys())}")
            
            obs_sequence = batch_output.non_tensor_batch['observation_sequence'][index]
            if obs_sequence is None:
                raise ValueError(f"❌ observation_sequence is None for trajectory {index}")
            
            if len(obs_sequence) == 0:
                raise ValueError(f"❌ observation_sequence is empty for trajectory {index}")
            
            real_states = obs_sequence 
            assert isinstance(real_states, list), f"❌ real_states is not a list for trajectory {index}"
            print(f"🐛 DEBUG: ✅ Extracted real_states from observation_sequence for trajectory {index}: {len(real_states)} states")
            
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
                'episode_length': len(complete_messages) // 2 if complete_messages else 0,
                'real_states': real_states
            }
            
        return trajectory
    
    
    def save_sft_data(self, require_success: bool = False, test_size: float = 0.1, worldmodel_mode: str = None) -> str:
        """
        Save collected trajectories as SFT training data.
        
        Args:
            require_success: If True, only save successful trajectories
            test_size: Fraction of data to use for validation
            worldmodel_mode: If specified, use world model format conversion ('add_worldmodel_1' or None)
            
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
        if worldmodel_mode == 'add_worldmodel_1':
            print(f"🌍 Using world model format conversion: {worldmodel_mode}")
            sft_data = self.convert_to_sft_format_add_worldmodel_1(trajectories_to_save)
        else:
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
    
    def convert_to_sft_format_add_worldmodel_1(self, trajectories: List[Dict]) -> List[Dict]:
        """Convert trajectories to SFT format by replacing predicted states with real states.
        
        Each turn has the structure with <observation> and <prediction> tags that need
        to be replaced with ground truth states.
        
        Args:
            trajectories: List of trajectory dictionaries containing messages_list and real_states
            
        Returns:
            List of SFT format dictionaries
        """
        sft_data = []
        print(f"🐛 DEBUG: Processing {len(trajectories)} trajectories for world model format conversion")
        
        for idx, traj in enumerate(trajectories):
            # Get the messages and states
            messages = traj.get('messages_list', [])
            real_states = traj.get('real_states', None)
            
            print(f"🐛 DEBUG: Trajectory {idx+1}: {len(messages)} messages, real_states: {'None' if real_states is None else len(real_states)}")
            
            if real_states is None:
                raise ValueError(f"❌ Trajectory {idx+1}: real_states is None")
            
            if not messages:
                raise ValueError(f"❌ Trajectory {idx+1}: messages list is empty")
            
            if not isinstance(real_states, list):
                raise ValueError(f"❌ Trajectory {idx+1}: real_states is not a list, got {type(real_states)}")
            
            # Strict invariant check before processing
            num_assistant_messages = sum(1 for msg in messages if msg.get('role') == 'assistant')
            if len(real_states) != num_assistant_messages + 1:
                raise ValueError(f"❌ Invariant violation for trajectory {idx+1}: "
                               f"Expected {num_assistant_messages + 1} observations for {num_assistant_messages} assistant messages, "
                               f"but got {len(real_states)} observations")
            
            # Create a copy of messages to modify
            modified_messages = []
            for msg in messages:
                modified_messages.append(msg.copy())
            
            replacements_made = 0
            
            # Debug: Show the complete message structure
            print(f"🐛 DEBUG: Complete message structure for trajectory {idx+1}:")
            for i, msg in enumerate(modified_messages):
                role = msg.get('role', 'UNKNOWN')
                content_preview = msg.get('content', '')[:100] + '...' if len(msg.get('content', '')) > 100 else msg.get('content', '')
                print(f"🐛 DEBUG:   Message {i}: role='{role}', content='{content_preview}'")
            
            # Process each turn (user + assistant pairs)
            # Determine if there's a system message at index 0
            has_system_message = len(messages) > 0 and messages[0].get('role') == 'system'
            print(f"🐛 DEBUG: Has system message: {has_system_message}")
            
            for turn_idx in range(len(messages) // 2):
                if has_system_message:
                    # Traditional format: [system, user, assistant, user, assistant, ...]
                    user_idx = 1 + turn_idx * 2  # Skip system message at index 0
                    assistant_idx = user_idx + 1
                else:
                    # No system message: [user, assistant, user, assistant, ...]
                    user_idx = turn_idx * 2
                    assistant_idx = user_idx + 1
                
                cur_state_idx = turn_idx
                next_state_idx = turn_idx + 1
                
                print(f"🐛 DEBUG: Turn {turn_idx+1}: user_idx={user_idx}, assistant_idx={assistant_idx}, cur_state_idx={cur_state_idx}, next_state_idx={next_state_idx}")
                print(f"🐛 DEBUG: Checking if assistant_idx ({assistant_idx}) < len(modified_messages) ({len(modified_messages)})")
                
                if assistant_idx < len(modified_messages):
                    print(f"🐛 DEBUG: Message at assistant_idx ({assistant_idx}) has role: '{modified_messages[assistant_idx].get('role', 'UNKNOWN')}'")
                else:
                    print(f"🐛 DEBUG: assistant_idx ({assistant_idx}) is out of bounds for modified_messages (length: {len(modified_messages)})")
                
                # Process assistant message if it exists
                if assistant_idx < len(modified_messages) and modified_messages[assistant_idx]['role'] == 'assistant':
                    assistant_message = modified_messages[assistant_idx]['content']
                    original_message = assistant_message
                    
                    # Debug: Show the actual assistant message content
                    print(f"🐛 DEBUG: Turn {turn_idx+1} assistant message content (first 300 chars):")
                    print(f"🐛 DEBUG: '{assistant_message[:300]}...'")
                    
                    # Check if the message contains the expected tags
                    has_obs_tags = '<observation>' in assistant_message and '</observation>' in assistant_message
                    has_pred_tags = '<prediction>' in assistant_message and '</prediction>' in assistant_message
                    has_think_tags = '<think>' in assistant_message and '</think>' in assistant_message
                    has_next_state_tags = '<next_state>' in assistant_message and '</next_state>' in assistant_message
                    print(f"🐛 DEBUG: Turn {turn_idx+1} - Has observation tags: {has_obs_tags}, Has prediction tags: {has_pred_tags}")
                    print(f"🐛 DEBUG: Turn {turn_idx+1} - Has think tags: {has_think_tags}, Has next_state tags: {has_next_state_tags}")
                    
                    # Get real states for replacement
                    cur_state = ""
                    if cur_state_idx < len(real_states) and real_states[cur_state_idx] is not None:
                        cur_state = str(real_states[cur_state_idx])
                        print(f"🐛 DEBUG: Using real_states[{cur_state_idx}] for current observation")
                    else:
                        print(f"🐛 DEBUG: WARNING - No current state available at index {cur_state_idx} (real_states length: {len(real_states)})")
                    
                    next_state = ""
                    if next_state_idx < len(real_states) and real_states[next_state_idx] is not None:
                        next_state = str(real_states[next_state_idx])
                        print(f"🐛 DEBUG: Using real_states[{next_state_idx}] for next prediction")
                    else:
                        print(f"🐛 DEBUG: WARNING - No next state available at index {next_state_idx} (real_states length: {len(real_states)})")
                        # For empty predictions, we might want to use an empty string or a default message
                        next_state = ""  # Keep empty for now
                    
                    print(f"🐛 DEBUG: States for turn {turn_idx+1}:")
                    print(f"🐛 DEBUG:   cur_state (len={len(cur_state)}): '{cur_state[:100]}...'")
                    print(f"🐛 DEBUG:   next_state (len={len(next_state)}): '{next_state[:100]}...'")
                    
                    # Replace predicted states with real states using regex
                    def replace_states_in_message(msg, obs, pred):
                        original_msg = msg
                        replacements = 0
                        
                        # Replace observation content
                        obs_pattern = r'(<observation>)(.*?)(</observation>)'
                        obs_matches = re.findall(obs_pattern, msg, flags=re.DOTALL)
                        if obs_matches:
                            print(f"🐛 DEBUG: Found {len(obs_matches)} observation tags")
                            for i, match in enumerate(obs_matches):
                                original_content = match[1].strip()
                                print(f"🐛 DEBUG: Observation {i+1} original content: '{original_content[:100]}...' (length: {len(original_content)})")
                                if not original_content:
                                    print(f"🐛 DEBUG: WARNING - Observation {i+1} is empty!")
                            msg = re.sub(obs_pattern, f'\\1{obs}\\3', msg, flags=re.DOTALL)
                            replacements += len(obs_matches)
                            print(f"🐛 DEBUG: Replaced observation content with: '{obs[:100]}...' (length: {len(obs)})")
                        
                        # Replace prediction content
                        pred_pattern = r'(<prediction>)(.*?)(</prediction>)'
                        pred_matches = re.findall(pred_pattern, msg, flags=re.DOTALL)
                        if pred_matches:
                            print(f"🐛 DEBUG: Found {len(pred_matches)} prediction tags")
                            for i, match in enumerate(pred_matches):
                                original_content = match[1].strip()
                                print(f"🐛 DEBUG: Prediction {i+1} original content: '{original_content[:100]}...' (length: {len(original_content)})")
                                if not original_content:
                                    print(f"🐛 DEBUG: WARNING - Prediction {i+1} is EMPTY! Will replace with ground truth.")
                            msg = re.sub(pred_pattern, f'\\1{pred}\\3', msg, flags=re.DOTALL)
                            replacements += len(pred_matches)
                            print(f"🐛 DEBUG: Replaced prediction content with: '{pred[:100]}...' (length: {len(pred)})")
                        
                        if replacements > 0:
                            print(f"🐛 DEBUG: Made {replacements} replacements in message")
                        else:
                            print(f"🐛 DEBUG: No observation/prediction tags found in message")
                        
                        return msg, replacements
                    
                    # Replace states in the assistant message
                    assistant_message, turn_replacements = replace_states_in_message(assistant_message, cur_state, next_state)
                    replacements_made += turn_replacements
                    
                    # Update the assistant message
                    modified_messages[assistant_idx]['content'] = assistant_message
                    
                    # Show before/after comparison for debugging
                    if turn_replacements > 0:
                        print(f"🐛 DEBUG: Turn {turn_idx+1} replacement summary:")
                        print(f"🐛 DEBUG: BEFORE: '{original_message[:200]}...'")
                        print(f"🐛 DEBUG: AFTER:  '{assistant_message[:200]}...'")
                    else:
                        print(f"🐛 DEBUG: Turn {turn_idx+1}: No replacements made")
            
            print(f"🐛 DEBUG: Trajectory {idx+1} total replacements: {replacements_made}")
            
            # Add to SFT data
            sft_data.append({
                'id': idx + 1,
                'messages_list': modified_messages,
                'success': traj.get('success', False),
                'final_reward': traj.get('final_reward', 0.0),
                'episode_length': traj.get('episode_length', 0),
                'task_info': traj.get('task_info', {})
            })
        
        print(f"🐛 DEBUG: World model conversion complete. Generated {len(sft_data)} SFT samples from {len(trajectories)} trajectories")
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
