"""
Script for generating REAL AlfWorld SFT trajectories using the actual environment.
This version uses your existing agent system to interact with real AlfWorld environments.
"""

import os
import json
import numpy as np
import time
import logging
import sys
import re
import argparse
from datetime import datetime
from transformers import AutoTokenizer
import pandas as pd
from sklearn.model_selection import train_test_split
from omegaconf import DictConfig, OmegaConf
import ray
from functools import partial


def init_logging(to_file_only=False, log_dir="log"):
    """Set up logging."""
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"alfworld_real_sft_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log")

    # Clear existing handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    handlers = [logging.FileHandler(log_file, mode='w', encoding='utf-8')]
    if not to_file_only:
        handlers.append(logging.StreamHandler(sys.__stdout__))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=handlers
    )
    return logging.getLogger(__name__)


def setup_environment(config):
    """Set up the real AlfWorld environment using your existing infrastructure."""
    
    # Import your existing components
    from agent_system.environments.env_manager import AlfWorldEnvironmentManager, make_envs
    from agent_system.environments.env_package.alfworld import alfworld_projection
    
    # Create environments using your existing make_envs function
    try:
        envs, val_envs = make_envs(config)
        logger.info(f"✅ Successfully created AlfWorld environments")
        return envs, val_envs
    except Exception as e:
        logger.error(f"❌ Failed to create environments: {e}")
        raise


def setup_agent_rollout(config, tokenizer):
    """Set up the agent rollout system for trajectory generation."""
    
    try:
        # Import rollout components
        from verl.workers.rollout.vllm_rollout.vllm_rollout_spmd import vLLMRollout
        from agent_system.multi_turn_rollout.rollout_loop import TrajectoryCollector
        
        # Initialize trajectory collector
        trajectory_collector = TrajectoryCollector(config, tokenizer)
        
        # Initialize vLLM rollout worker
        rollout_worker = vLLMRollout(
            model_path=config.actor_rollout_ref.model.path,
            config=config.actor_rollout_ref.rollout,
            tokenizer=tokenizer,
            model_hf_config=None
        )
        
        logger.info(f"✅ Successfully initialized agent rollout system")
        return trajectory_collector, rollout_worker
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize agent system: {e}")
        raise


def run_real_trajectories(envs, trajectory_collector, rollout_worker, tokenizer, num_episodes, max_steps, require_success=False):
    """Run real AlfWorld episodes using TrajectoryCollector like PPO validation."""
    
    all_trajectories = []
    successful_trajectories = []
    failed_trajectories = []
    
    logger.info(f"🚀 Starting trajectory collection using TrajectoryCollector...")
    logger.info(f"Episodes: {num_episodes}, Max steps: {max_steps}, Require success: {require_success}")
    
    # Create initial generation batch (empty prompts to start episodes)
    from verl import DataProto
    import torch
    
    # Create minimal initial batch for starting episodes
    # This will be processed by TrajectoryCollector.preprocess_batch()
    batch_size = min(num_episodes, 4)  # Process in batches
    
    for batch_start in range(0, num_episodes, batch_size):
        batch_end = min(batch_start + batch_size, num_episodes)
        current_batch_size = batch_end - batch_start
        
        logger.info(f"Processing episodes {batch_start + 1}-{batch_end}")
        
        try:
            # Create initial generation batch
            # Use empty prompts - the environment will provide the actual observations
            empty_input_ids = torch.ones((current_batch_size, 1), dtype=torch.long) * tokenizer.bos_token_id
            empty_attention_mask = torch.ones((current_batch_size, 1), dtype=torch.long)
            empty_position_ids = torch.zeros((current_batch_size, 1), dtype=torch.long)
            
            gen_batch = DataProto(
                batch={
                    'input_ids': empty_input_ids,
                    'attention_mask': empty_attention_mask,
                    'position_ids': empty_position_ids
                },
                non_tensor_batch=None,
                meta_info={
                    'eos_token_id': tokenizer.eos_token_id,
                    'pad_token_id': tokenizer.pad_token_id,
                    'recompute_log_prob': False,
                    'do_sample': True,
                    'validate': True,
                }
            )
            
            # Use TrajectoryCollector.multi_turn_loop() like PPO validation does!
            # This is the key alignment with PPO validation
            batch_output = trajectory_collector.multi_turn_loop(
                gen_batch=gen_batch,
                actor_rollout_wg=rollout_worker,
                envs=envs,
                is_train=False,  # Same as validation
            )
            
            # Extract trajectories from batch_output
            # The batch_output contains all the trajectory data we need
            batch_trajectories = extract_trajectories_from_batch(batch_output, tokenizer)
            
            for traj in batch_trajectories:
                all_trajectories.append(traj)
                
                # Check if trajectory was successful
                if traj.get('success', False):
                    successful_trajectories.append(traj)
                else:
                    failed_trajectories.append(traj)
            
        except Exception as e:
            logger.error(f"Error processing batch {batch_start}-{batch_end}: {e}")
            continue
    
    # Apply success filtering if required
    if require_success:
        final_trajectories = successful_trajectories
        logger.info(f"🎯 Rejection sampling: kept {len(final_trajectories)} successful out of {len(all_trajectories)} total")
    else:
        final_trajectories = all_trajectories
        logger.info(f"📊 Mixed dataset: {len(successful_trajectories)} successful, {len(failed_trajectories)} failed")
    
    return final_trajectories


def extract_trajectories_from_batch(batch_output, tokenizer):
    """Extract trajectory data from TrajectoryCollector batch output."""
    import torch
    
    trajectories = []
    
    # The batch_output from multi_turn_loop contains all trajectory information
    batch_size = len(batch_output.batch['input_ids'])
    
    for i in range(batch_size):
        try:
            # Extract conversation history from the batch
            # The batch contains prompts and responses
            prompt_ids = batch_output.batch['input_ids'][i]
            response_ids = batch_output.batch['responses'][i] if 'responses' in batch_output.batch else None
            
            # Decode the full conversation
            if response_ids is not None:
                full_conversation = tokenizer.decode(torch.cat([prompt_ids, response_ids]), skip_special_tokens=True)
            else:
                full_conversation = tokenizer.decode(prompt_ids, skip_special_tokens=True)
            
            # Extract success information from non_tensor_batch if available
            success = False
            rewards = []
            if hasattr(batch_output, 'non_tensor_batch') and batch_output.non_tensor_batch:
                # Try to extract success from environment info
                if 'success' in batch_output.non_tensor_batch:
                    success = batch_output.non_tensor_batch['success'][i]
                # Try to extract rewards
                if 'rewards' in batch_output.non_tensor_batch:
                    rewards = batch_output.non_tensor_batch['rewards'][i]
            
            # Create messages list by parsing the conversation
            # This is a simplified approach - you might need to enhance this
            # based on your specific conversation format
            messages_list = parse_conversation_to_messages(full_conversation)
            
            trajectory = {
                'messages_list': messages_list,
                'rewards': rewards if rewards else [0.0],
                'task_description': extract_task_from_conversation(full_conversation),
                'success': success,
                'final_reward': sum(rewards) if rewards else 0.0,
                'episode_length': len(messages_list) // 2,  # Approximate
                'actions_taken': extract_actions_from_conversation(full_conversation)
            }
            
            trajectories.append(trajectory)
            
        except Exception as e:
            logger.warning(f"Failed to extract trajectory {i}: {e}")
            continue
    
    return trajectories


def parse_conversation_to_messages(conversation_text):
    """Parse a full conversation text into messages list."""
    # This is a simplified parser - you might need to enhance it
    # based on your tokenizer's chat template format
    messages = []
    
    # Split by common chat template markers
    # This is a basic implementation - adjust based on your model's format
    if "<|im_start|>" in conversation_text:
        # Qwen format
        parts = conversation_text.split("<|im_start|>")
        for part in parts[1:]:  # Skip first empty part
            if "<|im_end|>" in part:
                role_content = part.split("<|im_end|>")[0]
                if "\n" in role_content:
                    role, content = role_content.split("\n", 1)
                    messages.append({'role': role.strip(), 'content': content.strip()})
    else:
        # Fallback: treat as single user message
        messages.append({'role': 'user', 'content': conversation_text})
    
    return messages


def extract_task_from_conversation(conversation_text):
    """Extract task description from conversation."""
    task_start = conversation_text.find('Your task is to: ')
    if task_start != -1:
        task_end = conversation_text.find('\n', task_start)
        if task_end == -1:
            task_end = len(conversation_text)
        return conversation_text[task_start + len('Your task is to: '):task_end].strip()
    return "Unknown task"


def extract_actions_from_conversation(conversation_text):
    """Extract actions from conversation."""
    import re
    actions = re.findall(r'<action>\s*(.*?)\s*</action>', conversation_text, re.DOTALL | re.IGNORECASE)
    return [action.strip() for action in actions]


def convert_to_sft_format(trajectories, tokenizer):
    """Convert real trajectories to SFT format."""
    sft_data = []
    
    for idx, traj in enumerate(trajectories):
        messages = traj['messages_list']
        
        if not messages or len(messages) == 0:
            continue
        
        sft_data.append({
            'id': idx + 1,
            'messages_list': messages,
            'rewards': traj.get('rewards', []),
            'task_description': traj.get('task_description', ''),
            'success': traj.get('success', False),
            'final_reward': traj.get('final_reward', 0.0),
            'episode_length': traj.get('episode_length', 0),
            'actions_taken': traj.get('actions_taken', [])
        })
    
    return sft_data


def create_training_rows(sft_data):
    """Convert SFT data to training rows format."""
    rows = []
    DEFAULT_DATA_SOURCE = "alfworld_real"
    DEFAULT_ABILITY = "agent"
    DEFAULT_REWARD_MODEL = "{'ground_truth': {'numbers': [], 'target': 0}, 'style': 'rule'}"
    DEFAULT_EXTRA_INFO = "{'index': 0, 'split': 'train'}"

    for sample_idx, sample in enumerate(sft_data):
        messages = sample["messages_list"]
        
        for i, msg in enumerate(messages):
            if msg["role"] == "assistant":
                # Get all messages up to this point as prompt
                prompt_list = messages[:i]
                rows.append({
                    'data_source': DEFAULT_DATA_SOURCE,
                    'prompt': prompt_list,
                    'response': msg['content'],
                    'ability': DEFAULT_ABILITY,
                    'reward_model': DEFAULT_REWARD_MODEL,
                    'extra_info': DEFAULT_EXTRA_INFO,
                })
    
    return rows


def main():
    global logger
    
    parser = argparse.ArgumentParser(description="Generate REAL AlfWorld SFT trajectories")
    parser.add_argument('--model_path', default='Qwen/Qwen2.5-1.5B-Instruct', help='Path to the model')
    parser.add_argument('--num_episodes', type=int, default=20, help='Number of episodes to run')
    parser.add_argument('--max_steps', type=int, default=30, help='Maximum steps per episode')
    parser.add_argument('--output_dir', default='alfworld_real_sft_data', help='Output directory')
    parser.add_argument('--log_dir', default='logs', help='Log directory')
    parser.add_argument('--require_success', action='store_true', help='Only keep successful trajectories')
    parser.add_argument('--env_num', type=int, default=4, help='Number of parallel environments')
    parser.add_argument('--config_path', default='config_alfworld.yaml', help='Path to config file')
    
    args = parser.parse_args()
    
    # Initialize logging
    logger = init_logging(to_file_only=False, log_dir=args.log_dir)
    
    logger.info(f"🚀 Starting REAL AlfWorld SFT trajectory generation")
    logger.info(f"Model path: {args.model_path}")
    logger.info(f"Episodes: {args.num_episodes}")
    logger.info(f"Max steps: {args.max_steps}")
    logger.info(f"Require success: {args.require_success}")
    logger.info(f"Environment processes: {args.env_num}")
    
    try:
        # Load configuration
        if os.path.exists(args.config_path):
            config = OmegaConf.load(args.config_path)
        else:
            # Create minimal config
            config = OmegaConf.create({
                'env': {
                    'env_name': 'alfworld/AlfredTWEnv',
                    'seed': 0,
                    'max_steps': args.max_steps,
                    'resources_per_worker': {'num_cpus': 0.01}
                },
                'data': {
                    'train_batch_size': args.env_num,
                    'val_batch_size': args.env_num,
                    'max_prompt_length': 2048,
                    'max_response_length': 512,
                    'return_raw_chat': True
                },
                'actor_rollout_ref': {
                    'model': {'path': args.model_path},
                    'rollout': {
                        'tensor_model_parallel_size': 1,
                        'gpu_memory_utilization': 0.4,
                        'val_kwargs': {'temperature': 0.4, 'do_sample': True}
                    }
                }
            })
        
        logger.info(f"✅ Configuration loaded")
        
        # Initialize tokenizer
        logger.info(f"Loading tokenizer from {args.model_path}")
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        logger.info(f"✅ Tokenizer loaded")
        
        # Set up environment
        logger.info(f"Setting up AlfWorld environment...")
        envs, val_envs = setup_environment(config)
        
        # Set up agent rollout system
        logger.info(f"Setting up agent rollout system...")
        trajectory_collector, rollout_worker = setup_agent_rollout(config, tokenizer)
        
        # Run real trajectories
        logger.info(f"🎮 Running real AlfWorld episodes...")
        start_time = time.time()
        
        trajectories = run_real_trajectories(
            envs=envs,
            trajectory_collector=trajectory_collector,
            rollout_worker=rollout_worker,
            tokenizer=tokenizer,
            num_episodes=args.num_episodes,
            max_steps=args.max_steps,
            require_success=args.require_success
        )
        
        generation_time = time.time() - start_time
        logger.info(f"✅ Trajectory generation completed in {generation_time:.2f} seconds")
        
        if len(trajectories) == 0:
            logger.error("❌ No trajectories generated!")
            return
        
        # Convert to SFT format
        logger.info("Converting trajectories to SFT format...")
        sft_data = convert_to_sft_format(trajectories, tokenizer)
        logger.info(f"Converted {len(trajectories)} trajectories to {len(sft_data)} SFT examples")
        
        # Create output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"{args.output_dir}_{timestamp}"
        os.makedirs(output_dir, exist_ok=True)
        
        # Save raw trajectories
        raw_output_file = os.path.join(output_dir, f"real_trajectories_{timestamp}.json")
        with open(raw_output_file, 'w') as f:
            json.dump(sft_data, f, indent=2)
        logger.info(f"Saved raw trajectories to {raw_output_file}")
        
        # Create training rows
        logger.info("Creating training rows...")
        rows = create_training_rows(sft_data)
        logger.info(f"Created {len(rows)} training rows from {len(sft_data)} SFT samples")
        
        if len(rows) == 0:
            logger.error("No training rows created!")
            return
        
        # Create DataFrame and split
        df = pd.DataFrame(rows)
        train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, shuffle=True)
        
        # Save files
        train_csv = os.path.join(output_dir, 'alfworld_real_train.csv')
        val_csv = os.path.join(output_dir, 'alfworld_real_val.csv')
        train_parquet = os.path.join(output_dir, 'alfworld_real_train.parquet')
        val_parquet = os.path.join(output_dir, 'alfworld_real_val.parquet')
        
        train_df.to_csv(train_csv, index=False)
        val_df.to_csv(val_csv, index=False)
        train_df.to_parquet(train_parquet, index=False)
        val_df.to_parquet(val_parquet, index=False)
        
        # Calculate statistics
        successful_count = sum(1 for traj in trajectories if traj.get('success', False))
        success_rate = successful_count / len(trajectories) if trajectories else 0
        avg_episode_length = np.mean([traj.get('episode_length', 0) for traj in trajectories]) if trajectories else 0
        avg_reward = np.mean([traj.get('final_reward', 0) for traj in trajectories]) if trajectories else 0
        
        # Print summary
        print(f"\n{'='*60}")
        print(f"REAL ALFWORLD SFT GENERATION SUMMARY")
        print(f"{'='*60}")
        print(f"Episodes Run: {len(trajectories)}")
        print(f"Successful Episodes: {successful_count}")
        print(f"Success Rate: {success_rate:.1%}")
        print(f"Average Episode Length: {avg_episode_length:.1f} steps")
        print(f"Average Final Reward: {avg_reward:.3f}")
        print(f"Generation Time: {generation_time:.2f} seconds")
        print(f"")
        print(f"Training Data:")
        print(f"  SFT Examples: {len(sft_data)}")
        print(f"  Training Rows: {len(rows)}")
        print(f"  Train Samples: {len(train_df)}")
        print(f"  Validation Samples: {len(val_df)}")
        print(f"")
        print(f"Output Directory: {output_dir}")
        print(f"{'='*60}")
        
        if args.require_success and success_rate == 1.0:
            print(f"✅ Rejection sampling successful - all trajectories are successful!")
        elif success_rate > 0.5:
            print(f"✅ Good success rate achieved: {success_rate:.1%}")
        else:
            print(f"⚠️  Low success rate: {success_rate:.1%} - consider adjusting parameters")
        
        print(f"\n🚀 REAL trajectory data ready for SFT training!")
        
        # Clean up
        logger.info("Cleaning up...")
        if ray.is_initialized():
            ray.shutdown()
        
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        if ray.is_initialized():
            ray.shutdown()
        raise


if __name__ == "__main__":
    main()
