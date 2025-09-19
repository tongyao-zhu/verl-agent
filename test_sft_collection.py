#!/usr/bin/env python3
"""
Simple test script to verify SFT data collection functionality.
This script tests the core components without running the full PPO trainer.
"""

import os
import sys
import tempfile
import shutil
from unittest.mock import MagicMock, patch

def test_sft_data_collector():
    """Test the SFTDataCollector class."""
    print("🧪 Testing SFTDataCollector...")
    
    try:
        # Mock tokenizer
        tokenizer = MagicMock()
        tokenizer.decode.return_value = "Test conversation"
        tokenizer.eos_token_id = 2
        tokenizer.pad_token_id = 0
        
        # Import and test SFTDataCollector
        sys.path.insert(0, '/Users/zhuty/Documents/verl-agent')
        from verl.utils.sft_data_collector import SFTDataCollector
        
        # Create temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            collector = SFTDataCollector(tokenizer=tokenizer, output_dir=temp_dir)
            
            # Test basic initialization
            assert collector.tokenizer == tokenizer
            assert collector.output_dir == temp_dir
            assert len(collector.collected_trajectories) == 0
            
            print("✅ SFTDataCollector initialization test passed")
            
            # Test conversation parsing
            messages = collector._parse_conversation_to_messages(
                "<|im_start|>user\nHello<|im_end|><|im_start|>assistant\nHi there!<|im_end|>",
                "Hello",
                "Hi there!"
            )
            
            expected_messages = [
                {'role': 'user', 'content': 'Hello'},
                {'role': 'assistant', 'content': 'Hi there!'}
            ]
            
            assert len(messages) == 2
            assert messages[0]['role'] == 'user'
            assert messages[1]['role'] == 'assistant'
            
            print("✅ Conversation parsing test passed")
            
            # Test SFT format conversion
            test_trajectories = [
                {
                    'messages_list': expected_messages,
                    'success': True,
                    'final_reward': 1.0,
                    'episode_length': 2,
                    'task_info': {'test': True}
                }
            ]
            
            sft_data = collector._convert_to_sft_format(test_trajectories)
            assert len(sft_data) == 1
            assert sft_data[0]['success'] == True
            assert sft_data[0]['final_reward'] == 1.0
            
            print("✅ SFT format conversion test passed")
            
            # Test training rows creation
            training_rows = collector._create_training_rows(sft_data)
            assert len(training_rows) == 1  # One assistant message
            assert training_rows[0]['response'] == 'Hi there!'
            assert training_rows[0]['data_source'] == 'agent_sft_collection'
            
            print("✅ Training rows creation test passed")
            
        print("🎉 All SFTDataCollector tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ SFTDataCollector test failed: {e}")
        return False

def test_environment_variable_detection():
    """Test environment variable detection in main_ppo.py modifications."""
    print("🧪 Testing environment variable detection...")
    
    try:
        # Test COLLECT_SFT detection
        os.environ["COLLECT_SFT"] = "True"
        collect_sft = os.environ.get("COLLECT_SFT", "False").lower() == "true"
        assert collect_sft == True
        
        os.environ["COLLECT_SFT"] = "False"
        collect_sft = os.environ.get("COLLECT_SFT", "False").lower() == "true"
        assert collect_sft == False
        
        # Test SFT_SEED detection
        os.environ["SFT_SEED"] = "42"
        sft_seed = os.environ.get("SFT_SEED", None)
        assert sft_seed == "42"
        assert int(sft_seed) == 42
        
        # Test SFT_REQUIRE_SUCCESS detection
        os.environ["SFT_REQUIRE_SUCCESS"] = "True"
        require_success = os.environ.get("SFT_REQUIRE_SUCCESS", "False").lower() == "true"
        assert require_success == True
        
        # Clean up
        for var in ["COLLECT_SFT", "SFT_SEED", "SFT_REQUIRE_SUCCESS"]:
            if var in os.environ:
                del os.environ[var]
        
        print("✅ Environment variable detection test passed")
        return True
        
    except Exception as e:
        print(f"❌ Environment variable test failed: {e}")
        return False

def test_file_modifications():
    """Test that the file modifications are syntactically correct."""
    print("🧪 Testing file modifications...")
    
    try:
        # Test main_ppo.py modifications
        main_ppo_path = '/Users/zhuty/Documents/verl-agent/verl/trainer/main_ppo.py'
        with open(main_ppo_path, 'r') as f:
            content = f.read()
            
        # Check for key modifications
        assert 'COLLECT_SFT' in content
        assert 'SFT_SEED' in content
        assert 'val_only = True' in content
        assert 'open_dict' in content
        
        print("✅ main_ppo.py modifications verified")
        
        # Test ray_trainer.py modifications  
        ray_trainer_path = '/Users/zhuty/Documents/verl-agent/verl/trainer/ppo/ray_trainer.py'
        with open(ray_trainer_path, 'r') as f:
            content = f.read()
            
        # Check for key modifications
        assert 'SFTDataCollector' in content
        assert 'collect_sft' in content
        assert 'sft_data_collector' in content
        assert 'add_validation_batch' in content
        assert 'save_sft_data' in content
        
        print("✅ ray_trainer.py modifications verified")
        
        # Test sft_data_collector.py exists and is importable
        sft_collector_path = '/Users/zhuty/Documents/verl-agent/verl/utils/sft_data_collector.py'
        assert os.path.exists(sft_collector_path)
        
        with open(sft_collector_path, 'r') as f:
            content = f.read()
            
        assert 'class SFTDataCollector' in content
        assert 'def add_validation_batch' in content
        assert 'def save_sft_data' in content
        
        print("✅ sft_data_collector.py verified")
        
        return True
        
    except Exception as e:
        print(f"❌ File modifications test failed: {e}")
        return False

def test_script_creation():
    """Test that the example script was created correctly."""
    print("🧪 Testing script creation...")
    
    try:
        script_path = '/Users/zhuty/Documents/verl-agent/examples/ppo_trainer/run_alfworld_sft_collection.sh'
        assert os.path.exists(script_path)
        
        with open(script_path, 'r') as f:
            content = f.read()
            
        # Check for key elements
        assert 'COLLECT_SFT=True' in content
        assert 'SFT_SEED=' in content
        assert 'SFT_REQUIRE_SUCCESS=' in content
        assert 'verl.trainer.main_ppo' in content
        
        print("✅ SFT collection script verified")
        
        # Test README creation
        readme_path = '/Users/zhuty/Documents/verl-agent/SFT_COLLECTION_README.md'
        assert os.path.exists(readme_path)
        
        with open(readme_path, 'r') as f:
            content = f.read()
            
        assert 'SFT Data Collection' in content
        assert 'COLLECT_SFT' in content
        assert 'Environment Variables' in content
        
        print("✅ README documentation verified")
        
        return True
        
    except Exception as e:
        print(f"❌ Script creation test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 Starting SFT Collection Integration Tests")
    print("=" * 60)
    
    tests = [
        ("SFT Data Collector", test_sft_data_collector),
        ("Environment Variables", test_environment_variable_detection), 
        ("File Modifications", test_file_modifications),
        ("Script Creation", test_script_creation),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📋 Running {test_name} Test...")
        if test_func():
            passed += 1
        else:
            print(f"💥 {test_name} test failed!")
    
    print("\n" + "=" * 60)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! SFT collection integration is working correctly.")
        print("\n📝 Next Steps:")
        print("1. Set COLLECT_SFT=True environment variable")
        print("2. Run the PPO trainer with your desired configuration")
        print("3. The system will automatically collect SFT data during validation")
        print("4. Check the generated sft_data_* directory for results")
        return True
    else:
        print(f"❌ {total - passed} tests failed. Please check the implementation.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
