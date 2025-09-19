#!/bin/bash

# =============================================================================
# REAL AlfWorld SFT Trajectory Generation Script
# =============================================================================
# This script runs REAL AlfWorld environment interactions to generate SFT data
# Uses your existing agent system infrastructure for authentic trajectories
# =============================================================================

set -e  # Exit on any error

# =============================================================================
# CONFIGURATION VARIABLES - MODIFY THESE AS NEEDED
# =============================================================================

# Model configuration
MODEL_PATH="Qwen/Qwen2.5-1.5B-Instruct"  # Hugging Face model path

# Episode parameters
NUM_EPISODES=50             # Number of episodes to run
MAX_STEPS=30               # Maximum steps per episode
ENV_NUM=4                  # Number of parallel environments

# Success filtering
REQUIRE_SUCCESS=false       # Set to 'true' to only keep successful episodes

# Output configuration
OUTPUT_DIR="alfworld_real_sft_data"    # Output directory prefix
LOG_DIR="logs"                         # Directory for log files
CONFIG_PATH="config_alfworld.yaml"     # Configuration file path

# Environment variables
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export CUDA_VISIBLE_DEVICES="0,1,2,3"  # Adjust based on your GPU setup
export VLLM_WORKER_MULTIPROC_METHOD="spawn"

# =============================================================================
# DERIVED VARIABLES
# =============================================================================

SCRIPT_NAME="generate_alfworld_real_trajectories.py"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
FINAL_OUTPUT_DIR="${OUTPUT_DIR}_${TIMESTAMP}"

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

print_header() {
    echo "=============================================================================="
    echo "$1"
    echo "=============================================================================="
}

print_config() {
    echo "Configuration:"
    echo "  Model Path: $MODEL_PATH"
    echo "  Number of Episodes: $NUM_EPISODES"
    echo "  Max Steps per Episode: $MAX_STEPS"
    echo "  Environment Processes: $ENV_NUM"
    echo "  Require Success: $REQUIRE_SUCCESS"
    echo "  Config Path: $CONFIG_PATH"
    echo "  Output Directory: $FINAL_OUTPUT_DIR"
    echo "  Log Directory: $LOG_DIR"
    echo "  CUDA Devices: $CUDA_VISIBLE_DEVICES"
    echo ""
}

check_dependencies() {
    echo "Checking dependencies..."
    
    # Check if Python is available
    if ! command -v python &> /dev/null; then
        echo "Error: Python is not installed or not in PATH"
        exit 1
    fi
    
    # Check if the script exists
    if [ ! -f "$SCRIPT_NAME" ]; then
        echo "Error: $SCRIPT_NAME not found in current directory"
        echo "Please make sure you're running this script from the correct directory"
        exit 1
    fi
    
    # Check Python packages
    echo "Checking Python packages..."
    python -c "
import sys
missing = []
try:
    import transformers
    print('✓ transformers available')
except ImportError:
    missing.append('transformers')

try:
    import pandas
    print('✓ pandas available')
except ImportError:
    missing.append('pandas')

try:
    import sklearn
    print('✓ scikit-learn available')
except ImportError:
    missing.append('scikit-learn')

try:
    import numpy
    print('✓ numpy available')
except ImportError:
    missing.append('numpy')

try:
    import ray
    print('✓ ray available')
except ImportError:
    missing.append('ray')

try:
    import omegaconf
    print('✓ omegaconf available')
except ImportError:
    missing.append('omegaconf')

if missing:
    print(f'Error: Missing packages: {missing}')
    print('Install with: pip install ' + ' '.join(missing))
    sys.exit(1)
else:
    print('✓ All required packages available')
" || {
        echo "Error: Missing required Python packages"
        exit 1
    }
    
    # Check if agent system is available
    echo "Checking agent system..."
    python -c "
try:
    from agent_system.environments.env_manager import AlfWorldEnvironmentManager
    from agent_system.environments.env_package.alfworld import alfworld_projection
    print('✓ Agent system components available')
except ImportError as e:
    print(f'Error: Agent system not available: {e}')
    print('Make sure you have the agent system properly installed')
    import sys
    sys.exit(1)
" || {
        echo "Error: Agent system components not available"
        echo "Please make sure your agent system is properly set up"
        exit 1
    }
    
    echo "✓ All dependencies satisfied"
    echo ""
}

run_generation() {
    print_header "Starting REAL AlfWorld SFT Trajectory Generation"
    
    # Build command
    CMD="python $SCRIPT_NAME"
    CMD="$CMD --model_path '$MODEL_PATH'"
    CMD="$CMD --num_episodes $NUM_EPISODES"
    CMD="$CMD --max_steps $MAX_STEPS"
    CMD="$CMD --env_num $ENV_NUM"
    CMD="$CMD --output_dir '$OUTPUT_DIR'"
    CMD="$CMD --log_dir '$LOG_DIR'"
    CMD="$CMD --config_path '$CONFIG_PATH'"
    
    # Add success filtering if enabled
    if [ "$REQUIRE_SUCCESS" = "true" ]; then
        CMD="$CMD --require_success"
    fi
    
    echo "Running command:"
    echo "$CMD"
    echo ""
    
    # Run the command
    eval $CMD
    
    if [ $? -eq 0 ]; then
        echo ""
        print_header "Generation Completed Successfully!"
        return 0
    else
        echo ""
        print_header "Generation Failed!"
        return 1
    fi
}

show_results() {
    echo "Results:"
    
    # Find the output directory
    OUTPUT_DIRS=(${OUTPUT_DIR}_*)
    if [ ${#OUTPUT_DIRS[@]} -eq 0 ]; then
        echo "  No output directories found"
        return 1
    fi
    
    # Get the most recent output directory
    LATEST_DIR=""
    for dir in "${OUTPUT_DIRS[@]}"; do
        if [ -d "$dir" ]; then
            LATEST_DIR="$dir"
        fi
    done
    
    if [ -z "$LATEST_DIR" ]; then
        echo "  No valid output directory found"
        return 1
    fi
    
    echo "  Output Directory: $LATEST_DIR"
    echo "  Files generated:"
    
    # List generated files
    if [ -d "$LATEST_DIR" ]; then
        for file in "$LATEST_DIR"/*; do
            if [ -f "$file" ]; then
                filename=$(basename "$file")
                filesize=$(du -h "$file" | cut -f1)
                echo "    - $filename ($filesize)"
            fi
        done
    fi
    
    # Show training data statistics if available
    TRAIN_FILE="$LATEST_DIR/alfworld_real_train.parquet"
    VAL_FILE="$LATEST_DIR/alfworld_real_val.parquet"
    
    if [ -f "$TRAIN_FILE" ] && [ -f "$VAL_FILE" ]; then
        echo ""
        echo "  Training Data Statistics:"
        python -c "
import pandas as pd
try:
    train_df = pd.read_parquet('$TRAIN_FILE')
    val_df = pd.read_parquet('$VAL_FILE')
    print(f'    Training samples: {len(train_df)}')
    print(f'    Validation samples: {len(val_df)}')
    print(f'    Total samples: {len(train_df) + len(val_df)}')
except Exception as e:
    print(f'    Error reading parquet files: {e}')
"
    fi
    
    echo ""
    echo "🎮 REAL AlfWorld trajectory data is ready!"
    echo "This data comes from actual environment interactions!"
}

cleanup_on_exit() {
    echo ""
    echo "Cleaning up Ray processes..."
    python -c "
import ray
if ray.is_initialized():
    ray.shutdown()
    print('✓ Ray shutdown complete')
" 2>/dev/null || true
    echo "Script completed."
}

# =============================================================================
# MAIN EXECUTION
# =============================================================================

main() {
    # Set up cleanup on exit
    trap cleanup_on_exit EXIT
    
    print_header "REAL AlfWorld SFT Trajectory Generation"
    print_config
    
    # Pre-flight checks
    check_dependencies
    
    # Warn about real environment usage
    echo "⚠️  IMPORTANT: This script will run REAL AlfWorld environments!"
    echo "   - This requires proper AlfWorld installation"
    echo "   - Episodes will take longer than synthetic generation"
    echo "   - GPU/CPU resources will be used intensively"
    echo ""
    
    # Skipping confirmation, just run
    
    # Run generation
    if run_generation; then
        show_results
        echo ""
        echo "🎉 Success! REAL AlfWorld SFT data generated successfully!"
        echo "   This data contains authentic environment interactions."
        echo "   Use it for high-quality SFT training."
    else
        echo ""
        echo "❌ Generation failed. Check the logs for details."
        echo "   Common issues:"
        echo "   - AlfWorld environment not properly installed"
        echo "   - Insufficient GPU memory"
        echo "   - Ray initialization problems"
        exit 1
    fi
}

# =============================================================================
# COMMAND LINE ARGUMENT PARSING
# =============================================================================

while [[ $# -gt 0 ]]; do
    case $1 in
        --model_path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --num_episodes)
            NUM_EPISODES="$2"
            shift 2
            ;;
        --max_steps)
            MAX_STEPS="$2"
            shift 2
            ;;
        --env_num)
            ENV_NUM="$2"
            shift 2
            ;;
        --require_success)
            REQUIRE_SUCCESS="true"
            shift 1
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --log_dir)
            LOG_DIR="$2"
            shift 2
            ;;
        --config_path)
            CONFIG_PATH="$2"
            shift 2
            ;;
        --cuda_devices)
            export CUDA_VISIBLE_DEVICES="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [options]"
            echo ""
            echo "REAL AlfWorld SFT Trajectory Generation"
            echo "Generates training data from actual AlfWorld environment interactions"
            echo ""
            echo "Options:"
            echo "  --model_path PATH        Model path (default: $MODEL_PATH)"
            echo "  --num_episodes N         Number of episodes (default: $NUM_EPISODES)"
            echo "  --max_steps N           Max steps per episode (default: $MAX_STEPS)"
            echo "  --env_num N             Number of parallel envs (default: $ENV_NUM)"
            echo "  --require_success        Only keep successful episodes"
            echo "  --output_dir DIR        Output directory prefix (default: $OUTPUT_DIR)"
            echo "  --log_dir DIR           Log directory (default: $LOG_DIR)"
            echo "  --config_path PATH      Config file path (default: $CONFIG_PATH)"
            echo "  --cuda_devices DEVICES  CUDA devices (default: $CUDA_VISIBLE_DEVICES)"
            echo "  --help, -h              Show this help message"
            echo ""
            echo "Example:"
            echo "  $0 --num_episodes 100 --require_success"
            echo ""
            echo "⚠️  This script runs REAL AlfWorld environments!"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information."
            exit 1
            ;;
    esac
done

# Run main function
main
