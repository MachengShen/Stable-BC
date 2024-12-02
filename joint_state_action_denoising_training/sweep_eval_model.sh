#!/bin/bash

ROOT_DIR="/cephfs/cjyai/joint_denoising_bc_save_best/"
NUM_EVAL_EPISODES=30
SPECIFIC_TASK=$1

# Function to process a single task directory
process_task() {
    local task_dir=$1
    local task_name=$(basename "$task_dir")
    echo "Processing task: $task_name"
    
    # Loop through all timestamp directories in the task directory
    for timestamp_dir in "$task_dir"/*/ ; do
        if [ ! -d "$timestamp_dir" ]; then continue; fi
        timestamp=$(basename "$timestamp_dir")
        echo "  Timestamp: $timestamp"
        
        # Check if config.yaml exists
        config_path="$timestamp_dir/config.yaml"
        if [ ! -f "$config_path" ]; then
            echo "    Skipping: No config.yaml found"
            continue
        fi
        
        # Find the joint_training directory and its subdirectory
        joint_training_dir="$timestamp_dir/results/joint_training"
        if [ ! -d "$joint_training_dir" ]; then
            echo "    Skipping: No joint_training directory found"
            continue
        fi
        
        # Find the first dems directory (should only be one)
        dems_dir=$(find "$joint_training_dir" -maxdepth 1 -type d -name "*dems" | head -n 1)
        if [ -z "$dems_dir" ]; then
            echo "    Skipping: No dems directory found"
            continue
        fi
        
        echo "    Using demonstrations directory: $(basename "$dems_dir")"
        
        # Process each seed directory
        for seed_dir in "$dems_dir"/seed*/ ; do
            if [ ! -d "$seed_dir" ]; then continue; fi
            seed=$(basename "$seed_dir")
            
            # Check if evaluation results already exist
            eval_results_file="$seed_dir/eval_results/evaluation_results.txt"
            if [ -f "$eval_results_file" ]; then
                echo "    Skipping $seed: Evaluation results already exist"
                continue
            fi
            
            echo "    Processing $seed"
            
            # Run evaluation
            python eval_model.py \
                --config_path "$config_path" \
                --checkpoint_dir "$seed_dir" \
                --num_eval_episodes $NUM_EVAL_EPISODES \
                --methods baseline joint_bc joint_denoising joint_state_only_bc
                
            echo "    Completed $seed"
        done
    done
}

# Modified main execution section at the bottom
if [ -n "$SPECIFIC_TASK" ]; then
    # Process only the specified task
    task_dir="$ROOT_DIR$SPECIFIC_TASK"
    if [ -d "$task_dir" ]; then
        process_task "$task_dir"
    else
        echo "Error: Task directory '$SPECIFIC_TASK' not found in $ROOT_DIR"
        exit 1
    fi
else
    # Process all task directories as before
    for task_dir in "$ROOT_DIR"/*/ ; do
        if [ ! -d "$task_dir" ]; then continue; fi
        process_task "$task_dir"
    done
fi

echo "Evaluation sweep completed!"