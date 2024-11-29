#!/bin/bash

ROOT_DIR="/cephfs/cjyai/joint_denoising_bc_debugged"
NUM_EVAL_EPISODES=30

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
        
        # Process each seed directory
        results_dir="$timestamp_dir/results/joint_training/10dems"
        for seed_dir in "$results_dir"/seed*/ ; do
            if [ ! -d "$seed_dir" ]; then continue; fi
            seed=$(basename "$seed_dir")
            
            # Check if evaluation results already exist
            # eval_results_file="$seed_dir/eval_results/evaluation_results.txt"
            # if [ -f "$eval_results_file" ]; then
            #     echo "    Skipping $seed: Evaluation results already exist"
            #     continue
            # fi
            
            echo "    Processing $seed"
            
            # Run evaluation with specified methods
            python eval_model.py \
                --config_path "$config_path" \
                --checkpoint_dir "$seed_dir" \
                --num_eval_episodes $NUM_EVAL_EPISODES \
                --methods baseline joint_bc joint_denoising joint_state_only_bc  # Specify methods to evaluate
                
            echo "    Completed $seed"
        done
    done
}

# Process each task directory
for task_dir in "$ROOT_DIR"/*/ ; do
    if [ ! -d "$task_dir" ]; then continue; fi
    process_task "$task_dir"
done

echo "Evaluation sweep completed!"