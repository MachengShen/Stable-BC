#!/bin/bash

ROOT_DIR="/cephfs/cjyai/joint_denoising_bc_10dems_metaworld_optimized/"
EVAL_RESULTS_FILE="gathered_eval_results.txt"
TRAINING_RESULTS_FILE="gathered_training_results.txt"

echo "Gathering evaluation results..." > $EVAL_RESULTS_FILE
echo "================================" >> $EVAL_RESULTS_FILE

# Function to process a single task directory for evaluation results
process_eval_results() {
    local task_dir=$1
    local task_name=$(basename "$task_dir")
    echo -e "\nTask: $task_name" >> $EVAL_RESULTS_FILE
    echo "================================" >> $EVAL_RESULTS_FILE
    
    # Loop through all timestamp directories in the task directory
    for timestamp_dir in "$task_dir"/*/ ; do
        if [ ! -d "$timestamp_dir" ]; then continue; fi
        timestamp=$(basename "$timestamp_dir")
        
        # Find the joint_training directory and its subdirectory
        joint_training_dir="$timestamp_dir/results/joint_training"
        if [ ! -d "$joint_training_dir" ]; then
            continue
        fi
        
        # Find the first dems directory (should only be one)
        dems_dir=$(find "$joint_training_dir" -maxdepth 1 -type d -name "*dems" | head -n 1)
        if [ -z "$dems_dir" ]; then
            continue
        fi
        
        # Process each seed directory
        for seed_dir in "$dems_dir"/seed*/ ; do
            if [ ! -d "$seed_dir" ]; then continue; fi
            seed=$(basename "$seed_dir")
            
            # Check if evaluation results exist
            eval_results_file="$seed_dir/eval_results/evaluation_results.txt"
            if [ -f "$eval_results_file" ]; then
                echo -e "\nTimestamp: $timestamp, $seed" >> $EVAL_RESULTS_FILE
                echo "--------------------------------" >> $EVAL_RESULTS_FILE
                cat "$eval_results_file" >> $EVAL_RESULTS_FILE
                echo "--------------------------------" >> $EVAL_RESULTS_FILE
            fi
        done
    done
}

# Process evaluation results
for task_dir in "$ROOT_DIR"/*/ ; do
    if [ ! -d "$task_dir" ]; then continue; fi
    process_eval_results "$task_dir"
done

echo "Processing training results from pickle files..."
python process_results.py --root_dir "$ROOT_DIR" --output_file "$TRAINING_RESULTS_FILE"

echo -e "\nResults gathering completed!"
echo "Evaluation results saved to: $EVAL_RESULTS_FILE"
echo "Training results saved to: $TRAINING_RESULTS_FILE"