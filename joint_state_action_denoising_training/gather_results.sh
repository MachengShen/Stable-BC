#!/bin/bash

ROOT_DIR="/cephfs/cjyai/joint_denoising_bc_debugged"
OUTPUT_FILE="gathered_results.txt"

echo "Gathering evaluation results..." > $OUTPUT_FILE
echo "================================" >> $OUTPUT_FILE

# Function to process a single task directory
process_task() {
    local task_dir=$1
    local task_name=$(basename "$task_dir")
    echo -e "\nTask: $task_name" >> $OUTPUT_FILE
    echo "================================" >> $OUTPUT_FILE
    
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
                echo -e "\nTimestamp: $timestamp, $seed" >> $OUTPUT_FILE
                echo "--------------------------------" >> $OUTPUT_FILE
                cat "$eval_results_file" >> $OUTPUT_FILE
                echo "--------------------------------" >> $OUTPUT_FILE
            fi
        done
    done
}

# Process each task directory
for task_dir in "$ROOT_DIR"/*/ ; do
    if [ ! -d "$task_dir" ]; then continue; fi
    process_task "$task_dir"
done

echo -e "\nResults gathering completed! Check $OUTPUT_FILE" 