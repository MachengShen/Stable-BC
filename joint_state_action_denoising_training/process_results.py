#!/usr/bin/env python3

import os
import pickle
import argparse
from glob import glob
from collections import defaultdict
import numpy as np
import yaml

def parse_args():
    parser = argparse.ArgumentParser(description='Process training results from pickle files')
    parser.add_argument('--root_dir', type=str, required=True,
                       help='Root directory containing the experiment results')
    parser.add_argument('--output_file', type=str, required=True,
                       help='Output file to save the processed results')
    return parser.parse_args()

def process_pickle_file(pickle_file):
    try:
        print(f"Processing pickle file: {pickle_file}")
        with open(pickle_file, 'rb') as f:
            data = pickle.load(f)
            if isinstance(data, dict) and 'all_eval_results' in data:
                return data
            return None
    except Exception as e:
        print(f"Error processing {pickle_file}: {str(e)}")
        return None

def calculate_method_averages(all_results):
    # Structure to store results for each method and noise level
    method_results = defaultdict(lambda: defaultdict(lambda: {'rewards': [], 'success_rates': []}))
    
    for result in all_results:
        if not isinstance(result, dict) or 'all_eval_results' not in result:
            continue
            
        eval_results = result['all_eval_results']
        for method, epochs_data in eval_results.items():
            if not isinstance(epochs_data, list):
                continue
                
            # Get the last epoch's results (most recent)
            if epochs_data:
                last_epoch_data = epochs_data[-1]
                if isinstance(last_epoch_data, dict):
                    for noise_level, metrics in last_epoch_data.items():
                        if isinstance(metrics, dict):
                            if 'mean_reward' in metrics:
                                method_results[method][noise_level]['rewards'].append(metrics['mean_reward'])
                            if 'success_rate' in metrics:
                                method_results[method][noise_level]['success_rates'].append(metrics['success_rate'])
    
    # Calculate averages
    averages = {}
    for method in method_results:
        averages[method] = {}
        for noise_level in method_results[method]:
            rewards = method_results[method][noise_level]['rewards']
            success_rates = method_results[method][noise_level]['success_rates']
            
            averages[method][noise_level] = {
                'avg_reward': np.mean(rewards) if rewards else None,
                'std_reward': np.std(rewards) if rewards else None,
                'avg_success_rate': np.mean(success_rates) if success_rates else None,
                'std_success_rate': np.std(success_rates) if success_rates else None,
                'num_seeds': len(rewards)
            }
    
    return averages

def write_results(results, output_file):
    print(f"\nWriting results to {output_file}")
    
    with open(output_file, 'w') as f:
        f.write("Training Results Summary\n")
        f.write("=======================\n\n")
        
        # Calculate method averages
        method_averages = calculate_method_averages(results)
        
        # Write averaged results
        f.write("Averaged Results Across Seeds\n")
        f.write("============================\n\n")
        
        for method in sorted(method_averages.keys()):
            f.write(f"\nMethod: {method}\n")
            f.write("=" * (len(method) + 8) + "\n")
            
            for noise_level in sorted(method_averages[method].keys(), key=float):
                metrics = method_averages[method][noise_level]
                f.write(f"\nNoise Level: {noise_level}\n")
                f.write("-" * 40 + "\n")
                f.write(f"Number of seeds: {metrics['num_seeds']}\n")
                
                if metrics['avg_reward'] is not None:
                    f.write(f"Average reward: {metrics['avg_reward']:.4f} ± {metrics['std_reward']:.4f}\n")
                if metrics['avg_success_rate'] is not None:
                    f.write(f"Average success rate: {metrics['avg_success_rate']:.4f} ± {metrics['std_success_rate']:.4f}\n")
                f.write("-" * 40 + "\n")
        
        # Write detailed results
        f.write("\n\nDetailed Results by Seed\n")
        f.write("=======================\n\n")
        
        for result in results:
            if not isinstance(result, dict) or 'all_eval_results' not in result:
                continue
                
            eval_results = result['all_eval_results']
            for method, epochs_data in eval_results.items():
                if not isinstance(epochs_data, list):
                    continue
                    
                f.write(f"\nMethod: {method}\n")
                f.write("=" * (len(method) + 8) + "\n")
                
                # Get the last epoch's results
                if epochs_data:
                    last_epoch_data = epochs_data[-1]
                    if isinstance(last_epoch_data, dict):
                        for noise_level in sorted(last_epoch_data.keys(), key=float):
                            metrics = last_epoch_data[noise_level]
                            f.write(f"\nNoise Level: {noise_level}\n")
                            f.write("-" * 40 + "\n")
                            for key, value in metrics.items():
                                f.write(f"{key}: {value}\n")
                            f.write("-" * 40 + "\n")

def find_training_results(task_dir):
    print(f"\nProcessing task directory: {task_dir}")
    results = []
    
    # Look for timestamp directories
    timestamp_dirs = glob(os.path.join(task_dir, "*/"))
    print(f"Found {len(timestamp_dirs)} timestamp directories")
    
    for timestamp_dir in timestamp_dirs:
        if not os.path.isdir(timestamp_dir):
            continue
            
        print(f"\nChecking timestamp directory: {timestamp_dir}")
        
        # Find the joint_training directory
        joint_training_dir = os.path.join(timestamp_dir, "results", "joint_training")
        if not os.path.exists(joint_training_dir):
            print(f"No joint_training directory found at: {joint_training_dir}")
            continue
            
        # Find the dems directory
        dems_dirs = glob(os.path.join(joint_training_dir, "*dems"))
        if not dems_dirs:
            print(f"No dems directories found in: {joint_training_dir}")
            continue
            
        dems_dir = dems_dirs[0]
        
        # Process each seed directory
        seed_dirs = glob(os.path.join(dems_dir, "seed*/"))
        print(f"Found {len(seed_dirs)} seed directories")
        
        for seed_dir in seed_dirs:
            if not os.path.isdir(seed_dir):
                continue
                
            print(f"\nProcessing seed directory: {seed_dir}")
            
            # Look for result files
            result_files = [
                "baseline_bc_results.pkl",
                "baseline_bc_results_noisy.pkl",
                "training_results_state_only.pkl"
            ]
            
            for result_file in result_files:
                pickle_path = os.path.join(seed_dir, result_file)
                if not os.path.exists(pickle_path):
                    continue
                    
                data = process_pickle_file(pickle_path)
                if data:
                    results.append(data)
    
    return results

def main():
    args = parse_args()
    
    print(f"Starting to process results from directory: {args.root_dir}")
    all_results = []
    
    # Process each task directory
    task_dirs = glob(os.path.join(args.root_dir, "*/"))
    print(f"Found {len(task_dirs)} task directories")
    
    for task_dir in task_dirs:
        if not os.path.isdir(task_dir):
            continue
        
        results = find_training_results(task_dir)
        all_results.extend(results)
    
    print(f"\nTotal results found across all tasks: {len(all_results)}")
    print(f"Writing results to {args.output_file}...")
    write_results(all_results, args.output_file)
    print("Processing completed!")

if __name__ == "__main__":
    main() 