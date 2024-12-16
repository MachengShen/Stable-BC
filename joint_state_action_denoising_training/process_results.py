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

def process_pickle_file(pickle_file, method_prefix=None):
    try:
        print(f"Processing pickle file: {pickle_file}")
        with open(pickle_file, 'rb') as f:
            data = pickle.load(f)
            if isinstance(data, dict) and 'all_eval_results' in data:
                # If this is a noisy BC result, rename the method in all_eval_results
                if method_prefix == 'baseline_noisy' and 'baseline_bc' in data['all_eval_results']:
                    data['all_eval_results']['baseline_noisy'] = data['all_eval_results'].pop('baseline_bc')
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
            if not isinstance(epochs_data, list) or method == 'epochs' or method == 'noise_levels':
                continue
            
            # Find best performance across epochs for each noise level
            best_rewards = defaultdict(lambda: float('-inf'))
            best_success_rates = defaultdict(lambda: float('-inf'))
            best_metrics = defaultdict(dict)
            
            for epoch_data in epochs_data:
                if not isinstance(epoch_data, dict):
                    continue
                    
                for noise_level, metrics in epoch_data.items():
                    if not isinstance(metrics, dict):
                        continue
                        
                    # Update best metrics if current performance is better
                    if 'mean_reward' in metrics and metrics['mean_reward'] > best_rewards[noise_level]:
                        best_rewards[noise_level] = metrics['mean_reward']
                        best_metrics[noise_level] = metrics.copy()
                    
                    # Optionally track best success rate separately if it doesn't align with best reward
                    if 'success_rate' in metrics and metrics['success_rate'] > best_success_rates[noise_level]:
                        best_success_rates[noise_level] = metrics['success_rate']
                        # Note: We're using the metrics from best mean_reward, not best success_rate
            
            # Store best results for this seed
            for noise_level, metrics in best_metrics.items():
                if 'mean_reward' in metrics:
                    method_results[method][noise_level]['rewards'].append(metrics['mean_reward'])
                if 'success_rate' in metrics:
                    method_results[method][noise_level]['success_rates'].append(metrics['success_rate'])
    
    # Calculate averages across seeds
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
    
    # Group results by task
    task_results = defaultdict(list)
    for result in results:
        if 'task_name' in result:
            task_results[result['task_name']].append(result)
        else:
            task_results['unknown'].append(result)
    
    with open(output_file, 'w') as f:
        f.write("Training Results Summary (Best Performance per Seed)\n")
        f.write("===============================================\n\n")
        
        # Process each task separately
        for task_name, task_data in sorted(task_results.items()):
            f.write(f"Task: {task_name}\n")
            f.write("=" * (len(task_name) + 6) + "\n\n")
            
            # Calculate method averages using best epoch performance for each seed
            method_averages = calculate_method_averages(task_data)
            
            # Write averaged results for this task
            f.write("Averaged Results Across Seeds (Using Best Epoch per Seed)\n")
            f.write("====================================================\n\n")
            
            for method in sorted(method_averages.keys()):
                f.write(f"\nMethod: {method}\n")
                f.write("-" * (len(method) + 8) + "\n")
                
                for noise_level in sorted(method_averages[method].keys(), key=float):
                    metrics = method_averages[method][noise_level]
                    f.write(f"\nNoise Level: {noise_level}\n")
                    f.write("." * 40 + "\n")
                    f.write(f"Number of seeds: {metrics['num_seeds']}\n")
                    
                    if metrics['avg_reward'] is not None:
                        f.write(f"Average reward: {metrics['avg_reward']:.4f} ± {metrics['std_reward']:.4f}\n")
                    if metrics['avg_success_rate'] is not None:
                        f.write(f"Average success rate: {metrics['avg_success_rate']:.4f} ± {metrics['std_success_rate']:.4f}\n")
                    f.write("." * 40 + "\n")
            
            # Write detailed results for this task
            f.write("\n\nDetailed Best Results by Seed\n")
            f.write("===========================\n\n")
            
            for result in task_data:
                if not isinstance(result, dict) or 'all_eval_results' not in result:
                    continue
                    
                eval_results = result['all_eval_results']
                for method, epochs_data in eval_results.items():
                    if not isinstance(epochs_data, list) or method == 'epochs' or method == 'noise_levels':
                        continue
                        
                    f.write(f"\nMethod: {method}\n")
                    f.write("-" * (len(method) + 8) + "\n")
                    
                    # Find and write best results across epochs
                    best_metrics = defaultdict(lambda: {'mean_reward': float('-inf')})
                    for epoch_data in epochs_data:
                        if not isinstance(epoch_data, dict):
                            continue
                            
                        for noise_level, metrics in epoch_data.items():
                            if not isinstance(metrics, dict):
                                continue
                                
                            if 'mean_reward' in metrics and metrics['mean_reward'] > best_metrics[noise_level]['mean_reward']:
                                best_metrics[noise_level] = metrics.copy()
                    
                    # Write best results for each noise level
                    for noise_level in sorted(best_metrics.keys(), key=float):
                        metrics = best_metrics[noise_level]
                        f.write(f"\nNoise Level: {noise_level}\n")
                        f.write("." * 40 + "\n")
                        for key, value in metrics.items():
                            f.write(f"{key}: {value}\n")
                        f.write("." * 40 + "\n")
            
            # Add separator between tasks
            f.write("\n" + "=" * 80 + "\n\n")

def find_training_results(task_dir):
    print(f"\nProcessing task directory: {task_dir}")
    results = []
    
    # Get task name from directory path
    task_name = os.path.basename(os.path.normpath(task_dir))
    print(f"Task name: {task_name}")
    
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
            
            # Look for result files with updated naming
            result_files = {
                "baseline_bc_results.pkl": "baseline",
                "baseline_bc_results_noisy.pkl": "baseline_noisy",  # This will be mapped to baseline_noisy in the results
                "training_results_state_only.pkl": "state_only_bc"
            }
            
            for result_file, method_prefix in result_files.items():
                pickle_path = os.path.join(seed_dir, result_file)
                if not os.path.exists(pickle_path):
                    continue
                    
                data = process_pickle_file(pickle_path, method_prefix)
                if data:
                    # Add task name to the data
                    data['task_name'] = task_name
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