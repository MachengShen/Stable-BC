import os
import argparse
import pickle
import numpy as np
import torch
from utils import seedEverything
from models import MLP
from config import Config
import d3rlpy
import d4rl
import time
from ccil_utils import load_env, evaluate_on_environment
from diffusion_model import DiffusionPolicy
from policy_agents import JointStateActionAgent, BaselineBCAgent, RandomAgent, DiffusionPolicyAgent

def evaluate_models(config, checkpoint_dir, num_eval_episodes=10, methods=None, noise_levels=None):
    """
    Evaluate models with given configuration
    Returns: Dictionary containing evaluation results
    """
    if methods is None:
        methods = ['baseline', 'joint_bc', 'joint_denoising']
    if noise_levels is None:
        noise_levels = [0, 0.0003, 0.001, 0.01, 0.1]
    
    # Set random seed
    seedEverything(config.SEED if hasattr(config, 'SEED') else 42)
    d3rlpy.seed(config.SEED if hasattr(config, 'SEED') else 42)

    # Load environment
    env, meta_env = load_env(config)
    env.seed(config.SEED if hasattr(config, 'SEED') else 42)
    env.action_space.seed(config.SEED if hasattr(config, 'SEED') else 42)
    env.observation_space.seed(config.SEED if hasattr(config, 'SEED') else 42)

    # Get dimensions from environment
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # Initialize results and agents
    results = {}
    agents = {}

    # Initialize requested models and agents
    if 'baseline' in methods:
        baseline_model = MLP(input_dim=state_dim, output_dim=action_dim).to(device)
        baseline_model.load_state_dict(torch.load(os.path.join(checkpoint_dir, "baseline_bc_model.pt")))
        baseline_model.eval()
        agents['baseline'] = BaselineBCAgent(baseline_model, device, action_dim, stats_path=checkpoint_dir)
        results['baseline'] = {'timing': []}

    if 'joint_bc' in methods or 'joint_denoising' in methods:
        bc_model = MLP(input_dim=state_dim, output_dim=action_dim + state_dim).to(device)
        bc_model.load_state_dict(torch.load(os.path.join(checkpoint_dir, "joint_bc_model.pt")))
        bc_model.eval()

        if 'joint_bc' in methods:
            agents['joint_bc'] = JointStateActionAgent(bc_model, None, device, action_dim, stats_path=checkpoint_dir)
            results['joint_bc'] = {'timing': []}

        if 'joint_denoising' in methods:
            denoising_model = MLP(input_dim=action_dim + state_dim * 2, output_dim=action_dim + state_dim).to(device)
            denoising_model.load_state_dict(torch.load(os.path.join(checkpoint_dir, "joint_denoising_model.pt")))
            denoising_model.eval()
            agents['joint_denoising'] = JointStateActionAgent(bc_model, denoising_model, device, action_dim, stats_path=checkpoint_dir)
            results['joint_denoising'] = {'timing': []}

    if 'random' in methods:
        agents['random'] = RandomAgent(env.action_space)
        results['random'] = {'timing': []}
        # Evaluate random agent only once
        start_time = time.time()
        rewards, success = evaluate_on_environment(
            env, agents['random'], n_trials=num_eval_episodes,
            metaworld=meta_env, sensor_noise_size=0.0, actuator_noise_size=0.0
        )
        results['random']['timing'].append(time.time() - start_time)
        results['random'][0.0] = {
            'rewards': rewards,
            'success_rate': success / num_eval_episodes,
            'mean_reward': np.mean(rewards),
            'std_reward': np.std(rewards) / np.sqrt(num_eval_episodes)
        }
        methods.remove('random')  # Remove random from noise sweep

    # Evaluate other methods with different noise levels
    for noise in noise_levels:
        for method in methods:
            start_time = time.time()
            rewards, success = evaluate_on_environment(
                env, agents[method], n_trials=num_eval_episodes,
                metaworld=meta_env, sensor_noise_size=noise, actuator_noise_size=noise
            )
            results[method]['timing'].append(time.time() - start_time)
            
            results[method][noise] = {
                'rewards': rewards,
                'success_rate': success / num_eval_episodes,
                'mean_reward': np.mean(rewards),
                'std_reward': np.std(rewards) / np.sqrt(num_eval_episodes)
            }

    return results

def print_evaluation_results(results, methods, num_eval_episodes):
    """Print evaluation results in a formatted table"""
    # Print header
    print("\nEvaluation Results:")
    header_width = 12 + 35 * len(methods)
    print("-" * header_width)
    
    header = f"{'Noise Level':^12} |"
    subheader = f"{'':^12} |"
    for method in methods:
        header += f" {method:^33} |"
        subheader += f" {'Mean±Std Reward':^18} {'Success':^13} |"
    print(header)
    print(subheader)
    print("-" * header_width)

    # Print results for each noise level
    noise_levels = sorted([k for k in results[methods[0]].keys() if isinstance(k, (int, float))])
    for noise in noise_levels:
        line = f"{noise:^12.5f} |"
        for method in methods:
            if noise in results[method]:
                line += f" {results[method][noise]['mean_reward']:^8.2f}±{results[method][noise]['std_reward']:^7.2f}"
                line += f" {results[method][noise]['success_rate']:^13.2f} |"
            else:
                line += f" {'N/A':^18} {'N/A':^13} |"
        print(line)

    # Print timing information
    print("\nTiming Statistics:")
    print("-" * 60)
    print(f"{'Method':^20} | {'Mean Time (s)':^15} | {'Time per Episode (ms)':^20}")
    print("-" * 60)
    for method in methods:
        if 'timing' in results[method]:
            mean_time = np.mean(results[method]['timing'])
            time_per_episode = (mean_time / num_eval_episodes) * 1000  # Convert to ms
            print(f"{method:^20} | {mean_time:^15.3f} | {time_per_episode:^20.3f}")

def save_evaluation_results(results, checkpoint_dir, methods, num_eval_episodes):
    """Save evaluation results to files"""
    results_dir = os.path.join(checkpoint_dir, "eval_results")
    os.makedirs(results_dir, exist_ok=True)
    
    # Save full results as pickle
    with open(os.path.join(results_dir, "evaluation_results.pkl"), "wb") as f:
        pickle.dump(results, f)
    
    # Save human-readable summary
    with open(os.path.join(results_dir, "evaluation_results.txt"), "w") as f:
        f.write(f"Evaluation Results (Methods: {', '.join(methods)})\n")
        f.write("=================\n\n")
        
        noise_levels = sorted([k for k in results[methods[0]].keys() if isinstance(k, (int, float))])
        for noise in noise_levels:
            f.write(f"Noise Level: {noise}\n")
            f.write("-" * 50 + "\n")
            
            for method in methods:
                if noise in results[method]:
                    f.write(f"\n{method}:\n")
                    f.write(f"  Mean Reward: {results[method][noise]['mean_reward']:.2f} ± {results[method][noise]['std_reward']:.2f}\n")
                    f.write(f"  Success Rate: {results[method][noise]['success_rate']:.2f}\n")
            f.write("\n" + "=" * 50 + "\n")
            
        f.write("\nTiming Statistics:\n")
        f.write("-" * 50 + "\n")
        for method in methods:
            if 'timing' in results[method]:
                mean_time = np.mean(results[method]['timing'])
                time_per_episode = (mean_time / num_eval_episodes) * 1000
                f.write(f"{method}:\n")
                f.write(f"  Mean Time (s): {mean_time:.3f}\n")
                f.write(f"  Time per Episode (ms): {time_per_episode:.3f}\n")

def main():
    parser = construct_parser()
    args = parser.parse_args()
    
    Config.load_config_for_testing(args.config_path)
    # Run evaluation
    results = evaluate_models(
        Config,
        args.checkpoint_dir,
        args.num_eval_episodes,
        args.methods
    )
    
    # Print results
    print_evaluation_results(results, args.methods, args.num_eval_episodes)
    
    # Save results
    save_evaluation_results(results, args.checkpoint_dir, args.methods, args.num_eval_episodes)
    
    return results  # Return results for use in hyperparameter tuning

def construct_parser():
    parser = argparse.ArgumentParser(description='Evaluating Joint State-Action Models')
    parser.add_argument("--config_path", type=str, required=True, help="Path to config file")
    parser.add_argument("--checkpoint_dir", type=str, required=True, help="Path to model checkpoint directory")
    parser.add_argument("--num_eval_episodes", type=int, default=100, help="Number of evaluation episodes")
    parser.add_argument("--methods", nargs='+', default=['baseline', 'joint_bc', 'joint_denoising', 'random'],
                      choices=['baseline', 'joint_bc', 'joint_denoising', 'diffusion', 'random'],
                      help="List of methods to evaluate")
    return parser

if __name__ == "__main__":
    main() 