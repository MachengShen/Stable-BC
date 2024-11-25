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




def construct_parser():
    parser = argparse.ArgumentParser(description='Evaluating Joint State-Action Models')
    parser.add_argument("--config_path", type=str, required=True, help="Path to config file")
    parser.add_argument("--checkpoint_dir", type=str, required=True, help="Path to model checkpoint directory")
    parser.add_argument("--num_eval_episodes", type=int, default=100, help="Number of evaluation episodes")
    parser.add_argument("--methods", nargs='+', default=['baseline', 'joint_bc', 'joint_denoising', 'random'],
                      choices=['baseline', 'joint_bc', 'joint_denoising', 'diffusion', 'random'],
                      help="List of methods to evaluate")
    return parser

def main():
    parser = construct_parser()
    args = parser.parse_args()

    # Load config
    Config.load_config_for_testing(args.config_path)
    
    # Set random seed
    seedEverything(Config.SEED if hasattr(Config, 'SEED') else 42)
    d3rlpy.seed(Config.SEED if hasattr(Config, 'SEED') else 42)

    # Load environment
    env, meta_env = load_env(Config)
    env.seed(Config.SEED if hasattr(Config, 'SEED') else 42)
    env.action_space.seed(Config.SEED if hasattr(Config, 'SEED') else 42)
    env.observation_space.seed(Config.SEED if hasattr(Config, 'SEED') else 42)

    # Load models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Get dimensions from environment
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # Initialize results dictionary with timing information
    results = {}
    agents = {}

    # Add random agent if requested
    if 'random' in args.methods:
        agents['random'] = RandomAgent(env.action_space)
        results['random'] = {'timing': []}
        
        # Evaluate random agent only once (no noise sweep)
        print("\nEvaluating Random Agent:")
        print("-" * 50)
        
        start_time = time.time()
        rewards, success = evaluate_on_environment(
            env,
            agents['random'],
            n_trials=args.num_eval_episodes,
            metaworld=meta_env,
            sensor_noise_size=0.0,  # No noise for random agent
            actuator_noise_size=0.0
        )
        results['random']['timing'].append(time.time() - start_time)
        
        results['random'][0.0] = {
            'rewards': rewards,
            'success_rate': success / args.num_eval_episodes,
            'mean_reward': np.mean(rewards),
            'std_reward': np.std(rewards) / np.sqrt(args.num_eval_episodes)
        }
        
        print(f"Random Agent Performance:")
        print(f"Mean Reward: {results['random'][0.0]['mean_reward']:^8.2f}±{results['random'][0.0]['std_reward']:^7.2f}")
        print(f"Success Rate: {results['random'][0.0]['success_rate']:^13.2f}")
        print("-" * 50 + "\n")

    # Remove random from methods for noise sweep evaluation
    sweep_methods = [m for m in args.methods if m != 'random']
    
    if sweep_methods:  # Only print and evaluate other methods if there are any
        # Print header
        print("\nEvaluation Results:")
        header_width = 12 + 35 * len(sweep_methods)
        print("-" * header_width)
        
        header = f"{'Noise Level':^12} |"
        subheader = f"{'':^12} |"
        for method in sweep_methods:
            header += f" {method:^33} |"
            subheader += f" {'Mean±Std Reward':^18} {'Success':^13} |"
        print(header)
        print(subheader)
        print("-" * header_width)

        # Initialize only the requested models and agents
        if 'baseline' in sweep_methods:
            baseline_model = MLP(input_dim=state_dim, output_dim=action_dim).to(device)
            baseline_model.load_state_dict(torch.load(os.path.join(args.checkpoint_dir, "baseline_bc_model.pt")))
            baseline_model.eval()
            agents['baseline'] = BaselineBCAgent(
                baseline_model, device, action_dim, stats_path=args.checkpoint_dir
            )
            results['baseline'] = {'timing': []}

        if 'joint_bc' in sweep_methods or 'joint_denoising' in sweep_methods:
            bc_model = MLP(input_dim=state_dim, output_dim=action_dim + state_dim).to(device)
            bc_model.load_state_dict(torch.load(os.path.join(args.checkpoint_dir, "joint_bc_model.pt")))
            bc_model.eval()

            if 'joint_bc' in sweep_methods:
                agents['joint_bc'] = JointStateActionAgent(
                    bc_model, None, device, action_dim, stats_path=args.checkpoint_dir
                )
                results['joint_bc'] = {'timing': []}

            if 'joint_denoising' in sweep_methods:
                denoising_model = MLP(input_dim=action_dim + state_dim * 2, output_dim=action_dim + state_dim).to(device)
                denoising_model.load_state_dict(torch.load(os.path.join(args.checkpoint_dir, "joint_denoising_model.pt")))
                denoising_model.eval()
                agents['joint_denoising'] = JointStateActionAgent(
                    bc_model, denoising_model, device, action_dim, stats_path=args.checkpoint_dir
                )
                results['joint_denoising'] = {'timing': []}

        if 'diffusion' in sweep_methods:
            diffusion = DiffusionPolicy(
                state_dim=state_dim,
                action_dim=action_dim,
                device=device,
            )
            diffusion.score_model.load_state_dict(torch.load(os.path.join(args.checkpoint_dir, "diffusion_model.pt")))
            diffusion.score_model.eval()
            agents['diffusion'] = DiffusionPolicyAgent(
                diffusion, device, action_dim, stats_path=args.checkpoint_dir
            )
            results['diffusion'] = {'timing': []}

        # Evaluate with different noise levels
        sweep_noises = [0, 0.0003, 0.001, 0.01, 0.1]

        for noise in sweep_noises:
            line = f"{noise:^12.5f} |"
            
            for method in sweep_methods:
                start_time = time.time()
                rewards, success = evaluate_on_environment(
                    env,
                    agents[method],
                    n_trials=args.num_eval_episodes,
                    metaworld=meta_env,
                    sensor_noise_size=noise,
                    actuator_noise_size=noise
                )
                results[method]['timing'].append(time.time() - start_time)
                
                # Calculate success rate (no std)
                success_rate = success / args.num_eval_episodes
                
                results[method][noise] = {
                    'rewards': rewards,
                    'success_rate': success_rate,
                    'mean_reward': np.mean(rewards),
                    'std_reward': np.std(rewards) / np.sqrt(args.num_eval_episodes)
                }
                
                line += f" {results[method][noise]['mean_reward']:^8.2f}±{results[method][noise]['std_reward']:^7.2f}"
                line += f" {results[method][noise]['success_rate']:^13.2f} |"
            
            print(line)

    # Print timing statistics
    print("\nTiming Statistics (averaged over all noise levels):")
    print("-" * 80)
    print(f"{'Method':^20} | {'Mean Time (s)':^15} | {'Time per Episode (ms)':^20}")
    print("-" * 80)
    
    for method in args.methods:
        mean_time = np.mean(results[method]['timing'])
        time_per_episode = (mean_time / args.num_eval_episodes) * 1000
        print(f"{method:^20} | {mean_time:^15.3f} | {time_per_episode:^20.3f}")

    # Save results
    results_dir = os.path.join(args.checkpoint_dir, "eval_results")
    os.makedirs(results_dir, exist_ok=True)
    
    with open(os.path.join(results_dir, "evaluation_results.txt"), "w") as f:
        f.write(f"Evaluation Results (Methods: {', '.join(args.methods)})\n")
        f.write("=================\n\n")
        
        for noise in sweep_noises:
            f.write(f"Noise Level: {noise}\n")
            f.write("-" * 50 + "\n")
            
            for method in args.methods:
                f.write(f"\n{method}:\n")
                f.write(f"  Mean Reward: {results[method][noise]['mean_reward']:.2f}\n")
                f.write(f"  Std Reward: {results[method][noise]['std_reward']:.2f}\n")
                f.write(f"  Success Rate: {results[method][noise]['success_rate']:.2f}\n")
            f.write("\n" + "=" * 50 + "\n")
            
        f.write("\nTiming Statistics (averaged over all noise levels):\n")
        f.write("-" * 50 + "\n")
        for method in args.methods:
            mean_time = np.mean(results[method]['timing'])
            time_per_episode = (mean_time / args.num_eval_episodes) * 1000
            f.write(f"{method}:\n")
            f.write(f"  Mean Time (s): {mean_time:.3f}\n")
            f.write(f"  Time per Episode (ms): {time_per_episode:.3f}\n")

if __name__ == "__main__":
    main() 