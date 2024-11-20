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
from ccil_utils import load_env, evaluate_on_environment
from diffusion_model import DiffusionPolicy
import time

class JointStateActionAgent:
    def __init__(self, bc_model, denoising_model, device, action_dim, stats_path):
        self.bc_model = bc_model
        self.denoising_model = denoising_model
        self.device = device
        self.action_dim = action_dim
        
        # Load normalization statistics
        with open(os.path.join(stats_path, 'normalization_stats.pkl'), 'rb') as f:
            stats = pickle.load(f)
            self.state_mean = torch.FloatTensor(stats['x_traj_mean']).to(device)
            self.state_std = torch.FloatTensor(stats['x_traj_std']).to(device)
            self.action_mean = torch.FloatTensor(stats['controls_mean']).to(device)
            self.action_std = torch.FloatTensor(stats['controls_std']).to(device)

    def normalize_state(self, state):
        return (state - self.state_mean) / (self.state_std + 1e-8)

    def denormalize_action(self, action):
        return action * self.action_std + self.action_mean

    def predict(self, state):
        state_tensor = torch.FloatTensor(state).to(self.device)
        if len(state_tensor.shape) == 1:
            state_tensor = state_tensor.unsqueeze(0)

        # Normalize state
        state_tensor = self.normalize_state(state_tensor)

        with torch.no_grad():
            # Get initial prediction from BC model
            bc_output = self.bc_model(state_tensor)
            action = bc_output[:, :self.action_dim]
            next_state_pred = bc_output[:, self.action_dim:]
            
            # Use denoising model if available
            if self.denoising_model is not None:
                # Prepare input for denoising model: [noisy_action, current_state, next_state]
                denoising_input = torch.cat([action, state_tensor, next_state_pred], dim=1)
                denoised_output = self.denoising_model(denoising_input)
                action = denoised_output[:, :self.action_dim]

            # Denormalize action before returning
            action = self.denormalize_action(action)
            
        if action is None:
            raise ValueError("Action prediction is None")
        if torch.isnan(action).any() or torch.isinf(action).any():
            raise ValueError("Action prediction contains NaN or Inf values")
        
        return action.cpu().numpy()[0]

class BaselineBCAgent:
    def __init__(self, bc_model, device, action_dim, stats_path):
        self.bc_model = bc_model
        self.device = device
        self.action_dim = action_dim
        
        # Load normalization statistics
        with open(os.path.join(stats_path, 'normalization_stats.pkl'), 'rb') as f:
            stats = pickle.load(f)
            self.state_mean = torch.FloatTensor(stats['x_traj_mean']).to(device)
            self.state_std = torch.FloatTensor(stats['x_traj_std']).to(device)
            self.action_mean = torch.FloatTensor(stats['controls_mean']).to(device)
            self.action_std = torch.FloatTensor(stats['controls_std']).to(device)

    def normalize_state(self, state):
        return (state - self.state_mean) / (self.state_std + 1e-8)

    def denormalize_action(self, action):
        return action * self.action_std + self.action_mean

    def predict(self, state):
        state_tensor = torch.FloatTensor(state).to(self.device)
        if len(state_tensor.shape) == 1:
            state_tensor = state_tensor.unsqueeze(0)

        # Normalize state
        state_tensor = self.normalize_state(state_tensor)

        with torch.no_grad():
            # Get initial prediction from BC model
            bc_output = self.bc_model(state_tensor)
            action = bc_output[:, :self.action_dim]
            next_state_pred = bc_output[:, self.action_dim:]
            
            # Denormalize action before returning
            action = self.denormalize_action(action)
            
        if action is None:
            raise ValueError("Action prediction is None")
        if torch.isnan(action).any() or torch.isinf(action).any():
            raise ValueError("Action prediction contains NaN or Inf values")
        
        return action.cpu().numpy()[0]

class DiffusionPolicyAgent:
    def __init__(self, diffusion_model, device, action_dim, stats_path):
        self.diffusion = diffusion_model
        self.device = device
        self.action_dim = action_dim
        
        # Load normalization statistics
        with open(os.path.join(stats_path, 'normalization_stats.pkl'), 'rb') as f:
            stats = pickle.load(f)
            self.state_mean = torch.FloatTensor(stats['x_traj_mean']).to(device)
            self.state_std = torch.FloatTensor(stats['x_traj_std']).to(device)
            self.action_mean = torch.FloatTensor(stats['controls_mean']).to(device)
            self.action_std = torch.FloatTensor(stats['controls_std']).to(device)

    def normalize_state(self, state):
        return (state - self.state_mean) / (self.state_std + 1e-8)

    def denormalize_action(self, action):
        return action * self.action_std + self.action_mean

    def predict(self, state):
        state_tensor = torch.FloatTensor(state).to(self.device)
        if len(state_tensor.shape) == 1:
            state_tensor = state_tensor.unsqueeze(0)

        # Normalize state
        state_tensor = self.normalize_state(state_tensor)

        with torch.no_grad():
            # Sample from diffusion model
            x_T = torch.randn(state_tensor.shape[0], self.action_dim + state_tensor.shape[1]).to(self.device)
            
            # Reverse diffusion process
            x_t = x_T
            for t in reversed(range(self.diffusion.n_steps)):
                t_batch = torch.ones(state_tensor.shape[0], device=self.device).long() * t
                x_t = self.diffusion.p_sample(x_t, state_tensor, t_batch)
            
            # Extract action from joint prediction
            action = x_t[:, :self.action_dim]
            
            # Denormalize action
            action = self.denormalize_action(action)

        return action.cpu().numpy()[0]

def construct_parser():
    parser = argparse.ArgumentParser(description='Evaluating Joint State-Action Models')
    parser.add_argument("--config_path", type=str, required=True, help="Path to config file")
    parser.add_argument("--checkpoint_dir", type=str, required=True, help="Path to model checkpoint directory")
    parser.add_argument("--num_eval_episodes", type=int, default=100, help="Number of evaluation episodes")
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

    # Initialize models
    bc_model = MLP(input_dim=state_dim, output_dim=action_dim + state_dim)
    denoising_model = MLP(input_dim=action_dim + state_dim * 2, output_dim=action_dim + state_dim)

    # Load model weights
    bc_model.load_state_dict(torch.load(os.path.join(args.checkpoint_dir, "joint_bc_model.pt")))
    denoising_model.load_state_dict(torch.load(os.path.join(args.checkpoint_dir, "joint_denoising_model.pt")))
    
    bc_model.to(device)
    denoising_model.to(device)
    bc_model.eval()
    denoising_model.eval()

    # Initialize and load baseline BC model
    baseline_model = MLP(input_dim=state_dim, output_dim=action_dim).to(device)
    baseline_model.load_state_dict(torch.load(os.path.join(args.checkpoint_dir, "baseline_bc_model.pt")))
    baseline_model.eval()

    # Initialize and load diffusion model
    diffusion = DiffusionPolicy(
        state_dim=state_dim,
        action_dim=action_dim,
        device=device,
    )
    diffusion.score_model.load_state_dict(torch.load(os.path.join(args.checkpoint_dir, "diffusion_model.pt")))
    diffusion.score_model.eval()

    # Create all agents
    baseline_agent = BaselineBCAgent(
        baseline_model,
        device,
        action_dim,
        stats_path=args.checkpoint_dir
    )

    joint_bc_only = JointStateActionAgent(
        bc_model,
        None,  # No denoising model
        device,
        action_dim,
        stats_path=args.checkpoint_dir
    )

    joint_with_denoising = JointStateActionAgent(
        bc_model,
        denoising_model,
        device,
        action_dim,
        stats_path=args.checkpoint_dir
    )

    diffusion_agent = DiffusionPolicyAgent(
        diffusion,
        device,
        action_dim,
        stats_path=args.checkpoint_dir
    )

    # Initialize results dictionary with timing information
    results = {
        'baseline': {'timing': []},
        'joint_bc_only': {'timing': []},
        'joint_with_denoising': {'timing': []},
        'diffusion': {'timing': []}
    }

    # Evaluate with different noise levels
    sweep_noises = [0, 0.0003, 0.001, 0.01, 0.1, 0.5, 1.0]
    
    print("\nEvaluation Results:")
    print("-" * 150)
    print(f"{'Noise Level':^12} | {'Baseline':^32} | {'Joint BC Only':^32} | {'Joint BC + Denoising':^32} | {'Diffusion':^32}")
    print(f"{'':^12} | {'Mean Reward':^15} {'Success':^15} | {'Mean Reward':^15} {'Success':^15} | {'Mean Reward':^15} {'Success':^15} | {'Mean Reward':^15} {'Success':^15}")
    print("-" * 150)

    for noise in sweep_noises:
        # Time each prediction method
        start_time = time.time()
        rewards_baseline, success_baseline = evaluate_on_environment(
            env, 
            baseline_agent,
            n_trials=args.num_eval_episodes,
            metaworld=meta_env,
            sensor_noise_size=noise,
            actuator_noise_size=noise
        )
        results['baseline']['timing'].append(time.time() - start_time)
        
        start_time = time.time()
        rewards_joint, success_joint = evaluate_on_environment(
            env,
            joint_bc_only,
            n_trials=args.num_eval_episodes,
            metaworld=meta_env,
            sensor_noise_size=noise,
            actuator_noise_size=noise
        )
        results['joint_bc_only']['timing'].append(time.time() - start_time)
        
        start_time = time.time()
        rewards_denoising, success_denoising = evaluate_on_environment(
            env,
            joint_with_denoising,
            n_trials=args.num_eval_episodes,
            metaworld=meta_env,
            sensor_noise_size=noise,
            actuator_noise_size=noise
        )
        results['joint_with_denoising']['timing'].append(time.time() - start_time)

        start_time = time.time()
        rewards_diff, success_diff = evaluate_on_environment(
            env,
            diffusion_agent,
            n_trials=args.num_eval_episodes,
            metaworld=meta_env,
            sensor_noise_size=noise,
            actuator_noise_size=noise
        )
        results['diffusion']['timing'].append(time.time() - start_time)

        # Store results
        results['baseline'][noise] = {
            'rewards': rewards_baseline,
            'success_rate': success_baseline / args.num_eval_episodes,
            'mean_reward': np.mean(rewards_baseline),
            'std_reward': np.std(rewards_baseline)
        }
        
        results['joint_bc_only'][noise] = {
            'rewards': rewards_joint,
            'success_rate': success_joint / args.num_eval_episodes,
            'mean_reward': np.mean(rewards_joint),
            'std_reward': np.std(rewards_joint)
        }
        
        results['joint_with_denoising'][noise] = {
            'rewards': rewards_denoising,
            'success_rate': success_denoising / args.num_eval_episodes,
            'mean_reward': np.mean(rewards_denoising),
            'std_reward': np.std(rewards_denoising)
        }

        results['diffusion'][noise] = {
            'rewards': rewards_diff,
            'success_rate': success_diff / args.num_eval_episodes,
            'mean_reward': np.mean(rewards_diff),
            'std_reward': np.std(rewards_diff)
        }

        print(
            f"{noise:^12.5f} | "
            f"{results['baseline'][noise]['mean_reward']:^15.2f}±{results['baseline'][noise]['std_reward']:^6.2f} {results['baseline'][noise]['success_rate']:^15.2f} | "
            f"{results['joint_bc_only'][noise]['mean_reward']:^15.2f}±{results['joint_bc_only'][noise]['std_reward']:^6.2f} {results['joint_bc_only'][noise]['success_rate']:^15.2f} | "
            f"{results['joint_with_denoising'][noise]['mean_reward']:^15.2f}±{results['joint_with_denoising'][noise]['std_reward']:^6.2f} {results['joint_with_denoising'][noise]['success_rate']:^15.2f} | "
            f"{results['diffusion'][noise]['mean_reward']:^15.2f}±{results['diffusion'][noise]['std_reward']:^6.2f} {results['diffusion'][noise]['success_rate']:^15.2f}"
        )

    # Print timing statistics
    print("\nTiming Statistics (averaged over all noise levels):")
    print("-" * 80)
    print(f"{'Method':^20} | {'Mean Time (s)':^15} | {'Time per Episode (ms)':^20}")
    print("-" * 80)
    
    for method in results.keys():
        mean_time = np.mean(results[method]['timing'])
        time_per_episode = (mean_time / args.num_eval_episodes) * 1000  # Convert to ms
        print(f"{method:^20} | {mean_time:^15.3f} | {time_per_episode:^20.3f}")

    # Save results
    results_dir = os.path.join(args.checkpoint_dir, "eval_results")
    os.makedirs(results_dir, exist_ok=True)
    
    with open(os.path.join(results_dir, "evaluation_results.txt"), "w") as f:
        f.write("Evaluation Results\n")
        f.write("=================\n\n")
        
        for noise in results['baseline'].keys():
            if noise == 'timing':
                continue
                
            f.write(f"Noise Level: {noise}\n")
            f.write("-" * 50 + "\n")
            
            for method in ['baseline', 'joint_bc_only', 'joint_with_denoising', 'diffusion']:
                f.write(f"\n{method}:\n")
                f.write(f"  Mean Reward: {results[method][noise]['mean_reward']:.2f}\n")
                f.write(f"  Std Reward: {results[method][noise]['std_reward']:.2f}\n")
                f.write(f"  Success Rate: {results[method][noise]['success_rate']:.2f}\n")
            f.write("\n" + "=" * 50 + "\n")
            
        f.write("\nTiming Statistics (averaged over all noise levels):\n")
        f.write("-" * 50 + "\n")
        for method in results.keys():
            mean_time = np.mean(results[method]['timing'])
            time_per_episode = (mean_time / args.num_eval_episodes) * 1000
            f.write(f"{method}:\n")
            f.write(f"  Mean Time (s): {mean_time:.3f}\n")
            f.write(f"  Time per Episode (ms): {time_per_episode:.3f}\n")

if __name__ == "__main__":
    main() 