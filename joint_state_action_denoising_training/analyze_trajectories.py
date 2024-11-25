import os
import argparse
import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt
from utils import seedEverything
from models import MLP
from config import Config
from ccil_utils import load_env
from diffusion_model import DiffusionPolicy
from policy_agents import JointStateActionAgent, BaselineBCAgent, DiffusionPolicyAgent
from dataset_utils import load_data

def construct_parser():
    parser = argparse.ArgumentParser(description='Analyzing trajectories from different methods')
    parser.add_argument("--config_path", type=str, required=True, help="Path to config file")
    parser.add_argument("--checkpoint_dir", type=str, required=True, help="Path to model checkpoint directory")
    parser.add_argument("--traj_idx", type=int, default=0, help="Index of trajectory to analyze")
    parser.add_argument("--methods", nargs='+', default=['baseline', 'joint_bc', 'joint_denoising'],
                      choices=['baseline', 'joint_bc', 'joint_denoising', 'diffusion'],
                      help="List of methods to analyze")
    return parser

def rollout_trajectory(env, agent, initial_state, horizon=1000):
    """Roll out a trajectory using the given agent"""
    state = env.reset()
    # state = initial_state
    states = [state]
    actions = []
    rewards = []
    
    # # Reset environment first
    # env.reset()
    
    # # For MuJoCo environments (like Ant)
    # if hasattr(env.unwrapped, 'sim'):
    #     # Print state dimensions for debugging
    #     print(f"Full state shape: {initial_state.shape}")
    #     print(f"qpos dim: {env.unwrapped.sim.model.nq}")
    #     print(f"qvel dim: {env.unwrapped.sim.model.nv}")
        
    #     # For Ant environment:
    #     # qpos: root(3) + rotation(4) + joint angles(8) = 15
    #     # qvel: root_vel(3) + rotation_vel(3) + joint velocities(8) = 14
    #     # rest: contact forces, etc.
    #     nq = env.unwrapped.sim.model.nq  # typically 15 for Ant
    #     nv = env.unwrapped.sim.model.nv  # typically 14 for Ant
        
    #     # Split the initial state into position and velocity components
    #     qpos = initial_state[:nq]  # First nq elements are positions
    #     qvel = initial_state[nq:nq+nv]  # Next nv elements are velocities
        
    #     print(f"Setting qpos: {qpos.shape}, qvel: {qvel.shape}")
        
    #     # Set the state
    #     env.unwrapped.sim.data.qpos[:] = qpos
    #     env.unwrapped.sim.data.qvel[:] = qvel
    #     env.unwrapped.sim.forward()
        
    #     # Get and print observation for debugging
    #     obs = env.unwrapped._get_obs()
    #     print(f"Initial state: {initial_state[:10]}...")  # Print first 10 elements
    #     print(f"Observation after setting state: {obs[:10]}...")  # Print first 10 elements
    #     print(f"Difference in first {nq+nv} elements: {np.abs(initial_state[:nq+nv] - obs[:nq+nv]).max()}")
        
    #     # Use the original initial state for the first entry
    #     state = initial_state
    # else:
    #     print("Warning: Environment does not support direct state setting. Using reset state.")
    #     state = env.unwrapped._get_obs()
    
    for t in range(horizon):
        action = agent.predict([state])
        next_state, reward, done, _ = env.step(action)
        
        states.append(next_state)
        actions.append(action)
        rewards.append(reward)
        
        if done:
            break
            
        state = next_state
    
    return np.array(states), np.array(actions), np.array(rewards)

def plot_trajectories(ground_truth_states, method_trajectories, save_path=None):
    """Plot state trajectories for all methods and ground truth"""
    num_states = ground_truth_states.shape[1]
    num_methods = len(method_trajectories)
    
    # Create subplots for each state dimension
    fig, axes = plt.subplots(num_states, 1, figsize=(12, 4*num_states))
    if num_states == 1:
        axes = [axes]
    
    # Plot each state dimension
    for i, ax in enumerate(axes):
        # Plot ground truth with dashed line
        ax.plot(ground_truth_states[:, i], 'k--', label='Ground Truth', alpha=0.7)
        
        # Plot each method with solid lines
        for method_name, states in method_trajectories.items():
            ax.plot(states[:, i], '-', label=method_name, alpha=0.7)
        
        ax.set_title(f'State Dimension {i}')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('State Value')
        ax.legend()
        ax.grid(True)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()

def main():
    parser = construct_parser()
    args = parser.parse_args()

    # Load config and set up environment
    Config.load_config_for_testing(args.config_path)
    seedEverything(Config.SEED if hasattr(Config, 'SEED') else 42)
    
    # Load environment
    env, meta_env = load_env(Config)
    env.seed(Config.SEED if hasattr(Config, 'SEED') else 42)
    
    # Load original data to get ground truth trajectory
    controls_list, x_traj_list = load_data(Config)
    ground_truth_states = x_traj_list[args.traj_idx]
    initial_state = ground_truth_states[0]
    
    # Initialize models and agents
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    method_trajectories = {}
    
    # Initialize and evaluate each method
    if 'baseline' in args.methods:
        baseline_model = MLP(input_dim=state_dim, output_dim=action_dim).to(device)
        baseline_model.load_state_dict(torch.load(os.path.join(args.checkpoint_dir, "baseline_bc_model.pt")))
        baseline_model.eval()
        baseline_agent = BaselineBCAgent(baseline_model, device, action_dim, stats_path=args.checkpoint_dir)
        states, _, rewards = rollout_trajectory(env, baseline_agent, initial_state)
        method_trajectories['baseline'] = states
        print(f"baseline method traj length {len(states)}")
        print(f"baseline method reward: {sum(rewards)}")
    
    if 'joint_bc' in args.methods or 'joint_denoising' in args.methods:
        bc_model = MLP(input_dim=state_dim, output_dim=action_dim + state_dim).to(device)
        bc_model.load_state_dict(torch.load(os.path.join(args.checkpoint_dir, "joint_bc_model.pt")))
        bc_model.eval()
        
        if 'joint_bc' in args.methods:
            joint_bc_agent = JointStateActionAgent(bc_model, None, device, action_dim, stats_path=args.checkpoint_dir)
            states, _, _ = rollout_trajectory(env, joint_bc_agent, initial_state)
            method_trajectories['joint_bc'] = states
        
        if 'joint_denoising' in args.methods:
            denoising_model = MLP(input_dim=action_dim + state_dim * 2, output_dim=action_dim + state_dim).to(device)
            denoising_model.load_state_dict(torch.load(os.path.join(args.checkpoint_dir, "joint_denoising_model.pt")))
            denoising_model.eval()
            joint_denoising_agent = JointStateActionAgent(bc_model, denoising_model, device, action_dim, stats_path=args.checkpoint_dir)
            states, _, _ = rollout_trajectory(env, joint_denoising_agent, initial_state)
            method_trajectories['joint_denoising'] = states
    
    if 'diffusion' in args.methods:
        diffusion = DiffusionPolicy(state_dim=state_dim, action_dim=action_dim, device=device)
        diffusion.score_model.load_state_dict(torch.load(os.path.join(args.checkpoint_dir, "diffusion_model.pt")))
        diffusion.score_model.eval()
        diffusion_agent = DiffusionPolicyAgent(diffusion, device, action_dim, stats_path=args.checkpoint_dir)
        states, _, _ = rollout_trajectory(env, diffusion_agent, initial_state)
        method_trajectories['diffusion'] = states
    
    # Plot trajectories
    save_path = os.path.join("./", f"trajectory_comparison_traj{args.traj_idx}.png")
    plot_trajectories(ground_truth_states, method_trajectories, save_path)
    
    # Save trajectory data
    trajectory_data = {
        'ground_truth': ground_truth_states,
        **method_trajectories
    }
    # with open(os.path.join(args.checkpoint_dir, f"trajectory_data_traj{args.traj_idx}.pkl"), 'wb') as f:
    #     pickle.dump(trajectory_data, f)

if __name__ == "__main__":
    main() 