import numpy as np
import pickle
import matplotlib.pyplot as plt
import multiprocessing
import random
import os
import torch
import math
from ccil_utils import evaluate_on_environment


# taken from https://www.kaggle.com/code/rhythmcam/random-seed-everything
DEFAULT_RANDOM_SEED = 42

def seedBasic(seed=DEFAULT_RANDOM_SEED):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    
def seedTorch(seed=DEFAULT_RANDOM_SEED):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
      
# basic + tensorflow + torch 
def seedEverything(seed=DEFAULT_RANDOM_SEED):
    seedBasic(seed)
    seedTorch(seed)

def sample_initial_conditions_array(y_range, z_range, num_of_sample):
    initial_x = 0.2
    initial_vx = 0
    initial_vy = 0
    initial_vz = 0
    initial_conditions_arrray = np.random.uniform( (initial_x, y_range[0], z_range[0], initial_vx, initial_vy, initial_vz), (initial_x, y_range[1], z_range[1], initial_vx, initial_vy, initial_vz), (num_of_sample, 6) )
    return initial_conditions_arrray

def to_tensor(array, device="cuda" if torch.cuda.is_available() else "cpu"):
    return torch.tensor(array, dtype=torch.float32).to(device)

def get_statistics(controls_list, x_traj_list, is_delta=False):
    # Concatenate the controls and trajectories
    controls_array = np.concatenate(controls_list)
    x_traj_array = np.concatenate([np.stack(traj) for traj in x_traj_list])

    # Calculate mean and std for controls and trajectories
    controls_mean = np.mean(controls_array, axis=0)
    controls_std = np.std(controls_array, axis=0)
    x_traj_mean = np.mean(x_traj_array, axis=0)
    x_traj_std = np.std(x_traj_array, axis=0)

    print("Controls mean:", controls_mean)
    print("Controls std:", controls_std)
    print("{'Delta ' if is_delta else ''}Trajectory mean:", x_traj_mean)
    print("{'Delta ' if is_delta else ''}Trajectory std:", x_traj_std)
    print("Controls range:", np.max(controls_array), np.min(controls_array))
    print("{'Delta ' if is_delta else ''}Trajectory range:", np.max(x_traj_array), np.min(x_traj_array))
    return controls_mean, controls_std, x_traj_mean, x_traj_std


def save_normalization_stats(controls_mean, controls_std, x_traj_mean, x_traj_std, num_dems, random_seed, Config):
        # Save normalization statistics to the log directory
    stats_path = Config.get_model_path(num_dems, random_seed)
    os.makedirs(stats_path, exist_ok=True)
    
    stats_dict = {
        'controls_mean': controls_mean,
        'controls_std': controls_std, 
        'x_traj_mean': x_traj_mean,
        'x_traj_std': x_traj_std
    }
    
    with open(os.path.join(stats_path, 'normalization_stats.pkl'), 'wb') as f:
        pickle.dump(stats_dict, f)
        
        
def save_models(model, denoising_model, num_dems, random_seed, Config, model_surfix=""):
    models_path = Config.get_model_path(num_dems, random_seed)
    if not os.path.exists(models_path):
        os.makedirs(models_path)

    torch.save(model.state_dict(), f"{models_path}/joint_bc_model{model_surfix}.pt")
    torch.save(denoising_model.state_dict(), f"{models_path}/joint_denoising_model{model_surfix}.pt")

def evaluate_model(agent, env, meta_env, device, num_episodes=20, noise_levels=None):
    """Helper function to evaluate a model during training"""
    if noise_levels is None:
        noise_levels = [0.0]
        
    results = {}
    
    for noise in noise_levels:
        rewards, success = evaluate_on_environment(
            env,
            agent,
            n_trials=num_episodes,
            metaworld=meta_env,
            sensor_noise_size=noise,
            actuator_noise_size=noise
        )
        
        results[noise] = {
            'mean_reward': np.mean(rewards),
            'std_reward': np.std(rewards) / np.sqrt(num_episodes),
            'success_rate': success / num_episodes
        }
    
    return results