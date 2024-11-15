import numpy as np
import pickle
import matplotlib.pyplot as plt
import multiprocessing
import random
import os
import torch
import math




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

def get_statistics(controls_list, x_traj_list):
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
    print("Trajectory mean:", x_traj_mean)
    print("Trajectory std:", x_traj_std)
    return controls_std, x_traj_std

def save_models(model, denoising_model, num_dems, random_seed, Config):
    models_path = Config.get_model_path(num_dems, random_seed)
    if not os.path.exists(models_path):
        os.makedirs(models_path)

    torch.save(model.state_dict(), f"{models_path}/joint_bc_model.pt")
    torch.save(denoising_model.state_dict(), f"{models_path}/joint_denoising_model.pt")