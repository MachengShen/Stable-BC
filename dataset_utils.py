import torch
import os
import pickle
import numpy as np
from torch.utils.data import Dataset
from utils import to_tensor


class StateImitationDataset(Dataset):
    def __init__(self, x_traj_array, controls_array, type=0):
        self.type = type
        self.x_traj_array = x_traj_array.astype(np.float32)
        self.controls_array = controls_array.astype(np.float32)

    def __len__(self):
        return self.controls_array.shape[0]

    def __getitem__(self, index):
        return self.x_traj_array[index], self.controls_array[index]


class BCDataset(Dataset):
    def __init__(self, x_traj_list, controls_list, action_only=False, state_only=False):
        self.x_traj_list = x_traj_list
        self.controls_list = controls_list
        self.action_only = action_only
        self.state_only = state_only
        assert not (action_only and state_only), "Cannot have both action_only and state_only set to True"

    def __len__(self):
        return sum(len(traj) - 1 for traj in self.x_traj_list)

    def __getitem__(self, idx):
        # Find the trajectory and step within the trajectory
        traj_idx = 0
        while idx >= len(self.x_traj_list[traj_idx]) - 1:
            idx -= len(self.x_traj_list[traj_idx]) - 1
            traj_idx += 1

        x_t = self.x_traj_list[traj_idx][idx]
        a_t = self.controls_list[traj_idx][idx]
        x_t_plus_1 = self.x_traj_list[traj_idx][idx + 1]

        input_tensor = to_tensor(x_t)
        if self.action_only:
            output_tensor = to_tensor(a_t)
        elif self.state_only:
            output_tensor = to_tensor(x_t_plus_1)
        else:
            output_tensor = to_tensor(np.concatenate([a_t, x_t_plus_1]))

        return input_tensor, output_tensor


class DenoisingDataset(Dataset):
    def __init__(
        self,
        x_traj_list,
        controls_list,
        action_noise_factor,
        state_noise_factor,
        action_noise_multiplier=0.02,
        state_noise_multiplier=0.02,
        input_state_only=False,
    ):
        self.x_traj_list = x_traj_list
        self.controls_list = controls_list
        self.action_noise_factor = action_noise_factor
        self.state_noise_factor = state_noise_factor
        self.action_noise_multiplier = action_noise_multiplier
        self.state_noise_multiplier = state_noise_multiplier
        self.input_state_only = input_state_only
        
    def __len__(self):
        return sum(len(traj) - 1 for traj in self.x_traj_list)

    def __getitem__(self, idx):
        # Find the trajectory and step within the trajectory
        traj_idx = 0
        while idx >= len(self.x_traj_list[traj_idx]) - 1:
            idx -= len(self.x_traj_list[traj_idx]) - 1
            traj_idx += 1

        clean_x_t = self.x_traj_list[traj_idx][idx]
        clean_a_t = self.controls_list[traj_idx][idx]
        clean_x_t_plus_1 = self.x_traj_list[traj_idx][idx + 1]

        noisy_a_t = (
            clean_a_t
            + self.action_noise_multiplier
            * self.action_noise_factor
            * np.random.randn(*clean_a_t.shape)
        )
        noisy_x_t_plus_1 = (
            clean_x_t_plus_1
            + self.state_noise_multiplier
            * self.state_noise_factor
            * np.random.randn(*clean_x_t_plus_1.shape)
        )

        if self.input_state_only:
            input_tensor = to_tensor(np.concatenate([clean_x_t, noisy_x_t_plus_1]))
        else:
            input_tensor = to_tensor(
                np.concatenate([clean_x_t, noisy_a_t, noisy_x_t_plus_1])
        )
        output_tensor = to_tensor(np.concatenate([clean_a_t, clean_x_t_plus_1]))

        return input_tensor, output_tensor
    
def read_ccil_data(directory='/root/neural_sde/CCIL-main/data/'):
    # Get all files in the directory
    files = [f for f in os.listdir(directory) if f.endswith('.pkl')]
    
    data = {}
    
    for file in files:
        file_path = os.path.join(directory, file)
        file_name = os.path.splitext(file)[0]
        
        # Read pickle files
        with open(file_path, 'rb') as f:
            df = pickle.load(f)
        data[file_name] = df
    
    return data    

def load_data(Config):
    if Config.TASK_TYPE == "sim-quadrotor":
        data_dict = pickle.load(open("../sim-quadrotor/sim-quadrotor/data/data_0.pkl", "rb"))
        controls_list = data_dict["controls_list"]
        x_traj_list = data_dict["x_trajectories_list"]
    
    elif Config.TASK_TYPE == "sim-intersection":
        data_dict = pickle.load(open("../sim-intersection/sim-intersection/data/data_0.pkl", "rb"))
        controls_list = data_dict["controls_list"]
        x_traj_list = data_dict["x_trajectories_list"]
    
    elif Config.TASK_TYPE == "CCIL":
        # Load CCIL data using the read_data function
        data = read_ccil_data(Config.CCIL_DATA_DIR)
        print(data.keys())
        task_data = data[Config.CCIL_TASK_NAME]
        
        # Convert CCIL format to our format
        controls_list = []
        x_traj_list = []
        for traj in task_data:
            # Convert observations and actions to numpy arrays if they're lists
            observations = np.vstack(traj['observations']) if isinstance(traj['observations'], list) else traj['observations']
            actions = np.vstack(traj['actions']) if isinstance(traj['actions'], list) else traj['actions']
            
            controls_list.append(actions)
            x_traj_list.append(observations)
    else:
        raise ValueError(f"Unknown task type: {Config.TASK_TYPE}")
    return controls_list, x_traj_list

def get_delta_x_traj_list(x_traj_list):
    """
    Compute state deltas from trajectory list and return statistics
    """
    delta_x_traj_list = []
    for traj in x_traj_list:
        # Convert to numpy array if it's a list
        traj = np.array(traj)
        # Compute deltas: x_{t+1} - x_t
        delta_traj = traj[1:] - traj[:-1]
        delta_x_traj_list.append(delta_traj)
    
    return delta_x_traj_list