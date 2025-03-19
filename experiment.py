import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing as mp
from tqdm import tqdm

from train_model import get_statistics, save_models
from config import Config
from utils import seedEverything, to_tensor

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from quadrotor_mppi import *


A_G = 9.81
roll_max = 0.4  # radians
pitch_max = 0.4
f_g_diff_max = 1.0  # max difference between thrust and gravity


class Model(nn.Module):
    def __init__(
        self,
        input_dim=6,
        output_dim=6,
        hidden_dim=256,
        is_denoising_net=False,
        joint_action_state=False,
        action_loss_weight=0.75,
        state_loss_weight=0.25,
    ):
        super(Model, self).__init__()

        # multi-layer perceptron
        self.pi_1 = nn.Linear(input_dim, hidden_dim)
        self.pi_2 = nn.Linear(hidden_dim, hidden_dim)
        self.pi_3 = nn.Linear(hidden_dim, hidden_dim)
        self.pi_4 = nn.Linear(hidden_dim, output_dim)

        self.mse_loss = nn.MSELoss(reduction="none")
        self.is_denoising_net = is_denoising_net
        self.joint_action_state = joint_action_state
        self.action_loss_weight = action_loss_weight
        self.state_loss_weight = state_loss_weight

        if self.is_denoising_net:
            # may need to check if the shift and normalizer are correct for action dim
            self.shift = torch.tensor([2.5, 2.5, 2.5, 0, 0, 0, 2.5, 2.5, 2.5, 0, 0, 0])
            self.normalizer = torch.tensor(
                [2.5, 2.5, 2.5, 4, 3, 1, 2.5, 2.5, 2.5, 4, 3, 1]
            )
        else:
            self.shift = torch.tensor([2.5, 2.5, 2.5, 0, 0, 0])
            self.normalizer = torch.tensor([2.5, 2.5, 2.5, 4, 3, 1])

    def loss_func(self, y_true, y_pred):
        # if self.joint_action_state:
        #     action_loss = self.mse_loss(y_true[:, :3], y_pred[:, :3]).mean()
        #     state_loss = self.mse_loss(y_true[:, 3:], y_pred[:, 3:]).mean()
        #     return (
        #         self.action_loss_weight * action_loss
        #         + self.state_loss_weight * state_loss
        #     )
        # else:
        return self.mse_loss(y_true, y_pred).mean()

    # policy
    def forward(self, state):
        state = state - self.shift.to(state.device)
        state = state / self.normalizer.to(state.device)
        x1 = torch.relu(self.pi_1(state))
        x2 = torch.relu(self.pi_2(x1))
        x3 = torch.relu(self.pi_3(x2))
        x = x1 + x2 + x3  # Skip connection
        x = torch.tanh(x)
        x = self.pi_4(x)

        x = x * torch.tensor([2.5, 2.5, 2.5, 4, 3, 1]).to(state.device) + torch.tensor(
            [2.5, 2.5, 2.5, 0, 0, 0]
        ).to(state.device)
        return x

    def get_action(self, state_tensor, device):
        if type(state_tensor) == np.ndarray:
            state_tensor = torch.tensor(state_tensor, dtype=torch.float32)
        self.eval()
        with torch.no_grad():
            # state_tensor = torch.cat( [ (state_tensor[:2] - torch.tensor([5.0, 5.0]) )/ 5, torch.sin(state_tensor[2]).reshape(1), torch.cos(state_tensor[2]).reshape(1) ] )
            # state_tensor = state_tensor.reshape((1, 4))
            # state_tensor = state_tensor.to(device)

            # for the state
            # divide by the some value

            # for the control
            # multiply by the control bound
            # add A_G form the output 0
            state_tensor = state_tensor.to(device)
            action = self.forward(state_tensor)
        return action.squeeze().detach()
    

class BCModel(nn.Module):
    def __init__(
        self,
        input_dim=6,
        output_dim=3,
        hidden_dim=256,
        action_loss_weight=0.75,
        state_loss_weight=0.25,
    ):
        super(BCModel, self).__init__()

        # multi-layer perceptron
        self.pi_1 = nn.Linear(input_dim, hidden_dim)
        self.pi_2 = nn.Linear(hidden_dim, hidden_dim)
        self.pi_3 = nn.Linear(hidden_dim, hidden_dim)
        self.pi_4 = nn.Linear(hidden_dim, output_dim)

        self.mse_loss = nn.MSELoss(reduction="none")
        self.action_loss_weight = action_loss_weight
        self.state_loss_weight = state_loss_weight

        self.shift = torch.tensor([2.5, 2.5, 2.5, 0, 0, 0])
        self.normalizer = torch.tensor([2.5, 2.5, 2.5, 4, 3, 1])

    def loss_func(self, y_true, y_pred):
        return self.mse_loss(y_true, y_pred).mean()

    # policy
    def forward(self, state):
        state = state - self.shift.to(state.device)
        state = state / self.normalizer.to(state.device)
        x1 = torch.relu(self.pi_1(state))
        x2 = torch.relu(self.pi_2(x1))
        x3 = torch.relu(self.pi_3(x2))
        x = x1 + x2 + x3  # Skip connection
        x = torch.tanh(x)
        x = self.pi_4(x)

        x = x * torch.tensor([f_g_diff_max, roll_max, pitch_max]).to(
                state.device
            ) + torch.tensor([A_G, 0, 0]).to(state.device)
        return x

    def get_action(self, state_tensor, device):
        if type(state_tensor) == np.ndarray:
            state_tensor = torch.tensor(state_tensor, dtype=torch.float32)
        self.eval()
        with torch.no_grad():
            state_tensor = state_tensor.to(device)
            action = self.forward(state_tensor)
        return action.squeeze().detach()


class BCDataset(Dataset):
    def __init__(self, x_traj_list, controls_list, input_noise_factor=0.0, output_type="next_state"):
        self.x_traj_list = x_traj_list
        self.controls_list = controls_list
        self.input_noise_factor = input_noise_factor
        self.output_type = output_type
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

        if self.output_type == "next_state":
            output_tensor = to_tensor(x_t_plus_1)
        elif self.output_type == "action":
            output_tensor = to_tensor(a_t)
        else:
            raise ValueError(f"Invalid output type: {self.output_type}")

        input_tensor = to_tensor(x_t) 
        input_tensor += self.input_noise_factor * torch.randn_like(input_tensor)
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
    ):
        self.x_traj_list = x_traj_list
        self.controls_list = controls_list
        self.action_noise_factor = action_noise_factor
        self.state_noise_factor = state_noise_factor
        self.action_noise_multiplier = action_noise_multiplier
        self.state_noise_multiplier = state_noise_multiplier

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

        # input_tensor = to_tensor(
        #     np.concatenate([clean_x_t, noisy_a_t, noisy_x_t_plus_1])
        # )
        # output_tensor = to_tensor(np.concatenate([clean_a_t, clean_x_t_plus_1]))

        input_tensor = to_tensor(
            # np.concatenate([clean_x_t, noisy_x_t_plus_1])
            noisy_x_t_plus_1
        )
        output_tensor = to_tensor(clean_x_t_plus_1)
        return input_tensor, output_tensor


def get_data(num_dems, traj_downsample_factor=20):
    data_dict = pickle.load(open("sim-quadrotor/data/data_0.pkl", "rb"))
    controls_list = data_dict["controls_list"]
    x_traj_list = data_dict["x_trajectories_list"]

    controls_std, x_traj_std = get_statistics(controls_list, x_traj_list)
    # randomly select num_dems indices

    indices = np.random.choice(len(controls_list), num_dems, replace=False)
    controls_list = [controls_list[i] for i in indices]
    x_traj_list = [x_traj_list[i] for i in indices]
    # Downsample trajectories by taking every traj_downsample_factor-th point
    controls_list = [controls[::traj_downsample_factor] for controls in controls_list]
    x_traj_list = [x_traj[::traj_downsample_factor] for x_traj in x_traj_list]
    # Split data into train and validation sets
    split_factor = 0.5
    train_size = int(len(controls_list) * split_factor)

    train_controls_list = controls_list[:train_size]
    train_x_traj_list = x_traj_list[:train_size]
    val_controls_list = controls_list[train_size:]
    val_x_traj_list = x_traj_list[train_size:]
    return (
        train_controls_list,
        train_x_traj_list,
        val_controls_list,
        val_x_traj_list,
        controls_std,
        x_traj_std,
    )


def train_model_joint(num_dems, random_seed, Config, save_ckpt=True):
    (
        train_controls_list,
        train_x_traj_list,
        val_controls_list,
        val_x_traj_list,
        controls_std,
        x_traj_std,
    ) = get_data(num_dems)
    # Set up the models and optimizers
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = BCModel(
    #     input_dim=6,
    #     output_dim=3,
    #     # is_denoising_net=False,
    #     # joint_action_state=False,
    #     action_loss_weight=Config.ACTION_LOSS_WEIGHT_BC,
    #     state_loss_weight=Config.STATE_LOSS_WEIGHT_BC,
    # )
    model = Model(
        input_dim=6,
        output_dim=6,
        is_denoising_net=False,
        joint_action_state=False,
        action_loss_weight=Config.ACTION_LOSS_WEIGHT_BC,
        state_loss_weight=Config.STATE_LOSS_WEIGHT_BC,
    )
    denoising_model = Model(
        input_dim=6,
        output_dim=6,
        is_denoising_net=False,
        joint_action_state=False,
        action_loss_weight=Config.ACTION_LOSS_WEIGHT_DENOISING,
        state_loss_weight=Config.STATE_LOSS_WEIGHT_DENOISING,
    )
    model.to(device)
    denoising_model.to(device)

    optimizer_model = torch.optim.Adam(
        model.parameters(), lr=Config.LR, weight_decay=1e-5
    )
    optimizer_denoising = torch.optim.Adam(
        denoising_model.parameters(), lr=Config.LR, weight_decay=1e-5
    )

    # Load and prepare datasets
    train_bc_dataset = BCDataset(
        x_traj_list=train_x_traj_list, controls_list=train_controls_list, input_noise_factor=0.0, output_type="next_state"
    )
    train_denoising_dataset = DenoisingDataset(
        x_traj_list=train_x_traj_list,
        controls_list=train_controls_list,
        action_noise_factor=controls_std,
        state_noise_factor=x_traj_std,
        action_noise_multiplier=Config.ACTION_NOISE_MULTIPLIER,
        state_noise_multiplier=Config.STATE_NOISE_MULTIPLIER,
    )
    val_bc_dataset = BCDataset(
        x_traj_list=val_x_traj_list, controls_list=val_controls_list, input_noise_factor=0.0, output_type="next_state"
    )
    val_denoising_dataset = DenoisingDataset(
        x_traj_list=val_x_traj_list,
        controls_list=val_controls_list,
        action_noise_factor=controls_std,
        state_noise_factor=x_traj_std,
        action_noise_multiplier=Config.ACTION_NOISE_MULTIPLIER,
        state_noise_multiplier=Config.STATE_NOISE_MULTIPLIER,
    )

    BATCH_SIZE = Config.BATCH_SIZE
    train_bc_dataloader = torch.utils.data.DataLoader(
        train_bc_dataset, batch_size=BATCH_SIZE, shuffle=True
    )
    train_denoising_dataloader = torch.utils.data.DataLoader(
        train_denoising_dataset, batch_size=BATCH_SIZE, shuffle=True
    )
    val_bc_dataloader = torch.utils.data.DataLoader(
        val_bc_dataset, batch_size=BATCH_SIZE, shuffle=False
    )
    val_denoising_dataloader = torch.utils.data.DataLoader(
        val_denoising_dataset, batch_size=BATCH_SIZE, shuffle=False
    )

    # Set up TensorBoard
    writer = SummaryWriter(Config.get_tensorboard_path(num_dems, random_seed))

    # Training loop
    for epoch in range(Config.EPOCH):
        model.train()
        denoising_model.train()
        total_bc_loss = 0
        total_denoising_loss = 0

        train_bar = tqdm(
            zip(train_bc_dataloader, train_denoising_dataloader),
            total=len(train_bc_dataloader),
            position=0,
            leave=True,
        )

        for batch_idx, (bc_batch, denoising_batch) in enumerate(train_bar):
            # Unpack BC batch
            states, actions_next_states = bc_batch
            states = states.to(device)
            actions_next_states = actions_next_states.to(device)

            # Train behavior cloning model
            predicted_actions = model(states)
            bc_loss = model.loss_func(predicted_actions, actions_next_states)

            # Unpack denoising batch
            noisy_actions_next_states, clean_actions_next_states = denoising_batch
            noisy_actions_next_states = noisy_actions_next_states.to(device)
            clean_actions_next_states = clean_actions_next_states.to(device)

            # Train denoising model
            denoised_actions_next_states = denoising_model(noisy_actions_next_states)
            denoising_loss = model.loss_func(
                denoised_actions_next_states, clean_actions_next_states
            )

            # Update both models
            optimizer_model.zero_grad()
            optimizer_denoising.zero_grad()
            bc_loss.backward()
            denoising_loss.backward()
            optimizer_model.step()
            optimizer_denoising.step()

            total_bc_loss += bc_loss.item()
            total_denoising_loss += denoising_loss.item()

            train_bar.set_description(
                f"Epoch {epoch}, BC Loss: {total_bc_loss / (batch_idx + 1):.4f}, Denoising Loss: {total_denoising_loss / (batch_idx + 1):.4f}"
            )

            # Log training losses
            writer.add_scalar(
                "Training/BC Loss",
                bc_loss.item(),
                epoch * len(train_bc_dataloader) + batch_idx,
            )
            writer.add_scalar(
                "Training/Denoising Loss",
                denoising_loss.item(),
                epoch * len(train_denoising_dataloader) + batch_idx,
            )

        # Validation
        model.eval()
        denoising_model.eval()
        val_bc_loss = 0
        val_denoising_loss = 0

        with torch.no_grad():
            for val_bc_batch, val_denoising_batch in zip(
                val_bc_dataloader, val_denoising_dataloader
            ):
                # Validate BC model
                val_states, val_actions_next_states = val_bc_batch
                val_states = val_states.to(device)
                val_actions_next_states = val_actions_next_states.to(device)
                val_predicted_actions = model(val_states)
                val_bc_loss += model.loss_func(
                    val_predicted_actions, val_actions_next_states
                ).item()

                # Validate denoising model
                val_noisy_actions_next_states, val_clean_actions_next_states = (
                    val_denoising_batch
                )
                val_noisy_actions_next_states = val_noisy_actions_next_states.to(device)
                val_clean_actions_next_states = val_clean_actions_next_states.to(device)
                val_denoised_actions_next_states = denoising_model(
                    val_noisy_actions_next_states
                )
                val_denoising_loss += model.loss_func(
                    val_denoised_actions_next_states, val_clean_actions_next_states
                ).item()

        val_bc_loss /= len(val_bc_dataloader)
        val_denoising_loss /= len(val_denoising_dataloader)

        print(
            f"Epoch {epoch}, Validation BC Loss: {val_bc_loss:.4f}, Validation Denoising Loss: {val_denoising_loss:.4f}"
        )

        # Log validation losses
        writer.add_scalar("Validation/BC Loss", val_bc_loss, epoch)
        writer.add_scalar("Validation/Denoising Loss", val_denoising_loss, epoch)

    if save_ckpt:
        # Save the trained models
        save_models(model, denoising_model, num_dems, random_seed, Config)

    print("Joint training completed and models saved.")
    writer.close()
    return model, denoising_model, train_x_traj_list, train_controls_list


def rollout_models(bc_model, denoising_model, initial_x_traj, num_steps):
    """
    This implements the joint denoising rollout 
    """
    x_trajs = []
    x_traj = to_tensor(initial_x_traj)
    for step in range(num_steps):
        x_trajs.append(x_traj)
        next_x_traj = bc_model(x_traj) # here the bc_model should output joint action and state
        if denoising_model is not None:
            # next_x_traj = denoising_model(torch.cat([x_traj, next_x_traj], dim=-1))
            next_x_traj = denoising_model(next_x_traj)
        # model system noise
        next_x_traj += torch.randn_like(next_x_traj) * 0.01
        x_traj = next_x_traj
    x_trajs.append(x_traj)
    return torch.stack(x_trajs)

def rollout_action_models(bc_model, denoising_model, initial_x_traj, num_steps):
    noise_std = np.array([f_g_diff_max, roll_max, pitch_max]) * 0.0
    x_trajs = []
    x_traj = to_tensor(initial_x_traj)
    for step in range(num_steps):
        x_trajs.append(x_traj)
        if denoising_model is not None:
            bc_input_state = denoising_model(x_traj)
        else:
            bc_input_state = x_traj
        action = bc_model(bc_input_state)
        applied_action = action + to_tensor(np.random.normal(0, noise_std))
        x_traj = x_traj.detach().cpu().numpy()
        applied_action = applied_action.detach().cpu().numpy()
        next_x_traj = get_next_step_state(x_traj, applied_action, DT)
        next_x_traj = to_tensor(next_x_traj)
        # model system noise
        next_x_traj += torch.randn_like(next_x_traj) * 0.025
        x_traj = next_x_traj
    
    x_trajs.append(x_traj)
    return torch.stack(x_trajs)

def distance_to_gt(x_trajs, gt_x_traj):
    """
    For each state in x_trajs, find the minimum distance to any state in gt_x_traj.
    Distance is measured as L2 norm (square root of squared L2 norm).
    """
    if isinstance(x_trajs, torch.Tensor):
        x_trajs = x_trajs.detach().cpu().numpy()
    if isinstance(gt_x_traj, torch.Tensor):
        gt_x_traj = gt_x_traj.detach().cpu().numpy()

    distances = []
    for x in x_trajs:
        # Calculate squared L2 norm between x and each gt state
        diffs = gt_x_traj - x
        squared_norms = np.sum(diffs * diffs, axis=1)
        # Take minimum distance and square root
        min_dist = np.sqrt(np.min(squared_norms))
        distances.append(min_dist)
    
    return np.array(distances)


def traj_to_numpy(x_trajs, x_trajs_denoising, gt_x_traj):
    if isinstance(x_trajs, torch.Tensor):
        x_trajs = x_trajs.detach().cpu().numpy()
        if x_trajs_denoising is not None:
            x_trajs_denoising = x_trajs_denoising.detach().cpu().numpy()
    if isinstance(gt_x_traj, torch.Tensor):
        gt_x_traj = gt_x_traj.detach().cpu().numpy()
    return x_trajs, x_trajs_denoising, gt_x_traj

# Modify the plot_denoising_rollout function to include gt_x_traj and gt_controls
def plot_denoising_rollout_with_gt(x_trajs, x_trajs_denoising, gt_x_traj, gt_controls):
    import matplotlib.pyplot as plt

    x_trajs, x_trajs_denoising, gt_x_traj = traj_to_numpy(x_trajs, x_trajs_denoising, gt_x_traj)

    num_steps, num_dims = x_trajs.shape
    num_control_dims = gt_controls.shape[1]

    fig, axs = plt.subplots(num_dims + num_control_dims, 1, figsize=(10, 3 * (num_dims + num_control_dims)), sharex=True)

    for dim in range(num_dims):
        axs[dim].plot(x_trajs[:, dim], label="BC rollout")
        if x_trajs_denoising is not None:
            axs[dim].plot(x_trajs_denoising[:, dim], label="BC + Denoising rollout")
        axs[dim].plot(gt_x_traj[:, dim], label="Ground truth", linestyle="--")
        axs[dim].set_ylabel(f"State Dim {dim}")
        axs[dim].grid(True)
        axs[dim].legend()
    
    for dim in range(num_control_dims):
        axs[num_dims + dim].plot(gt_controls[:, dim], label=f"Control Dim {dim}", linestyle=":")
        axs[num_dims + dim].set_ylabel(f"Control Dim {dim}")
        axs[num_dims + dim].grid(True)
        axs[num_dims + dim].legend()
    
    axs[-1].set_xlabel("Time steps")
    plt.tight_layout()
    plt.savefig("denoising_rollout_with_gt_plot.png")
    plt.close()
    
    # Plot 3D trajectory comparison
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot BC rollout trajectory
    ax.plot3D(x_trajs[:, 0], x_trajs[:, 1], x_trajs[:, 2], 
              label='BC rollout', linewidth=2)
    
    # Plot denoising rollout if available
    if x_trajs_denoising is not None:
        ax.plot3D(x_trajs_denoising[:, 0], x_trajs_denoising[:, 1], x_trajs_denoising[:, 2],
                 label='BC + Denoising rollout', linewidth=2)
    
    # Plot ground truth trajectory
    ax.plot3D(gt_x_traj[:, 0], gt_x_traj[:, 1], gt_x_traj[:, 2],
              label='Ground truth', linestyle='--', linewidth=2)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y') 
    ax.set_zlabel('Z')
    ax.legend()
    ax.grid(True)
    plt.title('3D Trajectory Comparison')
    plt.savefig("3d_trajectory_comparison.png")
    plt.close()
    
    # Compute and plot distances to gt
    distances_bc = distance_to_gt(x_trajs, gt_x_traj)
    
    print(f"Average distance of bc rollout to gt: {np.mean(distances_bc)}, max distance of bc rollout to gt: {np.max(distances_bc)}")
    if x_trajs_denoising is not None:
        distances_denoising = distance_to_gt(x_trajs_denoising, gt_x_traj)
        print(f"Average distance denoised rollout to gt: {np.mean(distances_denoising)}, max distance denoised rollout to gt: {np.max(distances_denoising)}")
    # Plot histogram of distances
    plt.figure(figsize=(10, 6))
    plt.hist(distances_bc, bins=30, alpha=0.5, label='BC rollout', density=True)
    if x_trajs_denoising is not None:
        plt.hist(distances_denoising, bins=30, alpha=0.5, label='BC + Denoising rollout', density=True)
    plt.xlabel('Distance to ground truth')
    plt.ylabel('Density')
    plt.title('Distribution of distances to ground truth trajectory')
    plt.legend()
    plt.grid(True)
    plt.savefig("distance_histogram.png")
    plt.close()

# Plot the denoising dataset points and their denoised counterparts
def plot_denoising_dataset_with_gt(denoising_dataset, gt_x_traj):
    fig, axs = plt.subplots(6, 1, figsize=(10, 18), sharex=True)
    
    # Plot ground truth trajectory
    for dim in range(6):
        axs[dim].plot(gt_x_traj[:, dim], label="Ground truth", color='black', linestyle='--')
    
    # Plot noisy and denoised points with arrows
    for i in range(len(denoising_dataset)):
        noisy_state, clean_state = denoising_dataset[i]
        for dim in range(6):
            noisy_y = noisy_state[dim].item()
            clean_y = clean_state[dim].item()
            axs[dim].scatter(i, noisy_y, color='red', alpha=0.5, s=10)
            axs[dim].scatter(i, clean_y, color='blue', alpha=0.5, s=10)
            axs[dim].annotate("", xy=(i, clean_y), xytext=(i, noisy_y),
                              arrowprops=dict(arrowstyle="->", color="green", alpha=0.3))
    
    for dim in range(6):
        axs[dim].set_ylabel(f"Dim {dim}")
        axs[dim].grid(True)
    
    axs[0].legend(['Ground truth', 'Noisy points', 'Clean points'])
    axs[-1].set_xlabel("Time steps")
    plt.tight_layout()
    plt.savefig("denoising_dataset_plot.png")
    plt.close()

def main():
    # number of demonstrations to train on
    num_dems = 50
    random_seed = 0
    config = Config()
    config.load_config_for_training("config.yaml")
    seedEverything(random_seed)
    # Train the model
    # next_state_model, denoising_model, train_x_traj_list, train_controls_list = train_model_joint(
    #     num_dems, random_seed, config, save_ckpt=False
    # )
    # # Save the trained models
    # torch.save(next_state_model.state_dict(), 'bc_model.pth')
    # torch.save(denoising_model.state_dict(), 'denoising_model.pth')

    # print("Models saved successfully.")
    # return


    train_controls_list, train_x_traj_list, val_controls_list, val_x_traj_list, controls_std, x_traj_std = get_data(num_dems)
    
    train_denoising_dataset = DenoisingDataset(
        x_traj_list=train_x_traj_list,
        controls_list=train_controls_list,
        action_noise_factor=controls_std,
        state_noise_factor=x_traj_std,
        action_noise_multiplier=Config.ACTION_NOISE_MULTIPLIER,
        state_noise_multiplier=Config.STATE_NOISE_MULTIPLIER,
    )


    # Call the function to create the plot
    # plot_denoising_dataset_with_gt(train_denoising_dataset, train_x_traj_list[0])
    # print("Denoising dataset plot saved as 'denoising_dataset_plot.png'")
    
    # Load the trained models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bc_model = Model(
        input_dim=6,
        output_dim=6,
        is_denoising_net=False,
        joint_action_state=False,
        action_loss_weight=Config.ACTION_LOSS_WEIGHT_BC,
        state_loss_weight=Config.STATE_LOSS_WEIGHT_BC,
    )
    # bc_model = BCModel(
    #     input_dim=6,
    #     output_dim=3,
    #     action_loss_weight=Config.ACTION_LOSS_WEIGHT_BC,
    #     state_loss_weight=Config.STATE_LOSS_WEIGHT_BC,
    # )
    denoising_model = Model(
        input_dim=6,
        output_dim=6,
        is_denoising_net=False,
        joint_action_state=False,
        action_loss_weight=Config.ACTION_LOSS_WEIGHT_DENOISING,
        state_loss_weight=Config.STATE_LOSS_WEIGHT_DENOISING,
    )
    bc_model.to(device)
    denoising_model.to(device)
    # bc_model.load_state_dict(torch.load("bc_action_model.pth"))
    bc_model.load_state_dict(torch.load("bc_model.pth"))
    denoising_model.load_state_dict(torch.load("denoising_model.pth"))
    bc_model.eval()
    denoising_model.eval()

    print("Models loaded successfully.")
    
    gt_x_traj = train_x_traj_list[0]
    gt_controls = train_controls_list[0]
    initial_x_traj = gt_x_traj[0] + 0.0 * np.random.randn(*gt_x_traj[0].shape)
    num_steps = gt_x_traj.shape[0]
    x_trajs = rollout_models(bc_model, None, initial_x_traj, num_steps)
    # x_trajs = rollout_action_models(bc_model, None, initial_x_traj, num_steps)
    x_trajs_denoising = rollout_models(
        bc_model, denoising_model, initial_x_traj, num_steps
    )
    # x_trajs_denoising = rollout_action_models(bc_model, denoising_model, initial_x_traj, num_steps)
    plot_denoising_rollout_with_gt(x_trajs, x_trajs_denoising, gt_x_traj, gt_controls)


if __name__ == "__main__":
    # mp.set_start_method('spawn')
    main()
