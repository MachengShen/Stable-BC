import torch
import numpy as np
from torch.utils.data import Dataset
from models import MyModel
import pickle
import matplotlib.pyplot as plt
import datetime
from utils import seedEverything
from tqdm import tqdm
import os, sys
from torch.utils.tensorboard import SummaryWriter
import datetime
from config import Config

A_G = 9.81


#  write x_dot function with torch
def dynamics_model(x, u):
    x_dot = torch.zeros_like(x)
    x_dot[:, 0] = x[:, 3]
    x_dot[:, 1] = x[:, 4]
    x_dot[:, 2] = x[:, 5]
    x_dot[:, 3] = A_G * torch.tan(u[:, 2])
    x_dot[:, 4] = -A_G * torch.tan(u[:, 1])
    x_dot[:, 5] = u[:, 0] - A_G
    return x_dot


class StateImitationDataset(Dataset):
    def __init__(self, x_traj_array, controls_array, type=0):
        self.type = type
        self.x_traj_array = x_traj_array.astype(np.float32)
        self.controls_array = controls_array.astype(np.float32)

    def __len__(self):
        return self.controls_array.shape[0]

    def __getitem__(self, index):
        return self.x_traj_array[index], self.controls_array[index]


NUM_ACTIONS = 3


def to_tensor(array, device="cuda" if torch.cuda.is_available() else "cpu"):
    return torch.tensor(array, dtype=torch.float32).to(device)


class BCDataset(Dataset):
    def __init__(self, x_traj_list, controls_list):
        self.x_traj_list = x_traj_list
        self.controls_list = controls_list

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

        input_tensor = to_tensor(
            np.concatenate([clean_x_t, noisy_a_t, noisy_x_t_plus_1])
        )
        output_tensor = to_tensor(np.concatenate([clean_a_t, clean_x_t_plus_1]))

        return input_tensor, output_tensor


def train_model(
    num_dems,
    type,
    train_dataloader,
    valid_dataloader,
    savename,
    EPOCH=1000,
    LR=0.0001,
    stability_loss_coef=0.1,
    models_path=None,
):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MyModel()
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)

    relu = torch.nn.ReLU()

    for epoch in range(EPOCH):

        # validation
        model.eval()
        total_test_loss = 0
        for i, data in enumerate(valid_dataloader):
            states = data[0]
            actions = data[1]
            states = states.to(device)
            actions = actions.to(device)
            outputs = model(states)
            test_loss = model.loss_func(actions, outputs)
            total_test_loss += test_loss.item()
        validdation_loss_per_sample = total_test_loss / len(valid_dataloader)
        print(f"Epoch {epoch} Test Loss: { validdation_loss_per_sample }")

        model.train()
        total_loss = 0
        total_loss_bc = 0
        total_loss_stability = 0

        train_bar = tqdm(train_dataloader, position=0, leave=True)

        for batch, data in enumerate(train_bar):
            states = data[0]
            actions = data[1]
            states = states.to(device)
            actions = actions.to(device)

            STATE_DIM = states.shape[1]
            X_DIM = STATE_DIM

            if type == 0:
                # get mse loss
                loss = model.loss_func(actions, model(states))

            elif type == 1:
                # get the overall matrix for delta_x F(x, pi(x, y))
                # this is a shortcut for the top left matrix in A

                states.requires_grad = True
                outputs = model(states)
                loss_bc = model.loss_func(actions, outputs)
                F = dynamics_model(states, outputs)

                # get the gradient of a wrt states using automatic differentiation
                J = torch.zeros((states.shape[0], STATE_DIM, STATE_DIM), device=device)
                for i in range(STATE_DIM):
                    J[:, i] = torch.autograd.grad(
                        F[:, i],
                        states,
                        grad_outputs=torch.ones_like(F[:, i], device=device),
                        create_graph=True,
                    )[0]
                J = J[:, :, 0:X_DIM]

                # get the eigenvalues of the matrix
                E = torch.linalg.eigvals(J).real
                # loss is the sum of positive eigenvalues
                loss_stability = stability_loss_coef * torch.sum(relu(E))  # + EPSILON)

                loss = loss_bc + loss_stability

            # update model parameters
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if type == 0:
                total_loss += loss.item()
                train_bar.set_description(
                    "Train iteration (epoch {}): [{}] Loss: {:.4f}".format(
                        epoch, batch, total_loss / (batch + 1)
                    )
                )
            else:
                total_loss += loss.item()
                total_loss_bc += loss_bc.item()
                total_loss_stability += loss_stability.item()
                train_bar.set_description(
                    "Train iteration (epoch {}): [{}] Loss: {:.4f}, BC Loss: {:.4f}, Stability Loss: {:.4f}".format(
                        epoch,
                        batch,
                        total_loss / (batch + 1),
                        total_loss_bc / (batch + 1),
                        total_loss_stability / (batch + 1),
                    )
                )

        n_training_samples = len(train_dataloader)

    if not os.path.exists(models_path):
        os.makedirs(models_path)

    torch.save(model.state_dict(), models_path + "/" + savename)


def train_imitation_agent(num_dems, type: int, random_seed, Config):
    EPOCH = Config.EPOCH
    LR = Config.LR
    
    if type == 2:  # Joint training
        train_model_joint(num_dems, random_seed, Config)
    else:
        stability_loss_coef = Config.STABILITY_LOSS_COEF
        import os

        # Change the working directory to the sim-quadrotor folder
        os.chdir(os.path.dirname(os.path.abspath(__file__)))

        data_dict = pickle.load(open("sim-quadrotor/data/data_0.pkl", "rb"))
        controls_list = data_dict["controls_list"]
        x_traj_list = data_dict["x_trajectories_list"]

        # randomly select num_dems indices
        seedEverything(random_seed)
        indices = np.random.choice(len(controls_list), num_dems, replace=False)
        controls_array = np.concatenate([controls_list[i] for i in indices])
        x_traj_array = np.concatenate([np.stack(x_traj_list[i]) for i in indices])

        del data_dict

        # split the data into training and testing
        train_x_traj_array = x_traj_array[: int(0.8 * x_traj_array.shape[0])]
        test_x_traj_array = x_traj_array[int(0.8 * x_traj_array.shape[0]) :]
        train_controls_array = controls_array[: int(0.8 * controls_array.shape[0])]
        test_controls_array = controls_array[int(0.8 * controls_array.shape[0]) :]

        train_dataset = StateImitationDataset(train_x_traj_array, train_controls_array)
        test_dataset = StateImitationDataset(test_x_traj_array, test_controls_array)

        batch_size = int(len(train_dataset) / 10)

        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        test_dataloader = torch.utils.data.DataLoader(
            test_dataset, batch_size=batch_size, shuffle=True
        )

        models_path = f"sim-quadrotor/results_{LR}lr_{EPOCH}epoch/lamda_{stability_loss_coef}/{num_dems}dems/{random_seed}"

        savename = f"im_model{type}.pt"
        train_model(
            num_dems,
            type,
            train_dataloader,
            test_dataloader,
            savename,
            EPOCH=EPOCH,
            LR=LR,
            stability_loss_coef=stability_loss_coef,
            models_path=models_path,
        )


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


def train_model_joint(num_dems, random_seed, Config, save_ckpt=True):
    data_dict = pickle.load(open("sim-quadrotor/data/data_0.pkl", "rb"))
    controls_list = data_dict["controls_list"]
    x_traj_list = data_dict["x_trajectories_list"]

    controls_std, x_traj_std = get_statistics(controls_list, x_traj_list)

    # randomly select num_dems indices
    seedEverything(random_seed)
    indices = np.random.choice(len(controls_list), num_dems, replace=False)
    controls_list = [controls_list[i] for i in indices]
    x_traj_list = [x_traj_list[i] for i in indices]

    # Split data into train and validation sets
    split_factor = 0.8
    train_size = int(len(controls_list) * split_factor)
    
    train_controls_list = controls_list[:train_size]
    train_x_traj_list = x_traj_list[:train_size]
    val_controls_list = controls_list[train_size:]
    val_x_traj_list = x_traj_list[train_size:]

    # Set up the models and optimizers
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MyModel(
        input_dim=6, output_dim=9, is_denoising_net=False, joint_action_state=True,
        action_loss_weight=Config.ACTION_LOSS_WEIGHT_BC, state_loss_weight=Config.STATE_LOSS_WEIGHT_BC
    )
    denoising_model = MyModel(
        input_dim=15, output_dim=9, is_denoising_net=True, joint_action_state=True,
        action_loss_weight=Config.ACTION_LOSS_WEIGHT_DENOISING, state_loss_weight=Config.STATE_LOSS_WEIGHT_DENOISING
    )
    model.to(device)
    denoising_model.to(device)

    optimizer_model = torch.optim.Adam(model.parameters(), lr=Config.LR, weight_decay=1e-5)
    optimizer_denoising = torch.optim.Adam(
        denoising_model.parameters(), lr=Config.LR, weight_decay=1e-5
    )

    # Load and prepare datasets
    train_bc_dataset = BCDataset(x_traj_list=train_x_traj_list, controls_list=train_controls_list)
    train_denoising_dataset = DenoisingDataset(
        x_traj_list=train_x_traj_list,
        controls_list=train_controls_list,
        action_noise_factor=controls_std,
        state_noise_factor=x_traj_std,
        action_noise_multiplier=Config.ACTION_NOISE_MULTIPLIER,
        state_noise_multiplier=Config.STATE_NOISE_MULTIPLIER
    )
    val_bc_dataset = BCDataset(x_traj_list=val_x_traj_list, controls_list=val_controls_list)
    val_denoising_dataset = DenoisingDataset(
        x_traj_list=val_x_traj_list,
        controls_list=val_controls_list,
        action_noise_factor=controls_std,
        state_noise_factor=x_traj_std,
        action_noise_multiplier=Config.ACTION_NOISE_MULTIPLIER,
        state_noise_multiplier=Config.STATE_NOISE_MULTIPLIER
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
            writer.add_scalar('Training/BC Loss', bc_loss.item(), epoch * len(train_bc_dataloader) + batch_idx)
            writer.add_scalar('Training/Denoising Loss', denoising_loss.item(), epoch * len(train_denoising_dataloader) + batch_idx)

        # Validation
        model.eval()
        denoising_model.eval()
        val_bc_loss = 0
        val_denoising_loss = 0

        with torch.no_grad():
            for val_bc_batch, val_denoising_batch in zip(val_bc_dataloader, val_denoising_dataloader):
                # Validate BC model
                val_states, val_actions_next_states = val_bc_batch
                val_states = val_states.to(device)
                val_actions_next_states = val_actions_next_states.to(device)
                val_predicted_actions = model(val_states)
                val_bc_loss += model.loss_func(val_predicted_actions, val_actions_next_states).item()

                # Validate denoising model
                val_noisy_actions_next_states, val_clean_actions_next_states = val_denoising_batch
                val_noisy_actions_next_states = val_noisy_actions_next_states.to(device)
                val_clean_actions_next_states = val_clean_actions_next_states.to(device)
                val_denoised_actions_next_states = denoising_model(val_noisy_actions_next_states)
                val_denoising_loss += model.loss_func(val_denoised_actions_next_states, val_clean_actions_next_states).item()

        val_bc_loss /= len(val_bc_dataloader)
        val_denoising_loss /= len(val_denoising_dataloader)

        print(f"Epoch {epoch}, Validation BC Loss: {val_bc_loss:.4f}, Validation Denoising Loss: {val_denoising_loss:.4f}")

        # Log validation losses
        writer.add_scalar('Validation/BC Loss', val_bc_loss, epoch)
        writer.add_scalar('Validation/Denoising Loss', val_denoising_loss, epoch)

    if save_ckpt:
        # Save the trained models
        models_path = Config.get_model_path(num_dems, random_seed)
        if not os.path.exists(models_path):
            os.makedirs(models_path)

        torch.save(model.state_dict(), f"{models_path}/joint_bc_model.pt")
        torch.save(denoising_model.state_dict(), f"{models_path}/joint_denoising_model.pt")

    print("Joint training completed and models saved.")
    writer.close()
    return model, denoising_model
