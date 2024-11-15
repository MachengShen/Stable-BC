import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset
import pickle
import matplotlib.pyplot as plt
import datetime
from utils import seedEverything, get_statistics, save_models
from tqdm import tqdm
import os, sys
from torch.utils.tensorboard import SummaryWriter
import datetime
from dataset_utils import BCDataset, DenoisingDataset, StateImitationDataset, load_data
from models import MLP

# stable bc training, may require a dynamics model for each task
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
    def dynamics_model(x, u):
        raise NotImplementedError("Dynamics model not implemented")
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


def loss_func(y_true, y_pred, action_dim, state_dim):
    assert y_true.shape[1] == action_dim + state_dim
    assert y_pred.shape[1] == action_dim + state_dim
    loss = nn.MSELoss(reduction="mean")
    action_loss = loss(y_true[:, :action_dim], y_pred[:, :action_dim])
    state_loss = loss(y_true[:, action_dim:], y_pred[:, action_dim:])
    return action_loss, state_loss

def train_model_joint(num_dems, random_seed, Config, save_ckpt=True):
    # Load data based on the task type
    controls_list, x_traj_list = load_data(Config)
    print("stats before normalization")
    controls_mean, controls_std, x_traj_mean, x_traj_std = get_statistics(controls_list, x_traj_list)

    controls_list = [((controls - controls_mean) / controls_std) for controls in controls_list]
    x_traj_list = [((x_traj - x_traj_mean) / x_traj_std) for x_traj in x_traj_list]
    print("stats after normalization")
    _, controls_std, _, x_traj_std = get_statistics(controls_list, x_traj_list)
    
    # randomly select num_dems indices
    seedEverything(random_seed)
    indices = np.random.choice(len(controls_list), num_dems, replace=False)
    controls_list = [controls_list[i] for i in indices]
    x_traj_list = [x_traj_list[i] for i in indices]

    # Get input and output dimensions based on the data
    sample_state = x_traj_list[0][0]
    sample_action = controls_list[0][0]
    state_dim = len(sample_state)
    action_dim = len(sample_action)

    # Split data into train and validation sets
    split_factor = 0.8
    train_size = int(len(controls_list) * split_factor)
    
    train_controls_list = controls_list[:train_size]
    train_x_traj_list = x_traj_list[:train_size]
    val_controls_list = controls_list[train_size:]
    val_x_traj_list = x_traj_list[train_size:]

    # Set up the models and optimizers with dynamic input/output dimensions
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLP(
        input_dim=state_dim, 
        output_dim=action_dim + state_dim, 
    )
    denoising_model = MLP(
        input_dim=action_dim + state_dim * 2, 
        output_dim=action_dim + state_dim, 
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

    train_bc_dataloader = torch.utils.data.DataLoader(
        train_bc_dataset, batch_size=Config.BATCH_SIZE, shuffle=True
    )
    train_denoising_dataloader = torch.utils.data.DataLoader(
        train_denoising_dataset, batch_size=Config.BATCH_SIZE, shuffle=True
    )
    val_bc_dataloader = torch.utils.data.DataLoader(
        val_bc_dataset, batch_size=Config.BATCH_SIZE, shuffle=False
    )
    val_denoising_dataloader = torch.utils.data.DataLoader(
        val_denoising_dataset, batch_size=Config.BATCH_SIZE, shuffle=False
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
            bc_action_loss, bc_state_loss = loss_func(actions_next_states, predicted_actions, action_dim, state_dim)
            bc_loss = Config.ACTION_LOSS_WEIGHT_BC * bc_action_loss + Config.STATE_LOSS_WEIGHT_BC * bc_state_loss

            # Unpack denoising batch
            noisy_actions_next_states, clean_actions_next_states = denoising_batch
            noisy_actions_next_states = noisy_actions_next_states.to(device)
            clean_actions_next_states = clean_actions_next_states.to(device)

            # Train denoising model
            denoised_actions_next_states = denoising_model(noisy_actions_next_states)
            denoising_action_loss, denoising_state_loss = loss_func(
                clean_actions_next_states, denoised_actions_next_states, action_dim, state_dim
            )
            denoising_loss = Config.ACTION_LOSS_WEIGHT_DENOISING * denoising_action_loss + Config.STATE_LOSS_WEIGHT_DENOISING * denoising_state_loss

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
            writer.add_scalar('Training/BC Action Loss', bc_action_loss.item(), epoch * len(train_bc_dataloader) + batch_idx)
            writer.add_scalar('Training/BC State Loss', bc_state_loss.item(), epoch * len(train_bc_dataloader) + batch_idx)
            writer.add_scalar('Training/Denoising Loss', denoising_loss.item(), epoch * len(train_denoising_dataloader) + batch_idx)
            writer.add_scalar('Training/Denoising Action Loss', denoising_action_loss.item(), epoch * len(train_denoising_dataloader) + batch_idx)
            writer.add_scalar('Training/Denoising State Loss', denoising_state_loss.item(), epoch * len(train_denoising_dataloader) + batch_idx)

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
                val_bc_action_loss, val_bc_state_loss = loss_func(val_predicted_actions, val_actions_next_states, action_dim, state_dim)
                val_bc_loss = Config.ACTION_LOSS_WEIGHT_BC * val_bc_action_loss + Config.STATE_LOSS_WEIGHT_BC * val_bc_state_loss
                val_bc_loss += val_bc_loss

                # Validate denoising model
                val_noisy_actions_next_states, val_clean_actions_next_states = val_denoising_batch
                val_noisy_actions_next_states = val_noisy_actions_next_states.to(device)
                val_clean_actions_next_states = val_clean_actions_next_states.to(device)
                val_denoised_actions_next_states = denoising_model(val_noisy_actions_next_states)
                val_denoising_action_loss, val_denoising_state_loss = loss_func(val_denoised_actions_next_states, val_clean_actions_next_states, action_dim, state_dim)
                val_denoising_loss = Config.ACTION_LOSS_WEIGHT_DENOISING * val_denoising_action_loss + Config.STATE_LOSS_WEIGHT_DENOISING * val_denoising_state_loss
                val_denoising_loss += val_denoising_loss

        val_bc_loss /= len(val_bc_dataloader)
        val_denoising_loss /= len(val_denoising_dataloader)

        print(f"Epoch {epoch}, Validation BC Loss: {val_bc_loss:.4f}, Validation Denoising Loss: {val_denoising_loss:.4f}")

        # Log validation losses
        writer.add_scalar('Validation/BC Loss', val_bc_loss, epoch)
        writer.add_scalar('Validation/BC Action Loss', val_bc_action_loss, epoch)
        writer.add_scalar('Validation/BC State Loss', val_bc_state_loss, epoch)
        writer.add_scalar('Validation/Denoising Loss', val_denoising_loss, epoch)
        writer.add_scalar('Validation/Denoising Action Loss', val_denoising_action_loss, epoch)
        writer.add_scalar('Validation/Denoising State Loss', val_denoising_state_loss, epoch)

    if save_ckpt:
        # Save the trained models
        save_models(model, denoising_model, num_dems, random_seed, Config)

    print("Joint training completed and models saved.")
    writer.close()
    return model, denoising_model
