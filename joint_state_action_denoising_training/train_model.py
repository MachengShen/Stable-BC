import torch
import numpy as np
from torch.utils.data import Dataset
from models import MyModel
import pickle
import matplotlib.pyplot as plt
import datetime
from utils import seedEverything, get_statistics, save_models
from tqdm import tqdm
import os, sys
from torch.utils.tensorboard import SummaryWriter
import datetime
from config import Config
from dataset_utils import BCDataset, DenoisingDataset, StateImitationDataset

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


def train_model_joint(num_dems, random_seed, Config, save_ckpt=True):
    # Load data based on the task type
    if Config.TASK_TYPE == "sim-quadrotor":
        data_dict = pickle.load(open("sim-quadrotor/data/data_0.pkl", "rb"))
        controls_list = data_dict["controls_list"]
        x_traj_list = data_dict["x_trajectories_list"]
    
    elif Config.TASK_TYPE == "sim-intersection":
        data_dict = pickle.load(open("sim-intersection/data/data_0.pkl", "rb"))
        controls_list = data_dict["controls_list"]
        x_traj_list = data_dict["x_trajectories_list"]
    
    elif Config.TASK_TYPE == "CCIL":
        # Load CCIL data using the read_data function
        data = read_data(Config.CCIL_DATA_DIR)
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

    controls_std, x_traj_std = get_statistics(controls_list, x_traj_list)

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
    model = MyModel(
        input_dim=state_dim, 
        output_dim=action_dim + state_dim, 
        is_denoising_net=False, 
        joint_action_state=True,
        action_loss_weight=Config.ACTION_LOSS_WEIGHT_BC, 
        state_loss_weight=Config.STATE_LOSS_WEIGHT_BC
    )
    denoising_model = MyModel(
        input_dim=action_dim + state_dim * 2, 
        output_dim=action_dim + state_dim, 
        is_denoising_net=True, 
        joint_action_state=True,
        action_loss_weight=Config.ACTION_LOSS_WEIGHT_DENOISING, 
        state_loss_weight=Config.STATE_LOSS_WEIGHT_DENOISING
    )

    # Rest of the function remains the same...
    # ...
