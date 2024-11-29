import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset
import pickle
import matplotlib.pyplot as plt
import datetime
from utils import seedEverything, get_statistics, save_models, save_normalization_stats, evaluate_model
from tqdm import tqdm
import os, sys
from torch.utils.tensorboard import SummaryWriter
import datetime
from dataset_utils import BCDataset, DenoisingDataset, StateImitationDataset, load_data, get_delta_x_traj_list
from models import MLP
from diffusion_model import DiffusionPolicy
from policy_agents import JointStateActionAgent, BaselineBCAgent, RandomAgent, DiffusionPolicyAgent
from ccil_utils import load_env
# import torch.multiprocessing as mp

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

def loss_func_state_only_bc(y_true, y_pred, state_dim):
    assert y_true.shape[1] == state_dim
    assert y_pred.shape[1] == state_dim
    loss = nn.MSELoss(reduction="mean")
    state_loss = loss(y_true, y_pred)
    return state_loss


def get_dataloader_kwargs(device, Config):
    """Helper function to get consistent DataLoader settings"""
    return {
        'batch_size': Config.BATCH_SIZE,
        # 'num_workers': 1,
        # 'pin_memory': not (device.type == 'cuda'),
        # 'persistent_workers': True
    }

def train_model_joint(num_dems, random_seed, Config, save_ckpt=True, predict_state_delta=False, state_only_bc=False):
    """
    Train joint model with multiple options:
    1. BC predicts [a_t, x_t+1] or [a_t, delta_x_t], denoiser predicts clean version from noisy inputs
    2. BC predicts x_t+1 or delta_x_t only, denoiser predicts [a_t, x_t+1] or [a_t, delta_x_t] from [x_t, noisy_x_t+1]
    """
    # Load data based on the task type
    controls_list, x_traj_list = load_data(Config)
    
    # Get state deltas if needed
    if predict_state_delta:
        delta_x_traj_list = get_delta_x_traj_list(x_traj_list)
        x_traj_list = delta_x_traj_list
    
    print("stats before normalization")
    controls_mean, controls_std, x_traj_mean, x_traj_std = get_statistics(controls_list, x_traj_list, is_delta=predict_state_delta)
    save_normalization_stats(controls_mean, controls_std, x_traj_mean, x_traj_std, num_dems, random_seed, Config)

    controls_list = [((controls - controls_mean) / (controls_std + 1e-8)) for controls in controls_list]
    x_traj_list = [((x_traj - x_traj_mean) / (x_traj_std + 1e-8)) for x_traj in x_traj_list]
    print("stats after normalization")
    _, controls_std, _, x_traj_std = get_statistics(controls_list, x_traj_list, is_delta=predict_state_delta)
    
    # randomly select num_dems indices
    seedEverything(random_seed)
    num_dems = min(num_dems, len(controls_list))
    indices = np.random.choice(len(controls_list), min(num_dems, len(controls_list)), replace=False)
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
        output_dim=state_dim if state_only_bc else action_dim + state_dim, 
    )
    denoising_model = MLP(
        input_dim=state_dim * 2 if state_only_bc else action_dim + state_dim * 2, 
        output_dim=action_dim + state_dim, 
    )

    model.to(device)
    denoising_model.to(device)

    optimizer_model = torch.optim.Adam(model.parameters(), 
                                     lr=Config.LR, 
                                     weight_decay=Config.WEIGHT_DECAY)
    optimizer_denoising = torch.optim.Adam(
        denoising_model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY
    )

    # Load and prepare datasets
    train_bc_dataset = BCDataset(x_traj_list=train_x_traj_list, controls_list=train_controls_list, state_only=state_only_bc)
    train_denoising_dataset = DenoisingDataset(
        x_traj_list=train_x_traj_list,
        controls_list=train_controls_list,
        action_noise_factor=controls_std,
        state_noise_factor=x_traj_std,
        action_noise_multiplier=Config.ACTION_NOISE_MULTIPLIER,
        state_noise_multiplier=Config.STATE_NOISE_MULTIPLIER,
        input_state_only=state_only_bc
    )
    val_bc_dataset = BCDataset(
        x_traj_list=val_x_traj_list, 
        controls_list=val_controls_list,
        state_only=state_only_bc
    )
    val_denoising_dataset = DenoisingDataset(
        x_traj_list=val_x_traj_list,
        controls_list=val_controls_list,
        action_noise_factor=controls_std,
        state_noise_factor=x_traj_std,
        action_noise_multiplier=Config.ACTION_NOISE_MULTIPLIER,
        state_noise_multiplier=Config.STATE_NOISE_MULTIPLIER,
        input_state_only=state_only_bc
    )

    # Get DataLoader settings
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader_kwargs = get_dataloader_kwargs(device, Config)

    train_bc_dataloader = torch.utils.data.DataLoader(
        train_bc_dataset, 
        shuffle=True,
        **loader_kwargs
    )
    train_denoising_dataloader = torch.utils.data.DataLoader(
        train_denoising_dataset, 
        shuffle=True,
        **loader_kwargs
    )
    val_bc_dataloader = torch.utils.data.DataLoader(
        val_bc_dataset, 
        shuffle=False,
        **loader_kwargs
    )
    val_denoising_dataloader = torch.utils.data.DataLoader(
        val_denoising_dataset, 
        shuffle=False,
        **loader_kwargs
    )

    # Set up TensorBoard with delta state notation
    state_type = "delta_state" if predict_state_delta else "next_state"
    if state_only_bc:
        state_type = "state_only_bc"
    writer = SummaryWriter(Config.get_tensorboard_path(num_dems, random_seed) + f"_{state_type}")

    # Initialize environment for evaluation
    env, meta_env = load_env(Config)
    env.seed(random_seed)
    
    mean_rewards = {'joint_bc': [], 'denoising_joint_bc': []}
    
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
            if state_only_bc:
                # here action_next_states and predicted_actions are actually next_states
                bc_state_loss = loss_func_state_only_bc(actions_next_states, predicted_actions, state_dim)
                bc_loss = bc_state_loss
            else:
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

            # Log training losses with clear state type notation
            writer.add_scalar(f'Training/BC_{state_type}_Loss', bc_state_loss.item(), epoch * len(train_bc_dataloader) + batch_idx)
            if not state_only_bc:
                writer.add_scalar('Training/BC_Action_Loss', bc_action_loss.item(), epoch * len(train_bc_dataloader) + batch_idx)
            writer.add_scalar(f'Training/Denoising_{state_type}_Loss', denoising_state_loss.item(), epoch * len(train_denoising_dataloader) + batch_idx)
            writer.add_scalar('Training/Denoising_Action_Loss', denoising_action_loss.item(), epoch * len(train_denoising_dataloader) + batch_idx)

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
                if state_only_bc:
                    val_bc_state_loss = loss_func_state_only_bc(val_actions_next_states, val_predicted_actions, state_dim)
                    val_bc_loss = val_bc_state_loss
                else:
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

        # Log validation losses with clear state type notation
        writer.add_scalar(f'Validation/BC_{state_type}_Loss', val_bc_state_loss, epoch)
        if not state_only_bc:
            writer.add_scalar('Validation/BC_Action_Loss', val_bc_action_loss, epoch)
        writer.add_scalar(f'Validation/Denoising_{state_type}_Loss', val_denoising_state_loss, epoch)
        writer.add_scalar('Validation/Denoising_Action_Loss', val_denoising_action_loss, epoch)

        # Periodic evaluation (every 100 epochs or as specified in config)
        if epoch % (Config.EVAL_INTERVAL_FACTOR * Config.EPOCH) == 0:
            if not state_only_bc: # state_only_bc requires denoising model
                # Evaluate BC model
                bc_agent = JointStateActionAgent(
                model, None, device, action_dim, stats_path=Config.get_model_path(num_dems, random_seed)
                )
                bc_results = evaluate_model(bc_agent, env, meta_env, device)
            
            # Evaluate denoising model
            denoising_agent = JointStateActionAgent(
                model, denoising_model, device, action_dim, stats_path=Config.get_model_path(num_dems, random_seed), state_only_bc=state_only_bc
            )
            denoising_results = evaluate_model(denoising_agent, env, meta_env, device)
            
            if not state_only_bc:
                mean_rewards['joint_bc'].append(bc_results[0.0]['mean_reward'])
            mean_rewards['denoising_joint_bc'].append(denoising_results[0.0]['mean_reward'])
            
            # Log evaluation results with state type notation
            for noise in denoising_results:
                if not state_only_bc:
                    writer.add_scalar(f'Eval/BC_{state_type}_Mean_Reward_{noise}', bc_results[noise]['mean_reward'], epoch)
                    writer.add_scalar(f'Eval/BC_{state_type}_Success_Rate_{noise}', bc_results[noise]['success_rate'], epoch)
                writer.add_scalar(f'Eval/Denoising_{state_type}_Mean_Reward_{noise}', denoising_results[noise]['mean_reward'], epoch)
                writer.add_scalar(f'Eval/Denoising_{state_type}_Success_Rate_{noise}', denoising_results[noise]['success_rate'], epoch)

    if save_ckpt:
        # Save the trained models
        model_surfix = "_state_only" if state_only_bc else ""
        save_models(model, denoising_model, num_dems, random_seed, Config, model_surfix)

    print("Joint training completed and models saved.")
    writer.close()
    return model, denoising_model, mean_rewards


def train_baseline_bc(num_dems, random_seed, Config):
    """Train a baseline BC model that only maps states to actions"""
    # Load and normalize data
    controls_list, x_traj_list = load_data(Config)
    print("stats before normalization")
    controls_mean, controls_std, x_traj_mean, x_traj_std = get_statistics(controls_list, x_traj_list)
    save_normalization_stats(controls_mean, controls_std, x_traj_mean, x_traj_std, num_dems, random_seed, Config)

    controls_list = [((controls - controls_mean) / (controls_std + 1e-8)) for controls in controls_list]
    x_traj_list = [((x_traj - x_traj_mean) / (x_traj_std + 1e-8)) for x_traj in x_traj_list]
    
    # Randomly select demonstrations
    seedEverything(random_seed)
    indices = np.random.choice(len(controls_list), min(num_dems, len(controls_list)), replace=False)
    controls_list = [controls_list[i] for i in indices]
    x_traj_list = [x_traj_list[i] for i in indices]

    # Get dimensions
    state_dim = len(x_traj_list[0][0])
    action_dim = len(controls_list[0][0])

    # Split data
    split_factor = 0.8
    train_size = int(len(controls_list) * split_factor)
    
    train_controls_list = controls_list[:train_size]
    train_x_traj_list = x_traj_list[:train_size]
    val_controls_list = controls_list[train_size:]
    val_x_traj_list = x_traj_list[train_size:]

    # Create datasets
    train_dataset = BCDataset(x_traj_list=train_x_traj_list, controls_list=train_controls_list, action_only=True)
    val_dataset = BCDataset(x_traj_list=val_x_traj_list, controls_list=val_controls_list, action_only=True)

    # Get DataLoader settings
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader_kwargs = get_dataloader_kwargs(device, Config)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, 
        shuffle=True,
        **loader_kwargs
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, 
        shuffle=False,
        **loader_kwargs
    )

    # Initialize model and optimizer with weight decay
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLP(input_dim=state_dim, output_dim=action_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), 
                               lr=Config.LR, 
                               weight_decay=Config.WEIGHT_DECAY)
    loss_fn = nn.MSELoss()

    # Set up tensorboard
    writer = SummaryWriter(Config.get_tensorboard_path(num_dems, random_seed) + "_baseline")

    # Initialize environment for evaluation
    env, meta_env = load_env(Config)
    env.seed(random_seed)
    
    # Training loop
    for epoch in range(Config.EPOCH):
        model.train()
        total_train_loss = 0
        
        train_bar = tqdm(train_dataloader, position=0, leave=True)
        for batch_idx, (states, actions) in enumerate(train_bar):
            states = states.to(device)
            actions = actions.to(device)
            
            pred_actions = model(states)
            loss = loss_fn(actions, pred_actions)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            train_bar.set_description(
                f"Epoch {epoch}, Train Loss: {total_train_loss / (batch_idx + 1):.4f}"
            )
            
            writer.add_scalar('Training/Loss', loss.item(), 
                            epoch * len(train_dataloader) + batch_idx)

        # Validation
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for states, actions in val_dataloader:
                states = states.to(device)
                actions = actions.to(device)
                pred_actions = model(states)
                loss = loss_fn(actions, pred_actions)
                total_val_loss += loss.item()
        
        avg_val_loss = total_val_loss / len(val_dataloader)
        writer.add_scalar('Validation/Loss', avg_val_loss, epoch)
        print(f"Epoch {epoch}, Validation Loss: {avg_val_loss:.4f}")

        # Periodic evaluation
        if epoch % (Config.EVAL_INTERVAL_FACTOR * Config.EPOCH) == 0:
            baseline_agent = BaselineBCAgent(
                model, device, action_dim, stats_path=Config.get_model_path(num_dems, random_seed)
            )
            results = evaluate_model(baseline_agent, env, meta_env, device)
            
            # Log evaluation results
            for noise in results:
                writer.add_scalar(f'Eval/Mean_Reward_{noise}', results[noise]['mean_reward'], epoch)
                writer.add_scalar(f'Eval/Success_Rate_{noise}', results[noise]['success_rate'], epoch)

    # Save model
    save_path = Config.get_model_path(num_dems, random_seed)
    os.makedirs(save_path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_path, "baseline_bc_model.pt"))
    
    writer.close()
    return model


def train_diffusion_policy(num_dems, random_seed, Config):
    """Train a diffusion policy that predicts joint [action, next_state]"""
    # Load and normalize data
    controls_list, x_traj_list = load_data(Config)
    print("stats before normalization")
    controls_mean, controls_std, x_traj_mean, x_traj_std = get_statistics(controls_list, x_traj_list)
    save_normalization_stats(controls_mean, controls_std, x_traj_mean, x_traj_std, num_dems, random_seed, Config)

    controls_list = [((controls - controls_mean) / (controls_std + 1e-8)) for controls in controls_list]
    x_traj_list = [((x_traj - x_traj_mean) / (x_traj_std + 1e-8)) for x_traj in x_traj_list]
    
    # Randomly select demonstrations
    seedEverything(random_seed)
    indices = np.random.choice(len(controls_list), min(num_dems, len(controls_list)), replace=False)
    controls_list = [controls_list[i] for i in indices]
    x_traj_list = [x_traj_list[i] for i in indices]

    # Get dimensions
    state_dim = len(x_traj_list[0][0])
    action_dim = len(controls_list[0][0])

    # Split data into training and validation
    split_factor = 0.8
    train_size = int(len(controls_list) * split_factor)
    
    train_controls_list = controls_list[:train_size]
    train_x_traj_list = x_traj_list[:train_size]
    val_controls_list = controls_list[train_size:]
    val_x_traj_list = x_traj_list[train_size:]

    # Create datasets
    train_dataset = BCDataset(x_traj_list=train_x_traj_list, controls_list=train_controls_list)
    val_dataset = BCDataset(x_traj_list=val_x_traj_list, controls_list=val_controls_list)

    # Get DataLoader settings
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader_kwargs = get_dataloader_kwargs(device, Config)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, 
        shuffle=True,
        **loader_kwargs
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, 
        shuffle=False,
        **loader_kwargs
    )

    # Initialize diffusion policy and optimizer with weight decay
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    diffusion = DiffusionPolicy(
        state_dim=state_dim,
        action_dim=action_dim,
        device=device,
        n_steps=Config.DIFFUSION_N_STEPS,
    )
    
    optimizer = torch.optim.Adam(diffusion.score_model.parameters(), 
                               lr=Config.LR, 
                               weight_decay=Config.WEIGHT_DECAY)
    
    # Set up tensorboard
    writer = SummaryWriter(Config.get_tensorboard_path(num_dems, random_seed) + "_diffusion")

    # Initialize environment for evaluation
    env, meta_env = load_env(Config)
    env.seed(random_seed)
    
    # Training loop with diffusion-specific epoch count
    for epoch in range(Config.DIFFUSION_EPOCH):
        diffusion.score_model.train()
        total_train_loss = 0
        
        train_bar = tqdm(train_dataloader, position=0, leave=True)
        for batch_idx, (states, actions_next_states) in enumerate(train_bar):
            states = states.to(device)
            actions_next_states = actions_next_states.to(device)
            
            loss = diffusion.train_step(actions_next_states, states, optimizer)
            total_train_loss += loss
            
            train_bar.set_description(
                f"Epoch {epoch}, Train Loss: {total_train_loss / (batch_idx + 1):.4f}"
            )
            
            writer.add_scalar('Training/Loss', loss, 
                            epoch * len(train_dataloader) + batch_idx)

        avg_train_loss = total_train_loss / len(train_dataloader)
        print(f"Epoch {epoch}, Average Train Loss: {avg_train_loss:.4f}")

        # Validation loop
        diffusion.score_model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for states, actions_next_states in val_dataloader:
                states = states.to(device)
                actions_next_states = actions_next_states.to(device)
                
                # Sample noise and add to data
                x_t, eps = diffusion.q_sample(actions_next_states, torch.randint(0, diffusion.n_steps, (actions_next_states.size(0),), device=device))
                
                # Predict noise
                eps_pred = diffusion.score_model(x_t, states, torch.randint(0, diffusion.n_steps, (actions_next_states.size(0),), device=device))
                
                # Loss is MSE between true and predicted noise
                val_loss = nn.MSELoss()(eps_pred, eps)
                total_val_loss += val_loss.item()

        avg_val_loss = total_val_loss / len(val_dataloader)
        writer.add_scalar('Validation/Loss', avg_val_loss, epoch)
        print(f"Epoch {epoch}, Average Validation Loss: {avg_val_loss:.4f}")

        # Periodic evaluation
        if epoch % (Config.EVAL_INTERVAL_FACTOR * Config.DIFFUSION_EPOCH) == 0:
            diffusion_agent = DiffusionPolicyAgent(
                diffusion, device, action_dim, stats_path=Config.get_model_path(num_dems, random_seed)
            )
            results = evaluate_model(diffusion_agent, env, meta_env, device)
            
            # Log evaluation results
            for noise in results:
                writer.add_scalar(f'Eval/Mean_Reward_{noise}', results[noise]['mean_reward'], epoch)
                writer.add_scalar(f'Eval/Success_Rate_{noise}', results[noise]['success_rate'], epoch)

    # Save model
    save_path = Config.get_model_path(num_dems, random_seed)
    os.makedirs(save_path, exist_ok=True)
    torch.save(diffusion.score_model.state_dict(), os.path.join(save_path, "diffusion_model.pt"))
    
    writer.close()
    return diffusion 

# if __name__ == '__main__':
#     # Set start method to spawn
#     mp.set_start_method('spawn', force=True)