import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from models import MyModel
import torch.nn.functional as F
import json



# import dataset for offline training
class MyData(Dataset):

    def __init__(self, cfg):
        if cfg.alg == 'ccil' or cfg.alg == 'stable_ccil':
            self.data = json.load(open("data/data_ccil.json", "r"))
        else:
            self.data = json.load(open("data/data.json", "r"))
        print("imported dataset of length:", len(self.data))

    def __len__(self):
        return len(self.data)

    def __getitem__(self,idx):
        return torch.FloatTensor(self.data[idx])

"""
We need to create a dataset for the denoising network.
The input is clean x_t, noisy a_t, noisy x_t+1, 
and the output is clean a_t and clean x_t+1.
This requires the trajectory data stored in data/data_with_trajectory.json.
We'll first train a BC model taking x_t as input and outputting a_t and x_t+1,
then we'll denoise the BC result to stabilize the trajectory.
"""

def to_tensor(array, device='cuda' if torch.cuda.is_available() else 'cpu'):
    return torch.tensor(array, dtype=torch.float32).to(device)

class DenoisingData(Dataset):
    def __init__(self, cfg, action_noise_factor=0.1, state_noise_factor=0.5):
        with open("data/data_with_trajectory.json", "r") as f:
            self.trajectories = json.load(f)
        self.action_noise_factor = action_noise_factor
        self.state_noise_factor = state_noise_factor
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def __len__(self):
        return sum(len(traj) - 1 for traj in self.trajectories)

    def __getitem__(self, idx):
        # Find the trajectory and step within the trajectory
        traj_idx = 0
        while idx >= len(self.trajectories[traj_idx]) - 1:
            idx -= len(self.trajectories[traj_idx]) - 1
            traj_idx += 1
        
        # Get clean samples
        clean_x_t = to_tensor(self.trajectories[traj_idx][idx][:4], self.device)
        clean_a_t = to_tensor(self.trajectories[traj_idx][idx][4:6], self.device)
        clean_x_t_plus_1 = to_tensor(self.trajectories[traj_idx][idx+1][:4], self.device)

        # Add noise to a_t and x_t+1 with separate noise factors
        noisy_a_t = clean_a_t + self.action_noise_factor * torch.randn_like(clean_a_t)
        noisy_x_t_plus_1 = clean_x_t_plus_1 + self.state_noise_factor * torch.randn_like(clean_x_t_plus_1)

        # Combine input and output tensors
        input_tensor = torch.cat([clean_x_t, noisy_a_t, noisy_x_t_plus_1])
        output_tensor = torch.cat([clean_a_t, clean_x_t_plus_1])

        return input_tensor, output_tensor

class BCData(Dataset):
    def __init__(self, cfg):
        with open("data/data_with_trajectory.json", "r") as f:
            self.trajectories = json.load(f)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def __len__(self):
        return sum(len(traj) - 1 for traj in self.trajectories)

    def __getitem__(self, idx):
        # Find the trajectory and step within the trajectory
        traj_idx = 0
        while idx >= len(self.trajectories[traj_idx]) - 1:
            idx -= len(self.trajectories[traj_idx]) - 1
            traj_idx += 1
        
        # Get clean samples
        x_t = to_tensor(self.trajectories[traj_idx][idx][:4], self.device)
        a_t = to_tensor(self.trajectories[traj_idx][idx][4:6], self.device)
        x_t_plus_1 = to_tensor(self.trajectories[traj_idx][idx+1][:4], self.device)

        # Combine input and output tensors
        input_tensor = x_t
        output_tensor = torch.cat([a_t, x_t_plus_1])

        return input_tensor, output_tensor


def train_model(cfg):
    alg = cfg.alg # The algorithm to be used to train policy
    savename = 'model_{}.pt'.format(cfg.alg)
    denoising_savename = 'denoising_model_{}.pt'.format(cfg.alg)
    # training parameters
    EPOCH = 4000
    LR = 0.0001

    # dataset and optimizer
    model = MyModel()
    denoising_model = MyModel(input_dim=6, output_dim=6)
    train_data = MyData(cfg)
    denoising_data = DenoisingData(cfg)
    BATCH_SIZE = int(len(train_data) / 10.)
    print("my batch size is:", BATCH_SIZE)
    train_set = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
    denoising_set = DataLoader(dataset=denoising_data, batch_size=BATCH_SIZE, shuffle=True)
    optimizer_model = torch.optim.Adam(model.parameters(), lr=LR)
    optimizer_denoising = torch.optim.Adam(denoising_model.parameters(), lr=LR)

    # helper code for getting eigenvalues
    relu = torch.nn.ReLU()

    # main training loop
    for epoch in range(EPOCH+1):
        for batch, x in enumerate(train_set):
            states = x[:, 0:4]
            actions = x[:, 4:6]
            
            # get mse loss
            loss = model.loss_func(actions, model(states))

            # get additional loss terms
            if alg == 'stable_bc' or alg == 'stable_ccil':
                # get the matrix A
                states.requires_grad = True
                a = model(states)
                J = torch.zeros((BATCH_SIZE, 2, 4))
                for i in range (2):
                    J[:, i] = torch.autograd.grad(a[:, i], states, 
                                        grad_outputs=torch.ones_like(a[:, i]), 
                                        create_graph=True)[0]
                
                # make sure top left of A is stable
                J_x = J[:,:,:2]
                # get the eigenvalues of the matrix
                E = torch.linalg.eigvals(J_x).real
                # loss is the sum of positive eigenvalues
                loss += 10.0 * torch.sum(relu(E))
                
                # penalize the magnitude of the top right of A
                J_y = J[:,:,2:]
                # get the norm of the matrix
                D = torch.linalg.matrix_norm(J_y)
                # loss is the average of the matrix magnitude
                loss += 0.1 * torch.mean(D)

            # update model parameters
            optimizer_model.zero_grad()
            loss.backward()
            optimizer_model.step()
        
        # Denoising network training
        for batch, (noisy_sample, clean_sample) in enumerate(denoising_set):
            noisy_state_action = noisy_sample
            clean_state_action = clean_sample

            # get mse loss for denoising
            denoising_loss = denoising_model.loss_func(clean_state_action, denoising_model(noisy_state_action))

            # update denoising model parameters
            optimizer_denoising.zero_grad()
            denoising_loss.backward()
            optimizer_denoising.step()

        if epoch % 500 == 0:
            print(epoch, loss.item(), denoising_loss.item())
    torch.save(model.state_dict(), "data/" + savename)
    torch.save(denoising_model.state_dict(), "data/" + denoising_savename)
    


def train_model_joint(cfg):
    # Set up the models and optimizers
    EPOCH = 4000
    LR = 0.0001
    model = MyModel(input_dim=4, output_dim=6, hidden_dim=64)
    denoising_model = MyModel(input_dim=10, output_dim=6, hidden_dim=64)
    optimizer_model = torch.optim.Adam(model.parameters(), lr=LR)
    optimizer_denoising = torch.optim.Adam(denoising_model.parameters(), lr=LR)

    # Load and prepare datasets
    bc_dataset = BCData(cfg)
    denoising_dataset = DenoisingData(cfg)
    BATCH_SIZE = int(len(bc_dataset) / 10.)
    bc_dataloader = DataLoader(bc_dataset, batch_size=BATCH_SIZE, shuffle=True)
    denoising_dataloader = DataLoader(denoising_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Training loop
    for epoch in range(EPOCH):
        for (bc_batch, denoising_batch) in zip(bc_dataloader, denoising_dataloader):
            # Unpack BC batch
            states, actions_next_states = bc_batch

            # Train behavior cloning model
            predicted_actions_next_states = model(states)
            bc_loss = F.mse_loss(predicted_actions_next_states, actions_next_states)

            # Unpack denoising batch
            noisy_actions_next_states, clean_actions_next_states = denoising_batch

            # Train denoising model
            denoised_actions_next_states = denoising_model(noisy_actions_next_states)
            denoising_loss = F.mse_loss(denoised_actions_next_states, clean_actions_next_states)

            # Combined loss
            total_loss = bc_loss + denoising_loss

            # Update both models
            optimizer_model.zero_grad()
            optimizer_denoising.zero_grad()
            total_loss.backward()
            optimizer_model.step()
            optimizer_denoising.step()

        print(f"Epoch {epoch}, BC Loss: {bc_loss.item():.4f}, Denoising Loss: {denoising_loss.item():.4f}")

    # Save the trained models
    torch.save(model.state_dict(), "data/joint_bc_model.pt")
    torch.save(denoising_model.state_dict(), "data/joint_denoising_model.pt")

    print("Joint training completed and models saved.")


