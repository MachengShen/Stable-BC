import os
import pickle
import torch

class JointStateActionAgent:
    def __init__(self, bc_model, denoising_model, device, action_dim, stats_path, state_only_bc=False):
        self.bc_model = bc_model
        self.denoising_model = denoising_model
        self.device = device
        self.action_dim = action_dim
        self.state_only_bc = state_only_bc
        
        # Load normalization statistics
        with open(os.path.join(stats_path, 'normalization_stats.pkl'), 'rb') as f:
            stats = pickle.load(f)
            self.state_mean = torch.FloatTensor(stats['x_traj_mean']).to(device)
            self.state_std = torch.FloatTensor(stats['x_traj_std']).to(device)
            self.action_mean = torch.FloatTensor(stats['controls_mean']).to(device)
            self.action_std = torch.FloatTensor(stats['controls_std']).to(device)

    def normalize_state(self, state):
        return (state - self.state_mean) / (self.state_std + 1e-8)

    def denormalize_action(self, action):
        return action * self.action_std + self.action_mean

    def predict(self, state):
        """Predict action given state"""
        with torch.no_grad():
            # Normalize state
            state_tensor = torch.FloatTensor(state).to(self.device)
            if len(state_tensor.shape) == 1:
                state_tensor = state_tensor.unsqueeze(0)
            state_tensor = self.normalize_state(state_tensor)
            
            # Get BC prediction
            bc_output = self.bc_model(state_tensor)
            
            if self.state_only_bc:
                # BC model only predicts next state
                next_state_pred = bc_output
                
                if self.denoising_model is not None:
                    # Use denoising model to get [action, next_state]
                    denoising_input = torch.cat([state_tensor, next_state_pred], dim=-1)
                    denoising_output = self.denoising_model(denoising_input)
                    action = denoising_output[:, :self.action_dim]
                else:
                    # Without denoising model, we can't get action
                    raise ValueError("Cannot predict action with state_only_bc=True without denoising model")
            else:
                # BC model predicts [action, next_state]
                action = bc_output[:, :self.action_dim]
                
                if self.denoising_model is not None:
                    # Use denoising model to refine prediction
                    denoising_input = torch.cat([state_tensor, bc_output], dim=-1)
                    denoising_output = self.denoising_model(denoising_input)
                    action = denoising_output[:, :self.action_dim]
            
            # Denormalize action
            action = self.denormalize_action(action)
            
        if action is None:
            raise ValueError("Action prediction is None")
        if torch.isnan(action).any() or torch.isinf(action).any():
            raise ValueError("Action prediction contains NaN or Inf values")
        
        return action.cpu().numpy()[0]

class BaselineBCAgent:
    def __init__(self, bc_model, device, action_dim, stats_path):
        self.bc_model = bc_model
        self.device = device
        self.action_dim = action_dim
        
        # Load normalization statistics
        with open(os.path.join(stats_path, 'normalization_stats.pkl'), 'rb') as f:
            stats = pickle.load(f)
            self.state_mean = torch.FloatTensor(stats['x_traj_mean']).to(device)
            self.state_std = torch.FloatTensor(stats['x_traj_std']).to(device)
            self.action_mean = torch.FloatTensor(stats['controls_mean']).to(device)
            self.action_std = torch.FloatTensor(stats['controls_std']).to(device)

    def normalize_state(self, state):
        return (state - self.state_mean) / (self.state_std + 1e-8)

    def denormalize_action(self, action):
        return action * self.action_std + self.action_mean

    def predict(self, state):
        state_tensor = torch.FloatTensor(state).to(self.device)
        if len(state_tensor.shape) == 1:
            state_tensor = state_tensor.unsqueeze(0)

        # Normalize state
        state_tensor = self.normalize_state(state_tensor)

        with torch.no_grad():
            # Get initial prediction from BC model
            bc_output = self.bc_model(state_tensor)
            action = bc_output[:, :self.action_dim]
            next_state_pred = bc_output[:, self.action_dim:]
            
            # Denormalize action before returning
            action = self.denormalize_action(action)
            
        if action is None:
            raise ValueError("Action prediction is None")
        if torch.isnan(action).any() or torch.isinf(action).any():
            raise ValueError("Action prediction contains NaN or Inf values")
        
        return action.cpu().numpy()[0]

class DiffusionPolicyAgent:
    def __init__(self, diffusion_model, device, action_dim, stats_path):
        self.diffusion = diffusion_model
        self.device = device
        self.action_dim = action_dim
        
        # Load normalization statistics
        with open(os.path.join(stats_path, 'normalization_stats.pkl'), 'rb') as f:
            stats = pickle.load(f)
            self.state_mean = torch.FloatTensor(stats['x_traj_mean']).to(device)
            self.state_std = torch.FloatTensor(stats['x_traj_std']).to(device)
            self.action_mean = torch.FloatTensor(stats['controls_mean']).to(device)
            self.action_std = torch.FloatTensor(stats['controls_std']).to(device)

    def normalize_state(self, state):
        return (state - self.state_mean) / (self.state_std + 1e-8)

    def denormalize_action(self, action):
        return action * self.action_std + self.action_mean

    def predict(self, state):
        state_tensor = torch.FloatTensor(state).to(self.device)
        if len(state_tensor.shape) == 1:
            state_tensor = state_tensor.unsqueeze(0)

        # Normalize state
        state_tensor = self.normalize_state(state_tensor)

        with torch.no_grad():
            # Sample from diffusion model
            x_T = torch.randn(state_tensor.shape[0], self.action_dim + state_tensor.shape[1]).to(self.device)
            
            # Reverse diffusion process
            x_t = x_T
            for t in reversed(range(self.diffusion.n_steps)):
                t_batch = torch.ones(state_tensor.shape[0], device=self.device).long() * t
                x_t = self.diffusion.p_sample(x_t, state_tensor, t_batch)
            
            # Extract action from joint prediction
            action = x_t[:, :self.action_dim]
            
            # Denormalize action
            action = self.denormalize_action(action)

        return action.cpu().numpy()[0]

class RandomAgent:
    def __init__(self, action_space):
        self.action_space = action_space
    
    def predict(self, state):
        return self.action_space.sample()