import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

class DiffusionScoreNetwork(nn.Module):
    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_dim=256,
        num_hidden_layers=3,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        output_dim = action_dim + state_dim  # predict joint [action, next_state]
        
        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # State embedding
        self.state_embed = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Score network (predicts noise)
        layers = []
        # Input: state embedding + time embedding + noisy [action, next_state]
        layers.append(nn.Linear(hidden_dim * 2 + output_dim, hidden_dim))
        
        for _ in range(num_hidden_layers):
            layers.append(nn.ReLU())
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            
        layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, output_dim))
        
        self.score_net = nn.Sequential(*layers)

    def forward(self, x, state, t):
        # x: noisy [action, next_state]
        # state: current state
        # t: diffusion time step
        t = t.float()  # Convert t to float
        t_embed = self.time_embed(t.unsqueeze(-1))
        state_embed = self.state_embed(state)
        
        h = torch.cat([x, state_embed, t_embed], dim=-1)
        score = self.score_net(h)
        return score

class DiffusionPolicy:
    def __init__(
        self,
        state_dim,
        action_dim,
        device,
        n_steps=50,
        beta_min=0.0001,
        beta_max=0.02,
    ):
        self.device = device
        self.n_steps = n_steps
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Linear noise schedule
        self.beta = torch.linspace(beta_min, beta_max, n_steps).to(device)
        self.alpha = 1 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
        assert torch.all(self.alpha_bar >= 0.01), "All elements of alpha_bar must be >= 0.01, otherwise maybe numerically unstable"
        
        self.score_model = DiffusionScoreNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
        ).to(device)
        
    def q_sample(self, x_0, t, eps=None):
        """Sample from forward diffusion process"""
        if eps is None:
            eps = torch.randn_like(x_0)
            
        alpha_bar_t = self.alpha_bar[t]
        if len(alpha_bar_t.shape) == 1:
            alpha_bar_t = alpha_bar_t.view(-1, 1)
            
        mean = torch.sqrt(alpha_bar_t) * x_0
        var = 1 - alpha_bar_t
        return mean + torch.sqrt(var) * eps, eps
    
    def p_sample(self, x_t, state, t):
        """Sample from reverse diffusion process"""
        t = t.long()
        score = self.score_model(x_t, state, t.float())
        alpha_t = self.alpha[t]
        alpha_bar_t = self.alpha_bar[t]
        
        if len(alpha_t.shape) == 1:
            alpha_t = alpha_t.view(-1, 1)
        if len(alpha_bar_t.shape) == 1:
            alpha_bar_t = alpha_bar_t.view(-1, 1)
            
        mean = 1/torch.sqrt(alpha_t) * (x_t - (1-alpha_t)/torch.sqrt(1-alpha_bar_t) * score)
        if t[0] > 0:
            var = (1-alpha_t)/(1-alpha_bar_t) * self.beta[t]
            eps = torch.randn_like(x_t)
            return mean + torch.sqrt(var) * eps
        else:
            return mean
        
    def train_step(self, x_0, state, optimizer):
        """Single training step"""
        batch_size = x_0.shape[0]
        t = torch.randint(0, self.n_steps, (batch_size,), device=self.device).long()
        
        # Sample noise and add to data
        x_t, eps = self.q_sample(x_0, t)
        
        # Predict noise
        eps_pred = self.score_model(x_t, state, t.float())
        
        # Loss is MSE between true and predicted noise
        loss = nn.MSELoss()(eps_pred, eps)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        return loss.item()

