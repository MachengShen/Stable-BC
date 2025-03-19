import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import math

def get_timestep_embedding(timesteps, embedding_dim, max_positions=10000):
    """
    Create sinusoidal timestep embeddings following Transformer architecture
    """
    half_dim = embedding_dim // 2
    emb = math.log(max_positions) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
    emb = timesteps[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
    if embedding_dim % 2 == 1:  # zero pad if needed
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb

class ResidualBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        
    def forward(self, x):
        return x + self.net(x)

class DiffusionScoreNetwork(nn.Module):
    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_dim=256,
        num_hidden_layers=3,
        time_embed_dim=128,
        condition_dropout=0.0
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.output_dim = action_dim + state_dim  # predict joint [action, next_state]
        self.time_embed_dim = time_embed_dim  # Store time_embed_dim as class attribute
        
        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(time_embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # State embedding with condition dropout
        self.state_embed = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(condition_dropout)
        )
        
        # Main network
        self.input_proj = nn.Linear(self.output_dim, hidden_dim)
        
        # Residual blocks
        self.res_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, hidden_dim)
            for _ in range(num_hidden_layers)
        ])
        
        # Cross attention for condition
        self.condition_cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            batch_first=True
        )
        
        self.final_layers = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.output_dim)
        )

    def forward(self, x, state, t):
        """
        x: noisy [action, next_state] (B, action_dim + state_dim)
        state: current state (B, state_dim)
        t: diffusion time step (B,)
        """
        # Get time embedding
        t = t.float()
        t_emb = get_timestep_embedding(t, self.time_embed_dim)  # Use stored time_embed_dim
        t_emb = self.time_embed(t_emb)  # (B, hidden_dim)
        
        # Get state embedding
        state_emb = self.state_embed(state)  # (B, hidden_dim)
        
        # Project input
        h = self.input_proj(x)  # (B, hidden_dim)
        
        # Apply residual blocks
        for res_block in self.res_blocks:
            h = res_block(h)
            
        # Cross attention with condition
        h_cond, _ = self.condition_cross_attention(
            query=h.unsqueeze(1),
            key=state_emb.unsqueeze(1),
            value=state_emb.unsqueeze(1)
        )
        h_cond = h_cond.squeeze(1)
        
        # Combine features
        h = torch.cat([h, h_cond], dim=-1)
        
        # Final layers
        score = self.final_layers(h)
        
        return score

class DiffusionPolicy:
    def __init__(
        self,
        state_dim,
        action_dim,
        device,
        n_steps=50,
        beta_schedule='linear'
    ):
        self.device = device
        self.n_steps = n_steps
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Set up noise schedule
        if beta_schedule == 'linear':
            self.beta = torch.linspace(0.0001, 0.02, n_steps).to(device)
        elif beta_schedule == 'cosine':
            # Cosine schedule as proposed in https://arxiv.org/abs/2102.09672
            steps = n_steps + 1
            x = torch.linspace(0, n_steps, steps)
            alphas_cumprod = torch.cos(((x / n_steps) + 0.008) / 1.008 * math.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            self.beta = torch.clip(betas, 0.0001, 0.02).to(device)
        
        self.alpha = 1 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
        
        # Ensure numerical stability
        self.alpha_bar = torch.clamp(self.alpha_bar, min=1e-5)
        
        self.score_model = DiffusionScoreNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
        ).to(device)
        
    def q_sample(self, x_0, t, eps=None):
        """Sample from forward diffusion process with reparameterization"""
        if eps is None:
            eps = torch.randn_like(x_0)
            
        alpha_bar_t = self.alpha_bar[t]
        if len(alpha_bar_t.shape) == 1:
            alpha_bar_t = alpha_bar_t.view(-1, 1)
            
        mean = torch.sqrt(alpha_bar_t) * x_0
        var = 1 - alpha_bar_t
        return mean + torch.sqrt(var) * eps, eps
    
    def p_sample(self, x_t, state, t):
        """Sample from reverse diffusion process using DDIM-like sampling"""
        t = t.long()
        score = self.score_model(x_t, state, t.float())
        alpha_t = self.alpha[t]
        alpha_bar_t = self.alpha_bar[t]
        
        if len(alpha_t.shape) == 1:
            alpha_t = alpha_t.view(-1, 1)
        if len(alpha_bar_t.shape) == 1:
            alpha_bar_t = alpha_bar_t.view(-1, 1)
            
        # DDIM-like update
        sigma_t = torch.sqrt(self.beta[t])
        mean = (1 / torch.sqrt(alpha_t)) * (
            x_t - (1 - alpha_t) / torch.sqrt(1 - alpha_bar_t) * score
        )
        
        if t[0] > 0:
            noise = torch.randn_like(x_t)
            return mean + sigma_t * noise
        else:
            return mean
        
    def train_step(self, x_0, state, optimizer):
        """Single training step with improved conditioning"""
        batch_size = x_0.shape[0]
        t = torch.randint(0, self.n_steps, (batch_size,), device=self.device).long()
        
        # Sample noise and add to data
        x_t, eps = self.q_sample(x_0, t)
        
        # Predict noise with improved conditioning
        eps_pred = self.score_model(x_t, state, t.float())
        
        # Loss is MSE between true and predicted noise
        loss = nn.MSELoss()(eps_pred, eps)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        return loss.item()

