import torch
import torch.nn as nn
import numpy as np

class MLP(nn.Module):
    def __init__(
        self,
        input_dim=6,
        output_dim=3,
        hidden_dim=256,
        num_hidden_layers=3,
    ):
        super(MLP, self).__init__()

        # Create list of hidden layers
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            
        self.hidden_layers = nn.ModuleList(layers)
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, state):
        hidden_outputs = []
        x = state
        
        # Pass through hidden layers and collect outputs
        for layer in self.hidden_layers:
            x = torch.relu(layer(x))
            hidden_outputs.append(x)
            
        # Skip connections - sum all hidden layer outputs
        x = sum(hidden_outputs)
        x = torch.tanh(x)
        x = self.output_layer(x)
        return x

    def get_action(self, state_tensor, device):
        if type(state_tensor) == np.ndarray:
            state_tensor = torch.tensor(state_tensor, dtype=torch.float32)
        self.eval()
        with torch.no_grad():
            state_tensor = state_tensor.to(device)
            action = self.forward(state_tensor)
        return action.squeeze().detach()

class DenoisingMLP(nn.Module):
    """
    Network architecture specifically for state-only BC denoising with inductive bias:
    - Input: [x_t, noisy_x_t+1]
    - Output: [action, clean_x_t+1]
    - Features:
        1. Residual connection between noisy and clean x_t+1
        2. Context-aware architecture for x_t
        3. Explicit denoising structure
    """
    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_dim=256,
        num_hidden_layers=3,
        context_dropout=0.1
    ):
        super(DenoisingMLP, self).__init__()
        
        # Context (x_t) processing
        self.context_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(context_dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(context_dropout)
        )
        
        # Denoising network
        denoising_layers = []
        # First layer combines context and noisy state
        denoising_layers.append(nn.Linear(hidden_dim + state_dim, hidden_dim))
        denoising_layers.append(nn.ReLU())
        
        # Hidden layers
        for _ in range(num_hidden_layers - 1):
            denoising_layers.append(nn.Linear(hidden_dim, hidden_dim))
            denoising_layers.append(nn.ReLU())
        
        # Split into two heads:
        # 1. Action prediction
        # 2. State noise prediction (to be subtracted from noisy state)
        self.denoising_net = nn.Sequential(*denoising_layers)
        self.action_head = nn.Linear(hidden_dim, action_dim)
        self.noise_head = nn.Linear(hidden_dim, state_dim)
        
        self.state_dim = state_dim
        self.action_dim = action_dim

    def forward(self, x):
        """
        Input x: [x_t, noisy_x_t+1]
        Output: [action, clean_x_t+1]
        """
        # Split input into current state and noisy next state
        x_t = x[:, :self.state_dim]
        noisy_x_next = x[:, self.state_dim:]
        
        # Process context (x_t)
        context = self.context_net(x_t)
        
        # Combine context with noisy state
        combined = torch.cat([context, noisy_x_next], dim=-1)
        
        # Get shared features
        features = self.denoising_net(combined)
        
        # Predict action
        action = self.action_head(features)
        
        # Predict noise to be subtracted from noisy state
        predicted_noise = self.noise_head(features)
        
        # Residual connection: clean_state = noisy_state - predicted_noise
        clean_x_next = noisy_x_next - predicted_noise
        
        # Combine action and denoised state
        output = torch.cat([action, clean_x_next], dim=-1)
        
        return output
