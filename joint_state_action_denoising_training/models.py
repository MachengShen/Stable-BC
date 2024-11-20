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
