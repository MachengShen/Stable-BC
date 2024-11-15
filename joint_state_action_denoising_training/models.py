import torch
import torch.nn as nn
import numpy as np


A_G = 9.81
roll_max = 0.4  # radians
pitch_max = 0.4
f_g_diff_max = 1.0  # max difference between thrust and gravity


class MyModel(nn.Module):
    def __init__(
        self,
        input_dim=6,
        output_dim=3,
        hidden_dim=256,
        is_denoising_net=False,
        joint_action_state=False,
        action_loss_weight=0.75,
        state_loss_weight=0.25,
    ):
        super(MyModel, self).__init__()

        # multi-layer perceptron
        self.pi_1 = nn.Linear(input_dim, hidden_dim)
        self.pi_2 = nn.Linear(hidden_dim, hidden_dim)
        self.pi_3 = nn.Linear(hidden_dim, hidden_dim)
        self.pi_4 = nn.Linear(hidden_dim, output_dim)

        self.mse_loss = nn.MSELoss(reduction="none")
        self.is_denoising_net = is_denoising_net
        self.joint_action_state = joint_action_state
        self.action_loss_weight = action_loss_weight
        self.state_loss_weight = state_loss_weight

        if self.is_denoising_net:
            # may need to check if the shift and normalizer are correct for action dim
            self.shift = torch.tensor(
                [2.5, 2.5, 2.5, 0, 0, 0, A_G, 0, 0, 2.5, 2.5, 2.5, 0, 0, 0]
            )
            self.normalizer = torch.tensor(
                [2.5, 2.5, 2.5, 4, 3, 1, 1, 1, 1, 2.5, 2.5, 2.5, 4, 3, 1]
            )
        else:
            self.shift = torch.tensor([2.5, 2.5, 2.5, 0, 0, 0])
            self.normalizer = torch.tensor([2.5, 2.5, 2.5, 4, 3, 1])

    def loss_func(self, y_true, y_pred):
        if self.joint_action_state:
            action_loss = self.mse_loss(y_true[:, :3], y_pred[:, :3]).mean()
            state_loss = self.mse_loss(y_true[:, 3:], y_pred[:, 3:]).mean()
            return (
                self.action_loss_weight * action_loss
                + self.state_loss_weight * state_loss
            )
        else:
            return self.mse_loss(y_true, y_pred).mean()

    # policy
    def forward(self, state):
        state = state - self.shift.to(state.device)
        state = state / self.normalizer.to(state.device)
        x1 = torch.relu(self.pi_1(state))
        x2 = torch.relu(self.pi_2(x1))
        x3 = torch.relu(self.pi_3(x2))
        x = x1 + x2 + x3  # Skip connection
        x = torch.tanh(x)
        x = self.pi_4(x)
        if len(x.shape) == 2:
            x[:, :3] = x[:, :3] * torch.tensor([f_g_diff_max, roll_max, pitch_max]).to(
                state.device
            ) + torch.tensor([A_G, 0, 0]).to(state.device)
            if x.shape[-1] > 3:
                x[:, 3:] = x[:, 3:] * torch.tensor([2.5, 2.5, 2.5, 4, 3, 1]).to(
                    state.device
                ) + torch.tensor([2.5, 2.5, 2.5, 0, 0, 0]).to(state.device)
        else:
            assert len(x.shape) == 1
            x[:3] = x[:3] * torch.tensor([f_g_diff_max, roll_max, pitch_max]).to(
                state.device
            ) + torch.tensor([A_G, 0, 0]).to(state.device)
            if x.shape[-1] > 3:
                x[3:] = x[3:] * torch.tensor([2.5, 2.5, 2.5, 4, 3, 1]).to(
                    state.device
                ) + torch.tensor([2.5, 2.5, 2.5, 0, 0, 0]).to(state.device)
        return x

    def get_action(self, state_tensor, device):
        if type(state_tensor) == np.ndarray:
            state_tensor = torch.tensor(state_tensor, dtype=torch.float32)
        self.eval()
        with torch.no_grad():
            # state_tensor = torch.cat( [ (state_tensor[:2] - torch.tensor([5.0, 5.0]) )/ 5, torch.sin(state_tensor[2]).reshape(1), torch.cos(state_tensor[2]).reshape(1) ] )
            # state_tensor = state_tensor.reshape((1, 4))
            # state_tensor = state_tensor.to(device)

            # for the state
            # divide by the some value

            # for the control
            # multiply by the control bound
            # add A_G form the output 0
            state_tensor = state_tensor.to(device)
            action = self.forward(state_tensor)
        return action.squeeze().detach()
