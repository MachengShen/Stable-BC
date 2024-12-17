import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

"""
This script generates illustrative figures demonstrating the joint denoising and drift field learning approach.

The script creates two main components:

1. Data Generation:
   - Creates a sinusoidal "manifold" representing the ground truth trajectory
   - Generates noisy points around this manifold
   - Computes ground truth drift vectors along the manifold
   
2. Neural Networks:
   - Drift network f: Learns to predict drift vectors (velocity field) along the manifold
   - Denoising network d: Learns to map noisy points back to the clean manifold

The script visualizes:
   - The learned drift field compared to ground truth
   - The denoising effect of mapping noisy points back to the manifold
   - A zoomed-in comparison showing how the combined (f + d) approach improves trajectory following

The figures help illustrate how joint training of drift and denoising networks can improve
robustness to noise in behavioral cloning settings.
"""

###############################################################################
# 1) Data generation: Sinusoidal "manifold" + noisy points
###############################################################################
def sinusoid_curve(x):
    """Returns y = sin(x)."""
    return np.sin(x)

def ground_truth_drift(x, y):
    """Toy drift field: f(x,y) = (1, cos(x)), ignoring y offset."""
    fx = np.ones_like(x)
    fy = np.cos(x)
    return fx, fy

def create_training_data(n_samples=2000, noise_scale=0.3):
    """
    Data for:
      - Drift network f: points on manifold -> drift vectors
      - Denoising network d: noisy points near manifold -> clean manifold points
    """
    # Generate points only in one period [-pi, pi]
    x_manifold = np.random.uniform(-np.pi, np.pi, size=(n_samples,))
    y_manifold = sinusoid_curve(x_manifold)
    fx, fy = ground_truth_drift(x_manifold, y_manifold)

    X_f = np.stack([x_manifold, y_manifold], axis=1)
    Y_f = np.stack([fx, fy], axis=1)

    x_clean = np.random.uniform(-np.pi, np.pi, size=(n_samples,))
    y_clean = sinusoid_curve(x_clean)
    x_noisy = x_clean + np.random.normal(scale=noise_scale, size=(n_samples,))
    y_noisy = y_clean + np.random.normal(scale=noise_scale, size=(n_samples,))
    X_d = np.stack([x_noisy, y_noisy], axis=1)
    Y_d = np.stack([x_clean, y_clean], axis=1)

    return X_f, Y_f, X_d, Y_d

###############################################################################
# 2) Simple neural nets for f and d
###############################################################################
class SimpleMLP(nn.Module):
    def __init__(self, hidden_size=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 2)
        )
    def forward(self, x):
        return self.net(x)

def train_network(net, X, Y, epochs=2000, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device)
    X_t = torch.tensor(X, dtype=torch.float32, device=device)
    Y_t = torch.tensor(Y, dtype=torch.float32, device=device)
    optimizer = optim.Adam(net.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        optimizer.zero_grad()
        pred = net(X_t)
        loss = loss_fn(pred, Y_t)
        loss.backward()
        optimizer.step()

        if (epoch+1) % 500 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss = {loss.item():.6f}")

    net.cpu()
    return net

###############################################################################
# 3) Euler integrator for quick trajectory simulation
###############################################################################
def euler_integration(x0, y0, field_func, t_steps=100, dt=0.1, noise_scale=0.0):
    """
    Euler integration with optional noise in the field function.
    
    Args:
        x0, y0: Initial state
        field_func: Function that returns (dx/dt, dy/dt)
        t_steps: Number of integration steps
        dt: Time step size
        noise_scale: Standard deviation of Gaussian noise added to field predictions
    """
    xs = [x0]
    ys = [y0]
    x, y = x0, y0
    for _ in range(t_steps):
        fx, fy = field_func(x, y)
        if noise_scale > 0:
            fx += np.random.normal(0, noise_scale)
            fy += np.random.normal(0, noise_scale)
        x = x + fx * dt
        y = y + fy * dt
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

###############################################################################
# 4) Compute f, d, f+d and divergences on a grid
###############################################################################
def compute_field_and_divergence(net_f, net_d, grid_points):
    """
    grid_points: (N,2) torch tensor with requires_grad=True
    Returns:
      f_np, d_np, fd_np: shape (N,2)
      div_f, div_d, div_fd: shape (N,)
    """
    # Single forward pass for the entire batch
    f_out = net_f(grid_points)              # shape (N,2)
    clean_pts = net_d(grid_points)          # shape (N,2)
    d_out = clean_pts - grid_points         # shape (N,2)
    fd_out = f_out + d_out                  # shape (N,2)

    N = len(grid_points)
    div_f  = np.zeros(N, dtype=np.float32)
    div_d  = np.zeros(N, dtype=np.float32)
    div_fd = np.zeros(N, dtype=np.float32)

    # Row-by-row partial derivatives
    for i in range(N):
        pt = grid_points[i].unsqueeze(0).requires_grad_(True)

        # Re-run forward pass for each row
        f_val = net_f(pt)      # shape (1,2)
        fx_i, fy_i = f_val[0,0], f_val[0,1]
        clean_pt = net_d(pt)
        dx_i = clean_pt[0,0] - pt[0,0]
        dy_i = clean_pt[0,1] - pt[0,1]
        fx_fd = fx_i + dx_i
        fy_fd = fy_i + dy_i

        # Divergence of f
        grad_fx_x_f = torch.autograd.grad(fx_i, pt, retain_graph=True)[0][0,0]
        grad_fy_y_f = torch.autograd.grad(fy_i, pt, retain_graph=True)[0][0,1]
        div_f[i] = (grad_fx_x_f + grad_fy_y_f).item()

        # Divergence of d
        grad_dx_x_d = torch.autograd.grad(dx_i, pt, retain_graph=True)[0][0,0]
        grad_dy_y_d = torch.autograd.grad(dy_i, pt, retain_graph=True)[0][0,1]
        div_d[i] = (grad_dx_x_d + grad_dy_y_d).item()

        # Divergence of f + d
        grad_fx_x_fd = torch.autograd.grad(fx_fd, pt, retain_graph=True)[0][0,0]
        grad_fy_y_fd = torch.autograd.grad(fy_fd, pt, retain_graph=False)[0][0,1]
        div_fd[i] = (grad_fx_x_fd + grad_fy_y_fd).item()

    return (
        f_out.detach().numpy(),
        d_out.detach().numpy(),
        fd_out.detach().numpy(),
        div_f, div_d, div_fd
    )

###############################################################################
# Main
###############################################################################
def main():
    # Set global random seed for reproducibility
    global_seed = 49 # 49
    np.random.seed(global_seed)
    torch.manual_seed(global_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(global_seed)
        torch.cuda.manual_seed_all(global_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # 1) Prepare data & train
    X_f, Y_f, X_d, Y_d = create_training_data(n_samples=300, noise_scale=0.3)
    net_f = SimpleMLP(hidden_size=64)
    net_d = SimpleMLP(hidden_size=64)

    print("Training drift network f...")
    net_f = train_network(net_f, X_f, Y_f, epochs=500, lr=1e-3)
    print("Finished training drift.\n")

    print("Training denoising network d...")
    net_d = train_network(net_d, X_d, Y_d, epochs=2000, lr=1e-3)
    print("Finished training denoising.\n")

    # Set noise scale for rollout
    rollout_noise_scale = 0.0
    initial_point_noise = 0.2  # Noise for initial point

    # Add noise to initial point
    np.random.seed(global_seed)  # Reset seed before generating initial point
    x0 = -np.pi + np.random.normal(0, initial_point_noise)
    y0 = 0.0 + np.random.normal(0, initial_point_noise)

    # Python-callable fields
    def learned_f(x, y):
        inp = torch.tensor([[x,y]], dtype=torch.float32)
        with torch.no_grad():
            out = net_f(inp).numpy()[0]
        return out[0], out[1]

    def learned_denoise_vector(x, y):
        inp = torch.tensor([[x,y]], dtype=torch.float32)
        with torch.no_grad():
            clean_pt = net_d(inp).numpy()[0]
        return clean_pt[0] - x, clean_pt[1] - y

    def learned_f_plus_d(x, y):
        fx, fy = learned_f(x, y)
        dx, dy = learned_denoise_vector(x, y)
        return fx + dx, fy + dy

    # Generate trajectories from same noisy initial point
    np.random.seed(global_seed)  # Reset seed before trajectory generation
    print(f"Initial point: ({x0:.3f}, {y0:.3f})")  # Print initial point for verification
    traj_f_x, traj_f_y = euler_integration(x0, y0, learned_f, t_steps=100, dt=0.1, noise_scale=rollout_noise_scale)
    traj_fd_x, traj_fd_y = euler_integration(x0, y0, learned_f_plus_d, t_steps=100, dt=0.1, noise_scale=rollout_noise_scale)
    
    # Verify first points of trajectories
    print(f"First point of f trajectory: ({traj_f_x[0]:.3f}, {traj_f_y[0]:.3f})")
    print(f"First point of f+d trajectory: ({traj_fd_x[0]:.3f}, {traj_fd_y[0]:.3f})")

    # Global grid for vector fields (only one period)
    gx = np.linspace(-1.2*np.pi, 1.2*np.pi, 20)  # Wider x range
    gy = np.linspace(-2.0, 2.0, 20)              # Smaller y range
    GX, GY = np.meshgrid(gx, gy)
    grid_points = np.stack([GX.ravel(), GY.ravel()], axis=1)
    grid_points_torch = torch.tensor(grid_points, dtype=torch.float32)

    with torch.no_grad():
        f_out = net_f(grid_points_torch)              # shape (N,2)
        clean_pts = net_d(grid_points_torch)          # shape (N,2)
        d_out = clean_pts - grid_points_torch         # shape (N,2)
        fd_out = f_out + d_out                        # shape (N,2)

    f_np = f_out.numpy()
    d_np = d_out.numpy()
    fd_np = fd_out.numpy()

    Fx = f_np[:,0].reshape(GX.shape)
    Fy = f_np[:,1].reshape(GX.shape)
    Dx = d_np[:,0].reshape(GX.shape)
    Dy = d_np[:,1].reshape(GX.shape)
    Cx = fd_np[:,0].reshape(GX.shape)
    Cy = fd_np[:,1].reshape(GX.shape)

    # Expert manifold for reference (one period plus margins)
    X_curve = np.linspace(-1.2*np.pi, 1.2*np.pi, 400)  # Match grid x range
    Y_curve = sinusoid_curve(X_curve)

    # Define colors for different fields
    f_color = '#1f77b4'  # blue
    d_color = '#2ca02c'  # green
    fd_color = '#ff7f0e' # orange

    # -----------------------------
    # Single plot comparison
    # -----------------------------
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot vector fields with shorter, thinner arrows
    ax.quiver(GX, GY, Fx, Fy, color=f_color, alpha=0.3, scale=4.0,
              angles='xy', scale_units='xy', width=0.003,
              headwidth=4, headlength=4, headaxislength=3,
              label="f field")
    ax.quiver(GX, GY, Cx, Cy, color=fd_color, alpha=0.3, scale=4.0,
              angles='xy', scale_units='xy', width=0.003,
              headwidth=4, headlength=4, headaxislength=3,
              label="f+d field")
    
    # Plot manifold and trajectories with thicker lines
    ax.plot(X_curve, Y_curve, 'k-', linewidth=4, label="Expert manifold")
    ax.plot(traj_f_x, traj_f_y, color=f_color, linestyle='-', linewidth=3, label="Trajectory (f only)")
    ax.plot(traj_fd_x, traj_fd_y, color=fd_color, linestyle='-', linewidth=3, label="Trajectory (f+d)")
    ax.plot([x0], [y0], 'ko', markersize=12, label="Initial state")
    
    ax.legend(loc='upper right')
    ax.set_xlim([-1.2*np.pi, 1.2*np.pi])  # Match grid x range
    ax.set_ylim([-2.0, 2.0])              # Match grid y range
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.set_title("Trajectory Comparison: f vs f+d")
    
    plt.tight_layout()
    # Save the comparison figure
    plt.savefig(f'trajectory_comparison_noise_{rollout_noise_scale:.3f}_init_{initial_point_noise:.3f}.png', 
                dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    main()