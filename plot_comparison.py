import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_comparison(env_name, noise_levels, direct_rewards, direct_rewards_std, direct_success, direct_success_std,
                   denoising_rewards, denoising_rewards_std, denoising_success, denoising_success_std):
    """Create comparison plots for rewards and success rates"""
    fig, ax1 = plt.subplots(figsize=(8, 6))
    
    # Plot mean rewards on the first y-axis
    color1 = '#1f77b4'  # Blue
    color2 = '#ff7f0e'  # Orange
    
    ln1 = ax1.errorbar(noise_levels, direct_rewards, yerr=direct_rewards_std, 
                      label='Joint state-action (reward)', color=color1, marker='o', capsize=5, linestyle='-')
    ln2 = ax1.errorbar(noise_levels, denoising_rewards, yerr=denoising_rewards_std, 
                      label='DeCIL (reward)', color=color2, marker='s', capsize=5, linestyle='-')
    
    ax1.set_xlabel('Noise Level')
    ax1.set_ylabel('Mean Reward')
    ax1.grid(True)
    
    # Format x-axis to remove trailing zeros
    ax1.xaxis.set_major_formatter(plt.FormatStrFormatter('%.4g'))
    
    # Create the second y-axis and plot success rates
    ax2 = ax1.twinx()
    ln3 = ax2.errorbar(noise_levels, direct_success, yerr=direct_success_std, 
                      label='Joint state-action (success)', color=color1, marker='o', capsize=5, linestyle='--')
    ln4 = ax2.errorbar(noise_levels, denoising_success, yerr=denoising_success_std, 
                      label='DeCIL (success)', color=color2, marker='s', capsize=5, linestyle='--')
    
    ax2.set_ylabel('Success Rate')
    
    # Add legends for both axes
    lines = [ln1[0], ln2[0], ln3[0], ln4[0]]  # Get the Line2D objects
    labels = ['Joint state-action (reward)', 'DeCIL (reward)', 
             'Joint state-action (success)', 'DeCIL (success)']
    ax1.legend(lines, labels, loc='upper right')
    
    plt.tight_layout()
    return plt.gcf()

def main():
    # Button Press Results
    button_noise_levels = [0.0, 0.0001, 0.0003, 0.0005, 0.0007, 0.001, 0.0015, 0.002]
    
    # Direct Mapping results for Button Press
    button_direct_rewards = [3205.19, 2070.85, 1380.83, 1219.86, 1117.22, 1036.71, 922.13, 857.51]
    button_direct_rewards_std = [381.14, 1036.69, 427.20, 342.24, 305.39, 352.90, 385.65, 377.98]
    button_direct_success = [0.95, 0.33, 0.15, 0.05, 0.03, 0.00, 0.00, 0.00]
    button_direct_success_std = [0.04, 0.47, 0.21, 0.07, 0.05, 0.00, 0.00, 0.00]
    
    # Denoising Joint BC results for Button Press
    button_denoising_rewards = [2877.01, 2889.50, 2731.81, 2593.42, 2414.10, 2114.28, 1667.96, 1713.24]
    button_denoising_rewards_std = [249.06, 381.28, 358.12, 304.86, 296.52, 501.23, 269.78, 503.18]
    button_denoising_success = [0.89, 0.80, 0.78, 0.80, 0.69, 0.58, 0.32, 0.25]
    button_denoising_success_std = [0.08, 0.13, 0.16, 0.10, 0.13, 0.29, 0.24, 0.32]
    
    # Create and save Button Press plots
    fig_button = plot_comparison(
        'Button Press',
        button_noise_levels,
        button_direct_rewards, button_direct_rewards_std,
        button_direct_success, button_direct_success_std,
        button_denoising_rewards, button_denoising_rewards_std,
        button_denoising_success, button_denoising_success_std
    )
    fig_button.savefig('button_press_comparison.png')
    plt.close()
    
    # Drawer Close Results
    drawer_noise_levels = [0.0, 0.0001, 0.0003, 0.0005, 0.0007, 0.001, 0.0015, 0.002]
    
    # Direct Mapping results for Drawer Close
    drawer_direct_rewards = [4239.57, 2661.86, 1499.14, 1404.20, 1318.11, 1350.28, 1040.64, 390.62]
    drawer_direct_rewards_std = [15.85, 1667.59, 1912.24, 1956.84, 1851.65, 1725.75, 1292.08, 542.01]
    drawer_direct_success = [1.00, 0.70, 0.37, 0.33, 0.33, 0.35, 0.30, 0.17]
    drawer_direct_success_std = [0.00, 0.42, 0.45, 0.47, 0.47, 0.46, 0.39, 0.24]
    
    # Denoising Joint BC results for Drawer Close
    drawer_denoising_rewards = [4269.04, 4250.50, 4236.82, 4191.48, 4146.94, 4011.18, 3808.96, 3457.09]
    drawer_denoising_rewards_std = [10.08, 5.99, 9.69, 53.02, 85.60, 98.95, 345.80, 330.96]
    drawer_denoising_success = [1.00, 1.00, 1.00, 0.99, 0.99, 0.95, 0.99, 0.99]
    drawer_denoising_success_std = [0.00, 0.00, 0.00, 0.02, 0.02, 0.05, 0.02, 0.02]
    
    # Create and save Drawer Close plots
    fig_drawer = plot_comparison(
        'Drawer Close',
        drawer_noise_levels,
        drawer_direct_rewards, drawer_direct_rewards_std,
        drawer_direct_success, drawer_direct_success_std,
        drawer_denoising_rewards, drawer_denoising_rewards_std,
        drawer_denoising_success, drawer_denoising_success_std
    )
    fig_drawer.savefig('drawer_close_comparison.png')
    plt.close()

if __name__ == "__main__":
    main()
    print("Plots have been saved as 'button_press_comparison.png' and 'drawer_close_comparison.png'") 