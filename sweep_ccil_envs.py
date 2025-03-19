import os
import sys
import yaml
import shutil
import argparse
import datetime
import torch.multiprocessing as mp
from config import Config, CCIL_TASK_ENV_MAP
from utils import seedEverything
from train_model import train_model_joint, train_baseline_bc, train_diffusion_policy

def get_env_config_path(env_name):
    """Get the appropriate config file path for the environment"""
    env_to_config = {
        'metaworld-button-press-top-down-v2': 'config_button_press.yaml',
        'metaworld-coffee-push-v2_50': 'config_coffee_push.yaml',
        'metaworld-coffee-pull-v2_50': 'config_coffee_pull.yaml',
        'metaworld-drawer-close-v2': 'config_drawer_close.yaml'
    }
    
    # Return environment-specific config if available, otherwise return default
    return env_to_config.get(env_name, 'config.yaml')

def train_model(config_path, seed, timestamp):
    try:
        # Load config with shared timestamp
        Config.load_config_for_training(config_path, timestamp=timestamp)
        
        # Set random seed
        seedEverything(seed)
        
        # Ensure the base log directory exists and save config
        os.makedirs(Config.BASE_LOG_PATH, exist_ok=True)
        config_copy_path = os.path.join(Config.BASE_LOG_PATH, 'config.yaml')
        shutil.copy2(config_path, config_copy_path)
        print(f"Config file copied to: {config_copy_path}")
        
        # Train all models
        eval_noise_levels = [0.0, 0.0001, 0.0003, 0.0005, 0.0007, 0.001, 0.0015, 0.002]
        print("number of epochs: ", Config.EPOCH)
        
        print("Training baseline BC model...")
        train_baseline_bc(Config.NUM_DEMS, seed, Config, eval_noise_levels=eval_noise_levels)

        print("Training baseline noisy BC model...")
        train_baseline_bc(Config.NUM_DEMS, seed, Config, train_with_noise=True, eval_noise_levels=eval_noise_levels)
        
        print("Training joint state-action model with state-only BC...")
        train_model_joint(Config.NUM_DEMS, seed, Config, state_only_bc=True, eval_noise_levels=eval_noise_levels, save_best_model_noise_level=0.0005)
        
        # print("Training joint state-action model...")
        # train_model_joint(Config.NUM_DEMS, seed, Config, state_only_bc=False, eval_noise_levels=eval_noise_levels)
        

        
        # print("Training joint state-action model with state-only BC and specialized denoising network...")
        # train_model_joint(Config.NUM_DEMS, seed, Config, state_only_bc=False, add_inductive_bias=True, eval_noise_levels=eval_noise_levels)
    
    # print("Training joint state-action model with delta state...")
    # train_model_joint(Config.NUM_DEMS, seed, Config, state_only_bc=False, predict_state_delta=True, eval_noise_levels=eval_noise_levels)
    
    # print("Training diffusion policy...")
    # train_diffusion_policy(Config.NUM_DEMS, seed, Config)
    
        return True
    except Exception as e:
        print(f"Training failed with error: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml', help='Base config file')
    parser.add_argument('--env', type=str, default=None, help='Specific environment to train on')
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2], help='Random seeds to use')
    args = parser.parse_args()
    
    # Generate timestamp for this sweep
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    
    # Get environments to train on
    if args.env:
        envs = [args.env]
    else:
        envs = list(CCIL_TASK_ENV_MAP.keys())
    
    print(f"Training on environments: {envs}")
    print(f"Using random seeds: {args.seeds}")
    
    for env_name in envs:
        print(f"\nTraining on environment: {env_name}")
        
        # Get the appropriate config file for this environment
        env_config_path = get_env_config_path(env_name)
        print(f"Using config file: {env_config_path}")
        
        # Load and update the config
        with open(env_config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Update environment name
        config['ccil_task_name'] = env_name
        
        # Save temporary config
        tmp_config_path = os.path.join('tmp', f'config_{env_name}.yaml')
        os.makedirs('tmp', exist_ok=True)
        with open(tmp_config_path, 'w') as f:
            yaml.dump(config, f)
        
        # Train with each seed
        for seed in args.seeds:
            print(f"\nTraining with seed {seed}")
            success = train_model(tmp_config_path, seed, timestamp)
            if not success:
                print(f"Training failed for environment {env_name} with seed {seed}")
        
        # Clean up temporary config
        os.remove(tmp_config_path)

if __name__ == "__main__":
    # Set start method to spawn
    mp.set_start_method('spawn', force=True)
    main() 