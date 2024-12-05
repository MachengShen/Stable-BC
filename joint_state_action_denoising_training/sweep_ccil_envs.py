import os
import yaml
import subprocess
import shutil
from config import CCIL_TASK_ENV_MAP, CCIL_NUM_DEMS_MAP, Config
import argparse
from datetime import datetime
import time
from train_model import train_model_joint, train_baseline_bc, train_diffusion_policy
from utils import seedEverything
# import torch.multiprocessing as mp

def construct_parser():
    parser = argparse.ArgumentParser(description='Sweep through CCIL environments')
    parser.add_argument('--base_config', type=str, default='config.yaml', help='Base config file path')
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2], help='Random seeds to use')
    parser.add_argument('--mode', type=str, choices=['train', 'eval', 'both'], default='both', help='Mode to run')
    parser.add_argument('--task', type=str, default=None, help='Specific task to run (optional)')
    return parser

def update_config(base_config_path, task_name, output_path):
    # Load base config
    with open(base_config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Update task-specific settings
    config['task_type'] = 'CCIL'
    config['ccil_task_name'] = task_name
    
    # Set different epoch numbers based on environment type
    mujoco_envs = [
        'walker2d-expert-v2_20',
        'hopper-expert-v2_25',
        'ant-expert-v2_10',
        'halfcheetah-expert-v2_50'
    ]
    
    # Set number of demos based on environment
    config['num_dems'] = CCIL_NUM_DEMS_MAP[task_name]
    
    # Set epochs based on environment type
    if task_name in mujoco_envs:
        config['epoch'] = 1000  # More epochs for MuJoCo environments
        config['diffusion_epoch'] = 5000  # Even more epochs for diffusion on MuJoCo
    else:
        config['epoch'] = 600  # Default epochs for other environments
        config['diffusion_epoch'] = 3000  # Default epochs for diffusion
    
    # Scale epochs based on number of demos (size of dataset)
    config['epoch'] = int(5 * config['epoch'] / config['num_dems'])
    config['diffusion_epoch'] = int(5 * config['diffusion_epoch'] / config['num_dems'])
    # Create task-specific directory
    os.makedirs(output_path, exist_ok=True)
    config_path = os.path.join(output_path, 'config.yaml')
    
    # Save updated config
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    return config_path

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
        # print("Training baseline BC model...")
        
        print("number of epochs: ", Config.EPOCH)
        train_baseline_bc(Config.NUM_DEMS, seed, Config)

        print("Training baseline noisy BC model")
        train_baseline_bc(Config.NUM_DEMS, seed, Config, train_with_noise=True)
        
        # print("Training joint state-action model...")
        # train_model_joint(Config.NUM_DEMS, seed, Config)
        
        # print("Training joint state-action model with state-only BC...")
        # train_model_joint(Config.NUM_DEMS, seed, Config, state_only_bc=True)
        
        # print("Training joint state-action model with state-only BC and specialized denoising network...")
        # train_model_joint(Config.NUM_DEMS, seed, Config, state_only_bc=True, add_inductive_bias=True)
    
    # print("Training joint state-action model with delta state...")
    # train_model_joint(Config.NUM_DEMS, seed, Config, predict_state_delta=True)
    
    # print("Training diffusion policy...")
    # train_diffusion_policy(Config.NUM_DEMS, seed, Config)
    
        return True
    except Exception as e:
        print(f"Training failed with error: {str(e)}")
        return False

def eval_model(config_path, checkpoint_dir, num_episodes=100):
    # Check if model files exist
    bc_model_path = os.path.join(checkpoint_dir, "joint_bc_model.pt")
    denoising_model_path = os.path.join(checkpoint_dir, "joint_denoising_model.pt")
    
    if not os.path.exists(bc_model_path) or not os.path.exists(denoising_model_path):
        print(f"Model files not found in {checkpoint_dir}")
        return False
    
    cmd = [
        'python', 'eval_model.py',
        '--config_path', config_path,
        '--checkpoint_dir', checkpoint_dir,
        '--num_eval_episodes', str(num_episodes)
    ]
    process = subprocess.run(cmd, capture_output=True, text=True)
    if process.returncode != 0:
        print("Evaluation failed with error:")
        print(process.stderr)
        return False
    return True

def get_checkpoint_dir(config_path, seed):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    base_log_path = config['base_log_path']
    task_name = config['ccil_task_name']
    return os.path.join(
        base_log_path,
        task_name,
        "results",
        f"joint_training/{config['num_dems']}dems/{seed}"
    )

def main():
    # # Set start method to spawn
    # mp.set_start_method('spawn', force=True)
    parser = construct_parser()
    args = parser.parse_args()

    # Create timestamp for this sweep
    sweep_timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    sweep_dir = f"sweeps/ccil_sweep_{sweep_timestamp}"
    os.makedirs(sweep_dir, exist_ok=True)

    # Save sweep configuration
    sweep_config = {
        'timestamp': sweep_timestamp,
        'seeds': args.seeds,
        'mode': args.mode,
        'base_config': args.base_config
    }
    with open(os.path.join(sweep_dir, 'sweep_config.yaml'), 'w') as f:
        yaml.dump(sweep_config, f)

    # Get tasks to process
    tasks = [args.task] if args.task else CCIL_TASK_ENV_MAP.keys()

    # Sweep through environments
    for task_name in tasks:
        # Create timestamp for this task
        task_timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        
        print(f"\n{'='*80}")
        print(f"Processing task: {task_name}")
        print(f"{'='*80}")

        # Create task-specific directory
        task_dir = os.path.join(sweep_dir, task_name)
        os.makedirs(task_dir, exist_ok=True)

        # Update config for this task
        config_path = update_config(args.base_config, task_name, task_dir)

        for seed in args.seeds:
            print(f"\nRunning with seed {seed}")
            checkpoint_dir = get_checkpoint_dir(config_path, seed)
            
            if args.mode in ['train', 'both']:
                print("Training model...")
                if not train_model(config_path, seed, task_timestamp):
                    print(f"Skipping evaluation for seed {seed} due to training failure")
                    continue
                # Wait a bit to ensure files are saved
                time.sleep(5)

            if args.mode in ['eval', 'both']:
                print("Evaluating model...")
                if not eval_model(config_path, checkpoint_dir):
                    print(f"Evaluation failed for seed {seed}")
                    continue

        # Aggregate results for this task
        print(f"\nCompleted task: {task_name}")
        
    print("\nSweep completed!")

if __name__ == "__main__":
    main() 