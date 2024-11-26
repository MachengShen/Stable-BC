import os
import argparse
import yaml
import optuna
import json
import numpy as np
from train_model import train_model_joint, train_baseline_bc
from eval_model import evaluate_models
from config import Config, CCIL_TASK_ENV_MAP
from utils import seedEverything
import copy
import torch.multiprocessing as mp

def get_default_envs():
    """Get list of MuJoCo and MetaWorld environments"""
    mujoco_envs = [
        'walker2d-expert-v2_20',
        'hopper-expert-v2_25',
        'ant-expert-v2_10',
        'halfcheetah-expert-v2_50'
    ]
    
    metaworld_envs = [
        'metaworld-button-press-top-down-v2',
        'metaworld-coffee-push-v2_50',
        'metaworld-coffee-pull-v2_50',
        'metaworld-drawer-close-v2'
    ]
    
    return metaworld_envs + mujoco_envs

def save_best_params(study, trial, env_name):
    """Save the best parameters to a JSON file"""
    best_params = {
        'value': trial.value,
        'params': trial.params,
    }
    
    # Create results directory if it doesn't exist
    os.makedirs('hparam_results', exist_ok=True)
    
    # Save in a human-readable format
    filename = os.path.join('hparam_results', f'best_params_{env_name}.txt')
    with open(filename, 'w') as f:
        f.write(f"Best Trial Value: {trial.value}\n")
        f.write("\nBest Parameters:\n")
        for key, value in trial.params.items():
            f.write(f"{key}: {value}\n")
    
    # Also save as JSON for programmatic access
    json_filename = os.path.join('hparam_results', f'best_params_{env_name}.json')
    with open(json_filename, 'w') as f:
        json.dump(best_params, f, indent=4)

def get_training_epochs(env_name):
    """Get appropriate number of epochs based on environment"""
    mujoco_envs = [
        'walker2d-expert-v2_20',
        'hopper-expert-v2_25',
        'ant-expert-v2_10',
        'halfcheetah-expert-v2_50'
    ]
    
    if env_name in mujoco_envs:
        return {
            'epoch': 5000,  # More epochs for MuJoCo environments
            'diffusion_epoch': 10000
        }
    else:
        return {
            'epoch': 1000,  # Default epochs for other environments
            'diffusion_epoch': 5000
        }

def objective(trial, base_config, seed=0, debug=False):
    """Optimization objective function"""
    # Create a deep copy of the Config object
    config = copy.deepcopy(base_config)
    
    # Override number of epochs based on environment
    epochs = get_training_epochs(config.CCIL_TASK_NAME)
    for key, value in epochs.items():
        setattr(config, key.upper(), value)
    
    # Training parameters
    config.LEARNING_RATE = trial.suggest_loguniform('learning_rate', 1e-4, 1e-3)
    config.BATCH_SIZE = trial.suggest_int('batch_size', 64, 256)
    config.WEIGHT_DECAY = trial.suggest_loguniform('weight_decay', 1e-5, 1e-3)
    
    # Loss weights and noise parameters
    config.ACTION_LOSS_WEIGHT_BC = trial.suggest_float('action_loss_weight_bc', 0.3, 0.9)
    config.ACTION_LOSS_WEIGHT_DENOISING = trial.suggest_float('action_loss_weight_denoising', 0.05, 0.9)
    config.ACTION_NOISE_MULTIPLIER = trial.suggest_loguniform('action_noise_multiplier', 0.001, 0.1)
    config.STATE_NOISE_MULTIPLIER = trial.suggest_loguniform('state_noise_multiplier', 0.001, 0.1)
    
    if debug:
        # Run without try-except for debugging
        seedEverything(seed)
        
        # Train model
        bc_model, denoising_model, mean_rewards = train_model_joint(config.NUM_DEMS, seed, config, save_ckpt=False)
        
        # Use the median of evaluation rewards as objective
        if mean_rewards and len(mean_rewards['denoising_joint_bc']) > 0:
            return np.median(mean_rewards['denoising_joint_bc'])
        else:
            return float('-inf')
    else:
        try:
            seedEverything(seed)
            
            # Train model
            bc_model, denoising_model, mean_rewards = train_model_joint(config.NUM_DEMS, seed, config, save_ckpt=False)
            
            # Use the median of evaluation rewards as objective
            if mean_rewards and len(mean_rewards['denoising_joint_bc']) > 0:
                return np.median(mean_rewards['denoising_joint_bc'])
            else:
                return float('-inf')
        
        except Exception as e:
            print(f"Trial failed with error: {str(e)}")
            return float('-inf')

def optimize_env(env_name, base_config, n_trials, debug=False):
    """Run optimization for a single environment"""
    print(f"\n{'='*80}")
    print(f"Optimizing for environment: {env_name}")
    print(f"{'='*80}")
    
    # Update config for this environment
    # Create temporary config file with updated task name
    tmp_config = base_config.copy()
    tmp_config['ccil_task_name'] = env_name
    
    # Save temporary config to file
    os.makedirs('tmp', exist_ok=True)
    tmp_config_path = os.path.join('tmp', 'tmp_config.yaml')
    with open(tmp_config_path, 'w') as f:
        yaml.dump(tmp_config, f)
    
    # Load config using Config class method
    Config.load_config_for_training(tmp_config_path)
    config = {k.lower(): v for k, v in vars(Config).items() if not k.startswith('__')}
    
    # Clean up temporary file
    os.remove(tmp_config_path)
    
    # Create study for optimization
    study = optuna.create_study(direction="maximize")
    
    if debug:
        # Run without try-except for debugging
        study.optimize(
            lambda trial: objective(trial, Config, debug=debug),
            n_trials=n_trials,
            callbacks=[
                lambda study, trial: save_best_params(study, trial, env_name)
            ],
        )
        
        print(f"\nOptimization completed for {env_name}!")
        print("\nBest trial:")
        trial = study.best_trial
        print("  Value: ", trial.value)
        print("  Params: ")
        for key, value in trial.params.items():
            print(f"    {key}: {value}")
    else:
        # Run with try-except for production
        try:
            study.optimize(
                lambda trial: objective(trial, Config, debug=debug),
                n_trials=n_trials,
                callbacks=[
                    lambda study, trial: save_best_params(study, trial, env_name)
                ],
            )
            
            print(f"\nOptimization completed for {env_name}!")
            print("\nBest trial:")
            trial = study.best_trial
            print("  Value: ", trial.value)
            print("  Params: ")
            for key, value in trial.params.items():
                print(f"    {key}: {value}")
                
        except KeyboardInterrupt:
            print(f"\nOptimization stopped early for {env_name}.")
            if study.best_trial:
                save_best_params(study, study.best_trial, env_name)
                print("\nBest parameters so far have been saved.")
    
    return study.best_trial if study.best_trial else None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Base config file')
    parser.add_argument('--n_trials', type=int, default=20, help='Number of optimization trials per environment')
    parser.add_argument('--envs', nargs='+', default=None, 
                       help='Specific environments to optimize. If not provided, will use all MuJoCo and MetaWorld envs')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode without try-except')
    args = parser.parse_args()
    
    # Load base config
    with open(args.config, 'r') as f:
        base_config = yaml.safe_load(f)
    
    # Get environments to optimize
    envs_to_optimize = args.envs if args.envs is not None else get_default_envs()
    
    # Run optimization for each environment
    results = {}
    for env_name in envs_to_optimize:
        best_trial = optimize_env(env_name, base_config, args.n_trials, debug=args.debug)
        if best_trial:
            results[env_name] = {
                'value': best_trial.value,
                'params': best_trial.params
            }
    
    # Save overall results
    with open(os.path.join('hparam_results', 'all_results.json'), 'w') as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    # Set start method to spawn
    mp.set_start_method('spawn', force=True)
    main() 