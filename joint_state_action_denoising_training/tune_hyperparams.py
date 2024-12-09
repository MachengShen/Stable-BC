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
from datetime import datetime
from misc import update_config

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
    """Save the best parameters to a JSON file with history"""
    best_params = {
        'trial_number': trial.number,
        'total_trials': len(study.trials),
        'value': trial.value,
        'params': trial.params,
        'timestamp': datetime.now().strftime("%Y%m%d-%H%M%S")
    }
    
    # Create results directory if it doesn't exist
    os.makedirs('hparam_results', exist_ok=True)
    
    # Load existing results or create new list
    json_filename = os.path.join('hparam_results', f'best_params_{env_name}.json')
    if os.path.exists(json_filename):
        with open(json_filename, 'r') as f:
            history = json.load(f)
            if not isinstance(history, list):
                history = [history]  # Convert old format to list
    else:
        history = []
    
    # Append new best parameters
    history.append(best_params)
    
    # Sort history by value (descending) and add rank
    history.sort(key=lambda x: x['value'], reverse=True)
    for i, result in enumerate(history):
        result['rank'] = i + 1
    
    # Save updated history
    with open(json_filename, 'w') as f:
        json.dump(history, f, indent=4)

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
            'epoch': 1000,  # More epochs for MuJoCo environments
            'diffusion_epoch': 5000
        }
    else:
        return {
            'epoch': 1500,  # Default epochs for other environments
            'diffusion_epoch': 3000
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
    # config.LEARNING_RATE = trial.suggest_loguniform('learning_rate', 1e-4, 1e-3)
    # # config.BATCH_SIZE = trial.suggest_int('batch_size', 64, 256)
    # config.WEIGHT_DECAY = trial.suggest_loguniform('weight_decay', 1e-5, 1e-3)
    
    # Loss weights and noise parameters
    # config.ACTION_LOSS_WEIGHT_BC = trial.suggest_float('action_loss_weight_bc', 0.3, 0.9)
    config.ACTION_LOSS_WEIGHT_DENOISING = trial.suggest_float('action_loss_weight_denoising', 0.05, 0.9)
    # config.ACTION_NOISE_MULTIPLIER = trial.suggest_loguniform('action_noise_multiplier', 0.0001, 0.1)
    config.STATE_NOISE_MULTIPLIER = trial.suggest_loguniform('state_noise_multiplier', 0.0001, 0.03)
    
    if debug:
        # Run without try-except for debugging
        seedEverything(seed)
        
        # Train model
        bc_model, denoising_model, mean_rewards = train_model_joint(config.NUM_DEMS, seed, config, save_ckpt=False, state_only_bc=True)
        
        # Use the median of evaluation rewards as objective
        if mean_rewards and len(mean_rewards['denoising_joint_bc']) > 0:
            return np.median(mean_rewards['denoising_joint_bc'])
        else:
            return float('-inf')
    else:
        try:
            max_scores = []
            for seed in [0, 1, 2]:
                seedEverything(seed)
                
                # Train model
                bc_model, denoising_model, mean_scores = train_model_joint(config.NUM_DEMS, seed, config, save_ckpt=False)
                max_scores.append(np.max(mean_scores['denoising_joint_bc']))
                
            # Use the median of evaluation rewards as objective
            return np.mean(max_scores)
        
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
    
    # Clean up temporary file
    os.remove(tmp_config_path)
    
    # Create directory for study storage
    os.makedirs('hparam_results', exist_ok=True)
    
    # Create storage for study persistence
    storage_name = f"sqlite:///hparam_results/study_{env_name}.db"
    
    # Load existing study or create new one
    try:
        study = optuna.load_study(
            study_name=f"optimization_{env_name}",
            storage=storage_name
        )
        print(f"Loaded existing study for {env_name} with {len(study.trials)} trials")
    except:
        study = optuna.create_study(
            study_name=f"optimization_{env_name}",
            storage=storage_name,
            direction="maximize"
        )
        print(f"Created new study for {env_name}")
    
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
    parser.add_argument('--n_trials', type=int, default=5, help='Number of optimization trials per environment')
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