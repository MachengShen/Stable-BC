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

RESULTS_DIR = "/cephfs/cjyai/joint_denoising_hparam_results"

tune_noise_levels = {
    "metaworld-button-press-top-down-v2": [0.0001, 0.0005],
    "metaworld-coffee-push-v2_50": [0.0001, 0.0005],
    "metaworld-coffee-pull-v2_50": [0.0005, 0.001],
    "metaworld-drawer-close-v2": [0.0005, 0.001],
}


def get_default_envs():
    """Get list of MuJoCo and MetaWorld environments"""
    mujoco_envs = [
        "walker2d-expert-v2_20",
        "hopper-expert-v2_25",
        "ant-expert-v2_10",
        "halfcheetah-expert-v2_50",
    ]

    metaworld_envs = [
        "metaworld-button-press-top-down-v2",
        "metaworld-coffee-push-v2_50",
        "metaworld-coffee-pull-v2_50",
        "metaworld-drawer-close-v2",
    ]

    return metaworld_envs  # + mujoco_envs


def save_best_params(study, trial, env_name):
    """Save the best parameters to a JSON file with history"""
    if isinstance(trial.values, (list, tuple)):
        # For multi-objective optimization
        best_params = {
            "trial_number": trial.number,
            "total_trials": len(study.trials),
            "values": trial.values,  # List of values for each objective
            "params": trial.params,
            "timestamp": datetime.now().strftime("%Y%m%d-%H%M%S"),
        }
    else:
        # For single-objective optimization
        best_params = {
            "trial_number": trial.number,
            "total_trials": len(study.trials),
            "value": trial.value,
            "params": trial.params,
            "timestamp": datetime.now().strftime("%Y%m%d-%H%M%S"),
        }

    # Create results directory if it doesn't exist
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Load existing results or create new list
    json_filename = os.path.join(RESULTS_DIR, f"best_params_{env_name}.json")
    if os.path.exists(json_filename):
        with open(json_filename, "r") as f:
            history = json.load(f)
            if not isinstance(history, list):
                history = [history]  # Convert old format to list
    else:
        history = []

    # Append new best parameters
    history.append(best_params)

    # Sort history by value (descending) and add rank
    if isinstance(trial.values, (list, tuple)):
        # For multi-objective, sort by sum of normalized values
        for result in history:
            if isinstance(result.get("values", None), (list, tuple)):
                result["normalized_sum"] = sum(
                    v / max(abs(v), 1e-10) for v in result["values"]
                )
            else:
                result["normalized_sum"] = result.get("value", float("-inf"))
        history.sort(key=lambda x: x["normalized_sum"], reverse=True)
    else:
        # For single-objective
        history.sort(key=lambda x: x.get("value", float("-inf")), reverse=True)

    for i, result in enumerate(history):
        result["rank"] = i + 1

    # Save updated history
    with open(json_filename, "w") as f:
        json.dump(history, f, indent=4)


def get_training_epochs(env_name):
    """Get appropriate number of epochs based on environment"""
    mujoco_envs = [
        "walker2d-expert-v2_20",
        "hopper-expert-v2_25",
        "ant-expert-v2_10",
        "halfcheetah-expert-v2_50",
    ]

    if env_name in mujoco_envs:
        return {
            "epoch": 1000,  # More epochs for MuJoCo environments
            "diffusion_epoch": 5000,
        }
    else:
        return {
            "epoch": 1500,  # Default epochs for other environments
            "diffusion_epoch": 3000,
        }


def objective(trial, base_config, seed=0, debug=False, noise_levels=None):
    """Multi-objective optimization function that evaluates performance across all noise levels"""
    # Create a deep copy of the Config object
    config = copy.deepcopy(base_config)

    # Override number of epochs based on environment
    epochs = get_training_epochs(config.CCIL_TASK_NAME)
    for key, value in epochs.items():
        setattr(config, key.upper(), value)

    # Training parameters
    config.ACTION_LOSS_WEIGHT_DENOISING = trial.suggest_float(
        "action_loss_weight_denoising", 0.01, 0.7
    )
    config.STATE_NOISE_MULTIPLIER = trial.suggest_loguniform(
        "state_noise_multiplier", 0.00001, 0.03
    )

    if debug:
        # Run without try-except for debugging
        seedEverything(seed)

        # Train model
        bc_model, denoising_model, mean_scores = train_model_joint(
            config.NUM_DEMS,
            seed,
            config,
            save_ckpt=False,
            state_only_bc=True,
            eval_noise_levels=noise_levels,
            score_type='mean_reward',
        )

        # Return scores for all noise levels as multiple objectives
        return [
            (
                np.median(mean_scores["denoising_joint_bc"][noise])
                if mean_scores and len(mean_scores["denoising_joint_bc"][noise]) > 0
                else float("-inf")
            )
            for noise in noise_levels
        ]
    else:
        try:
            all_noise_scores = {noise: [] for noise in noise_levels}
            for seed in [0, 1, 2]:
                seedEverything(seed)

                # Train model
                bc_model, denoising_model, mean_scores = train_model_joint(
                    config.NUM_DEMS,
                    seed,
                    config,
                    save_ckpt=False,
                    state_only_bc=True,
                    eval_noise_levels=noise_levels,
                    score_type='mean_reward',
                )

                # Collect scores for each noise level
                for noise in noise_levels:
                    all_noise_scores[noise].append(
                        np.max(mean_scores["denoising_joint_bc"][noise])
                    )

            # Return mean scores for all noise levels as multiple objectives
            return [np.mean(all_noise_scores[noise]) for noise in noise_levels]

        except Exception as e:
            print(f"Trial failed with error: {str(e)}")
            return [float("-inf")] * len(noise_levels)


def optimize_env(env_name, base_config, n_trials, debug=False):
    """Run multi-objective optimization for a single environment"""
    print(f"\n{'='*80}")
    print(f"Optimizing for environment: {env_name}")
    print(f"{'='*80}")

    # Get noise levels for this environment
    noise_levels = tune_noise_levels.get(
        env_name, [0.0]
    )  # Default to [0.0] if not specified

    # Update config for this environment
    tmp_config = base_config.copy()
    tmp_config["ccil_task_name"] = env_name

    # Save temporary config to file
    os.makedirs("tmp", exist_ok=True)
    tmp_config_path = os.path.join("tmp", "tmp_config.yaml")
    with open(tmp_config_path, "w") as f:
        yaml.dump(tmp_config, f)

    # Load config using Config class method
    Config.load_config_for_training(tmp_config_path)

    # Clean up temporary file
    os.remove(tmp_config_path)

    # Create directory for study storage
    os.makedirs(f"{RESULTS_DIR}", exist_ok=True)

    # Create storage for study persistence
    storage_name = f"sqlite:///{RESULTS_DIR}/study_{env_name}_multi_objective.db"

    # Load existing study or create new one with multiple objectives
    try:
        study = optuna.load_study(
            study_name=f"optimization_{env_name}_multi_objective", storage=storage_name
        )
        print(f"Loaded existing study for {env_name} with {len(study.trials)} trials")
    except:
        study = optuna.create_study(
            study_name=f"optimization_{env_name}_multi_objective",
            storage=storage_name,
            directions=["maximize"]
            * len(noise_levels),  # Maximize performance for each noise level
        )
        print(f"Created new study for {env_name}")

    if debug:
        # Run without try-except for debugging
        study.optimize(
            lambda trial: objective(
                trial, Config, debug=debug, noise_levels=noise_levels
            ),
            n_trials=n_trials,
            callbacks=[
                lambda study, trial: save_best_params(
                    study, trial, f"{env_name}_multi_objective"
                )
            ],
        )
    else:
        # Run with try-except for production
        try:
            study.optimize(
                lambda trial: objective(
                    trial, Config, debug=debug, noise_levels=noise_levels
                ),
                n_trials=n_trials,
                callbacks=[
                    lambda study, trial: save_best_params(
                        study, trial, f"{env_name}_multi_objective"
                    )
                ],
            )
        except KeyboardInterrupt:
            print(f"\nOptimization stopped early for {env_name}.")
            if study.trials:
                save_best_params(
                    study, study.best_trials[0], f"{env_name}_multi_objective"
                )
                print("\nBest parameters so far have been saved.")

    # Get Pareto front
    pareto_front = study.best_trials

    # Format results
    best_results = {
        "pareto_front": [
            {
                "values": trial.values,  # List of values for each objective
                "noise_levels": noise_levels,  # Corresponding noise levels
                "params": trial.params,  # Hyperparameters
            }
            for trial in pareto_front
        ]
    }

    # Print Pareto front results
    print("\nPareto Front Results:")
    for i, trial in enumerate(pareto_front):
        print(f"\nSolution {i+1}:")
        print("Values:")
        for noise, value in zip(noise_levels, trial.values):
            print(f"  Noise {noise}: {value:.4f}")
        print("Parameters:")
        for key, value in trial.params.items():
            print(f"  {key}: {value}")

    return best_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Base config file')
    parser.add_argument('--n_trials', type=int, default=20, help='Number of optimization trials per environment')
    parser.add_argument('--envs', nargs='+', default=None, 
                       help='Specific environments to optimize. If not provided, will use all MuJoCo and MetaWorld envs')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode without try-except')
    args = parser.parse_args()

    # Load base config
    with open(args.config, "r") as f:
        base_config = yaml.safe_load(f)

    # Get environments to optimize
    envs_to_optimize = args.envs if args.envs is not None else get_default_envs()

    # Run optimization for each environment
    results = {}
    for env_name in envs_to_optimize:
        results[env_name] = optimize_env(env_name, base_config, args.n_trials, debug=args.debug)
    
    # Save overall results
    with open(os.path.join(RESULTS_DIR, "all_results.json"), "w") as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    # Set start method to spawn
    mp.set_start_method('spawn', force=True)
    main()
