import optuna
from train_model import train_model_joint, save_models
from test_model import test_imitation_agent
import os
import numpy as np
from config import Config

def objective(trial):
    # Define hyperparameters to tune
    Config.ACTION_NOISE_MULTIPLIER = trial.suggest_float("action_noise_multiplier", 0.01, 0.1)
    Config.STATE_NOISE_MULTIPLIER = trial.suggest_float("state_noise_multiplier", 0.01, 0.1)
    Config.ACTION_LOSS_WEIGHT_BC = trial.suggest_float("action_loss_weight_bc", 0.3, 1.0)
    Config.ACTION_LOSS_WEIGHT_DENOISING = trial.suggest_float("action_loss_weight_denoising", 0.05, 1.0)
    Config.LR = trial.suggest_float("learning_rate", 5e-4, 2e-3, log=True)

    # Fixed parameters
    num_dems = Config.NUM_DEMS
    random_seed = np.random.randint(1, 11)

    # Train the model with the current hyperparameters
    bc_model, denoising_model = train_model_joint(
        num_dems,
        random_seed,
        Config,
        save_ckpt=False,
    )

    # Test the model and get the success rate
    # Sample 10 random seeds from 100 random seeds
    random_seeds = np.random.choice(100, 10, replace=False)
    success_rates = []

    for seed in random_seeds:
        success_rate = test_imitation_agent(
            num_dems,
            2,
            seed,
            "training_region",
            Config.BASE_LOG_PATH,
            early_return=True,
            bc_model=bc_model,
            denoising_model=denoising_model,
        )
        success_rates.append(success_rate)

    # Calculate the average success rate
    average_success_rate = np.mean(success_rates)

    return average_success_rate, bc_model, denoising_model, num_dems, random_seed

def run_hyperparameter_tuning():
    study = optuna.create_study(direction="maximize")
    
    best_value = float('-inf')
    best_trial = None
    best_trial_file = f'best_trials_num_dems_{Config.NUM_DEMS}.txt'

    for trial_number in range(500):
        trial = study.ask()
        value, bc_model, denoising_model, num_dems, random_seed = objective(trial)
        study.tell(trial, value)

        if value > best_value:
            best_value = value
            best_trial = trial
            
            # Save intermediate best trial
            print(f"New best trial (#{trial_number}):")
            print(f"Value: {best_value}")
            print("Params:")
            for key, value in best_trial.params.items():
                print(f"    {key}: {value}")
            
            # Append new best trial to file
            with open(best_trial_file, 'a') as f:
                f.write(f"\nNew best trial (#{trial_number})\n")
                f.write(f"Value: {best_value}\n")
                f.write("Params:\n")
                for key, value in best_trial.params.items():
                    f.write(f"    {key}: {value}\n")
                    
            save_models(bc_model, denoising_model, num_dems, random_seed, Config)

    print("\nFinal Best trial:")
    print(f"Value: {best_value}")
    print("Params:")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")

if __name__ == "__main__":
    Config.load_config_for_training("config.yaml")
    os.environ["CUDA_VISIBLE_DEVICES"] = Config.CUDA_VISIBLE_DEVICES
    run_hyperparameter_tuning()
