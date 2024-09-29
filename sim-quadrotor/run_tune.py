import optuna
from train_model import train_model_joint
from test_model import test_imitation_agent
import os
import numpy as np


def objective(trial):
    # Define hyperparameters to tune
    overall_noise_factor = trial.suggest_float("overall_noise_factor", 0.01, 0.1)
    action_loss_weight = trial.suggest_float("action_loss_weight", 0.3, 0.9)
    state_loss_weight = 1 - action_loss_weight
    LR = trial.suggest_float("learning_rate", 1e-4, 1e-3, log=True)

    # Fixed parameters
    num_dems = 10
    random_seed = np.random.randint(1, 11)
    EPOCH = 100

    # Tunable parameter

    # Train the model with the current hyperparameters
    bc_model, denoising_model = train_model_joint(
        num_dems,
        random_seed,
        EPOCH,
        LR,
        overall_noise_factor,
        action_loss_weight,
        state_loss_weight,
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
            "sim-quadrotor/results_0.001lr_1000epoch/joint_training",
            early_return=True,
            bc_model=bc_model,
            denoising_model=denoising_model,
        )
        success_rates.append(success_rate)

    # Calculate the average success rate
    average_success_rate = np.mean(success_rates)

    return average_success_rate


def run_hyperparameter_tuning():
    study = optuna.create_study(direction="maximize")
    
    best_value = float('-inf')
    best_trial = None
    best_trial_file = 'best_trials.txt'

    for trial_number in range(50):
        trial = study.ask()
        value = objective(trial)
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

    print("\nFinal Best trial:")
    print(f"Value: {best_value}")
    print("Params:")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    run_hyperparameter_tuning()
