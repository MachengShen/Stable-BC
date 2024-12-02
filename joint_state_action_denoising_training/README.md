# Joint State-Action Denoising BC

This repository contains code for training and evaluating behavioral cloning models with joint state-action prediction and denoising.

## Prerequisites

1. Dataset Location
- CCIL dataset should be placed under `../CCIL-main/data/`
- The dataset structure should be:
```
CCIL-main/
└── data/
    ├── ant-expert-v2_10.pkl
    ├── metaworld-drawer-close-v2.pkl
    ├── walker2d-expert-v2_20.pkl
    └── ...
```
You can download the CCIL dataset at https://github.com/personalrobotics/CCIL.git


2. Log Directory
- Before running any scripts, modify the `base_log_path` in `config.yaml` to your desired log directory path
- Current default is `/cephfs/cjyai/joint_denoising_bc_save_best/`
- Make sure you have write permissions to this directory

## Training

Use `sweep_ccil_envs.py` to train models on multiple environments:

```bash
# Train on all environments with default settings
python sweep_ccil_envs.py --config config.yaml --seeds 0 1 2 --mode train

# Train on specific environment
python sweep_ccil_envs.py --config config.yaml --seeds 0 1 2 --task ant-expert-v2_10 --mode train

# Train with state-only BC setting, this is the default setting of our proposed methods
python sweep_ccil_envs.py --config config.yaml --seeds 0 1 2 --state_only_bc --mode train
```
Note: "--mode eval" or "--mode both" may be broken

The script will:
1. Create timestamped directories for each environment
2. Train models with multiple random seeds
3. Save models and logs in the specified base_log_path. For each seed, the code will save the best model based on evaluation performance (based on a few evaluation episodes) during training.
4. The num_of_dems (number of trajectories) used for training is overwritten by the update_config function in `sweep_ccil_envs.py`.

## Directory Structure

After running the training script, your directory structure should look like:
```
base_log_path/
└── task_name/
    └── YYYYMMDD-HHMMSS/
        ├── config.yaml
        ├── results/
        │   └── joint_training/
        │       └── Ndems/
        │           ├── seed0/
        │           ├── seed1/
        │           └── seed2/
        └── tensorboard/
```

## Evaluation

After training, use `sweep_eval_model.sh` to evaluate trained models (based on more evaluation episodes), this script will find all the subdirectories in the log directory and evaluate each model:

1. First modify the ROOT_DIR in the script to match your log directory:
```bash
# In sweep_eval_model.sh
ROOT_DIR="/path/to/your/log/directory"  # Should match base_log_path in config.yaml
```

2. Run the evaluation:
```bash
bash sweep_eval_model.sh
```

The script will:
1. Find all trained models in the subfolders of the log directory recursively
2. Evaluate each model with different noise levels (specified in eval_model.py, with its default noise-level values specified in misc.py)
3. Save evaluation results in each model's directory

## Gathering Results

Use `gather_results.sh` to collect all evaluation results:

1. Modify the ROOT_DIR in the script:
```bash
# In gather_results.sh
ROOT_DIR="/path/to/your/log/directory"  # Should match the evaluation directory
```

2. Run the gathering script:
```bash
bash gather_results.sh
```

This will:
1. Collect all evaluation results
2. Combine them into a single file `gathered_results.txt`
3. Organize results by task and seed

## Common Issues

1. Dataset Path
- Error: "No such file or directory: '../CCIL-main/data/'"
- Solution: Make sure CCIL dataset is in the correct location

## Notes

- The evaluation script by default evaluates 
    - baseline BC
    - joint BC: jointly predicting next state and action
    - joint denoising models: jointly predicting next state and action, then apply denoising
    - joint state-only BC denoising models (our proposed method): only predicting next state, then applying denoising to get both next state and action

## Hyperparameter Tuning

For hyperparameter optimization across multiple environments, use `tune_hyperparams.py`:

```bash
# Tune all environments
python tune_hyperparams.py --config config.yaml --n_trials 50

# Tune specific environments
python tune_hyperparams.py --config config.yaml --n_trials 50 --envs ant-expert-v2_10 metaworld-drawer-close-v2

# Debug mode (shows full error traceback)
python tune_hyperparams.py --config config.yaml --debug
```

The script will:
1. Run hyperparameter optimization using Optuna
2. Save the best models during optimization
3. Track the best hyperparameters in a text file

Key hyperparameters being tuned:
- Learning rate
- Action/state noise multipliers 
- Loss weights for BC and denoising
- Weight decay

Note that current hyperparameter tuning is written for joint denoising models, for state-only denoising models, the action noise multipliers and action denoising loss weights are absent.

The tuning results will be saved in:
- `hparam_results/best_params_{env_name}.json`: Best parameters for each environment
- `hparam_results/study_{env_name}.db`: Full optimization history (SQLite database)

Note: The optimization can be resumed by running the same command again, it will continue from the last trial.


