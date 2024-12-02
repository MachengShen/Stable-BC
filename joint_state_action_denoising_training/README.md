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


## Example results of gathered_results.txt:
```
Task: metaworld-button-press-top-down-v2
================================

Timestamp: 20241130-152913, seed0
--------------------------------
Evaluation Results (Methods: baseline, joint_bc, joint_denoising, joint_state_only_bc)
=================

Noise Level: 0.0
--------------------------------------------------

baseline:
  Mean Reward: 3189.32 ± 121.14
  Median Reward: 3347.51
  Success Rate: 0.97

joint_bc:
  Mean Reward: 3432.39 ± 87.39
  Median Reward: 3442.51
  Success Rate: 1.00

joint_denoising:
  Mean Reward: 3148.68 ± 155.16
  Median Reward: 3385.01
  Success Rate: 0.93

joint_state_only_bc:
  Mean Reward: 3517.28 ± 103.06
  Median Reward: 3798.18
  Success Rate: 0.97

==================================================
Noise Level: 2e-05
--------------------------------------------------

baseline:
  Mean Reward: 1346.98 ± 48.73
  Median Reward: 1311.47
  Success Rate: 0.27

joint_bc:
  Mean Reward: 1401.81 ± 46.13
  Median Reward: 1396.27
  Success Rate: 0.33

joint_denoising:
  Mean Reward: 1451.92 ± 49.62
  Median Reward: 1405.74
  Success Rate: 0.23

joint_state_only_bc:
  Mean Reward: 3382.84 ± 119.37
  Median Reward: 3707.93
  Success Rate: 1.00

==================================================
Noise Level: 5e-05
--------------------------------------------------

baseline:
  Mean Reward: 1334.52 ± 23.63
  Median Reward: 1377.22
  Success Rate: 0.00

joint_bc:
  Mean Reward: 1276.49 ± 29.82
  Median Reward: 1262.22
  Success Rate: 0.00

joint_denoising:
  Mean Reward: 1272.93 ± 26.80
  Median Reward: 1239.27
  Success Rate: 0.00

joint_state_only_bc:
  Mean Reward: 1960.08 ± 126.11
  Median Reward: 1744.49
  Success Rate: 0.73

==================================================
Noise Level: 7e-05
--------------------------------------------------

baseline:
  Mean Reward: 1282.04 ± 19.16
  Median Reward: 1285.18
  Success Rate: 0.00

joint_bc:
  Mean Reward: 1327.27 ± 23.05
  Median Reward: 1366.52
  Success Rate: 0.00

joint_denoising:
  Mean Reward: 1292.64 ± 26.68
  Median Reward: 1349.76
  Success Rate: 0.00

joint_state_only_bc:
  Mean Reward: 1462.09 ± 35.95
  Median Reward: 1453.19
  Success Rate: 0.27

==================================================
Noise Level: 0.0001
--------------------------------------------------

baseline:
  Mean Reward: 1297.84 ± 14.61
  Median Reward: 1315.59
  Success Rate: 0.00

joint_bc:
  Mean Reward: 1293.06 ± 14.08
  Median Reward: 1318.76
  Success Rate: 0.00

joint_denoising:
  Mean Reward: 1265.60 ± 17.39
  Median Reward: 1298.24
  Success Rate: 0.00

joint_state_only_bc:
  Mean Reward: 1370.55 ± 39.87
  Median Reward: 1379.52
  Success Rate: 0.07

==================================================
Noise Level: 0.00012
--------------------------------------------------

baseline:
  Mean Reward: 1312.19 ± 11.74
  Median Reward: 1338.17
  Success Rate: 0.00

joint_bc:
  Mean Reward: 1276.54 ± 14.65
  Median Reward: 1291.44
  Success Rate: 0.00

joint_denoising:
  Mean Reward: 1299.71 ± 15.15
  Median Reward: 1315.92
  Success Rate: 0.00

joint_state_only_bc:
  Mean Reward: 1456.06 ± 48.38
  Median Reward: 1510.08
  Success Rate: 0.00
```