import os
import yaml
from config import CCIL_NUM_DEMS_MAP

EVAL_NOISE_LEVELS = {"metaworld-button-press-top-down-v2": [0.0, 2.0e-5, 5.0e-5, 7.0e-5, 0.0001, 0.00012, 0.00015, 0.0002],
                    "metaworld-coffee-pull-v2_50": [0.0, 2.0e-5, 5.0e-5, 7.0e-5, 0.0001, 0.00015, 0.0002, 0.00025, 0.0003],
                    "metaworld-coffee-push-v2_50": [0.0, 2.0e-5, 5.0e-5, 7.0e-5, 0.0001, 0.00015, 0.0002, 0.00025, 0.0003],
                    "metaworld-drawer-close-v2": [0.0, 2.0e-5, 5.0e-5, 7.0e-5, 0.0001, 0.00015, 0.0002, 0.00025, 0.0003, 0.0004, 0.0005],
                    "ant-expert-v2_10": [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1],
                    "halfcheetah-expert-v2_50": [0.0, 0.002, 0.003, 0.005, 0.007, 0.01, 0.015, 0.02, 0.1],
                    "hopper-expert-v2_25": [0.0, 0.002, 0.003, 0.005, 0.007, 0.01, 0.015, 0.02, 0.1],
                    "walker2d-expert-v2_20": [0.0, 0.005, 0.01, 0.015, 0.02, 0.03, 0.05, 0.075],
                    }

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
        config['epoch'] = 2000 # 600  # Default epochs for other environments
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