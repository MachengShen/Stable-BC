import os
import yaml
from datetime import datetime

CCIL_TASK_ENV_MAP = {
    # "pendulum_cont_100": "PendulumSwingupCont-v0",
    "walker2d-expert-v2_20": "walker2d-expert-v2",
    "hopper-expert-v2_25": "hopper-expert-v2",
    "ant-expert-v2_10": "ant-expert-v2",
    "halfcheetah-expert-v2_50": "halfcheetah-expert-v2",
    # "pendulum_disc_500": "PendulumSwingupDisc-v0",
    "metaworld-button-press-top-down-v2": "button-press-topdown-v2",
    "metaworld-coffee-push-v2_50": "coffee-push-v2",
    "metaworld-coffee-pull-v2_50": "coffee-pull-v2",
    "metaworld-drawer-close-v2": "drawer-close-v2",
    # "flythrugate_50": "flythrugate-aviary-v0",
    # "circle_50": "circle-aviary-v0",
    # "hover_5": "hover-aviary-v0",
    # "f1": "f110_gym:f110-v0",
}

class Config:
    TASK_TYPE = "sim-quadrotor"  # Options: "sim-quadrotor", "sim-intersection", "CCIL"
    CCIL_DATA_DIR = "/path/to/CCIL/data"  # Only needed for CCIL tasks
    CCIL_TASK_NAME = "task_name"  # Only needed for CCIL tasks, e.g., "door_opening"
    TIMESTAMP = None  # Will be set during training
    
    @classmethod
    def load_config_for_training(cls, config_path, timestamp=None):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        for key, value in config.items():
            setattr(cls, key.upper(), value)

        # Set up env config for CCIL compatibility
        if cls.TASK_TYPE == "CCIL":
            cls.env = CCIL_TASK_ENV_MAP[cls.CCIL_TASK_NAME]
        
        cls.STATE_LOSS_WEIGHT_BC = 1.0 - cls.ACTION_LOSS_WEIGHT_BC
        cls.STATE_LOSS_WEIGHT_DENOISING = 1.0 - cls.ACTION_LOSS_WEIGHT_DENOISING
        
        # Set timestamp if provided, otherwise generate new one
        cls.TIMESTAMP = timestamp or datetime.now().strftime("%Y%m%d-%H%M%S")
        
        # Set base path structure
        if cls.TASK_TYPE == "CCIL":
            cls.BASE_LOG_PATH = os.path.join(cls.BASE_LOG_PATH, cls.CCIL_TASK_NAME, cls.TIMESTAMP)
        else:
            cls.BASE_LOG_PATH = os.path.join(cls.BASE_LOG_PATH, cls.TASK_TYPE, cls.TIMESTAMP)

    @classmethod
    def load_config_for_testing(cls, config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        for key, value in config.items():
            setattr(cls, key.upper(), value)
        
        # Set up env config for CCIL compatibility
        if cls.TASK_TYPE == "CCIL":
            cls.env = CCIL_TASK_ENV_MAP[cls.CCIL_TASK_NAME]
        
        cls.STATE_LOSS_WEIGHT_BC = 1.0 - cls.ACTION_LOSS_WEIGHT_BC
        cls.STATE_LOSS_WEIGHT_DENOISING = 1.0 - cls.ACTION_LOSS_WEIGHT_DENOISING
        
        # Parse the config_path to set BASE_LOG_PATH
        cls.BASE_LOG_PATH = os.path.dirname(os.path.dirname(config_path))

    @classmethod
    def get_model_path(cls, num_dems, random_seed):
        return os.path.join(cls.BASE_LOG_PATH, "results", 
                          f"joint_training/{num_dems}dems/seed{random_seed}")

    @classmethod
    def get_tensorboard_path(cls, num_dems, random_seed):
        return os.path.join(cls.BASE_LOG_PATH, "tensorboard", 
                          f"joint_training_{num_dems}dems_seed{random_seed}")



# Load the config when the module is imported
# Config.load_config_for_training('./config.yaml')
