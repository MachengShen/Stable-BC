import os
import yaml
from datetime import datetime

class Config:
    @classmethod
    def load_config_for_training(cls, config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        for key, value in config.items():
            setattr(cls, key.upper(), value)

        cls.STATE_LOSS_WEIGHT_BC = 1.0 - cls.ACTION_LOSS_WEIGHT_BC
        cls.STATE_LOSS_WEIGHT_DENOISING = 1.0 - cls.ACTION_LOSS_WEIGHT_DENOISING
        # Set dynamic attributes for training
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        cls.TIMESTAMP = timestamp
        cls.BASE_LOG_PATH = os.path.join(cls.BASE_LOG_PATH, timestamp)
        
        # Save the config with the timestamp
        # cls.save_config(os.path.join(cls.BASE_LOG_PATH, 'config.yaml'))

    @classmethod
    def load_config_for_testing(cls, config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        for key, value in config.items():
            setattr(cls, key.upper(), value)
        
        cls.STATE_LOSS_WEIGHT_BC = 1.0 - cls.ACTION_LOSS_WEIGHT_BC
        cls.STATE_LOSS_WEIGHT_DENOISING = 1.0 - cls.ACTION_LOSS_WEIGHT_DENOISING
        # Parse the config_path to set BASE_LOG_PATH
        cls.BASE_LOG_PATH = os.path.dirname(os.path.dirname(config_path))
        cls.TIMESTAMP = os.path.basename(os.path.dirname(config_path))

    # @classmethod
    # def save_config(cls, config_path):
    #     config = {key.lower(): value for key, value in vars(cls).items() if not key.startswith('__') and not callable(value)}
    #     os.makedirs(os.path.dirname(config_path), exist_ok=True)
    #     with open(config_path, 'w') as f:
    #         yaml.dump(config, f, default_flow_style=False)

    @classmethod
    def get_model_path(cls, num_dems, random_seed):
        return os.path.join(cls.BASE_LOG_PATH, f"results_{cls.LR}lr_{cls.EPOCH}epoch", 
                            f"joint_training/{num_dems}dems/{random_seed}")

    @classmethod
    def get_tensorboard_path(cls, num_dems, random_seed):
        return os.path.join(cls.BASE_LOG_PATH, "tensorboard", f"joint_training_{num_dems}dems_{random_seed}")

# Load the config when the module is imported
# Config.load_config_for_training('./config.yaml')
