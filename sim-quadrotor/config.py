import os
import yaml
from datetime import datetime

class Config:
    @classmethod
    def load_config(cls, config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        for key, value in config.items():
            setattr(cls, key.upper(), value)

        # Set dynamic attributes
        cls.BASE_LOG_PATH = os.path.join(cls.BASE_LOG_PATH, datetime.now().strftime("%Y%m%d-%H%M%S"))

    @classmethod
    def save_config(cls, config_path):
        config = {key.lower(): value for key, value in vars(cls).items() if not key.startswith('__') and not callable(value)}
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)

    @classmethod
    def get_model_path(cls, num_dems, random_seed):
        return os.path.join(cls.BASE_LOG_PATH, f"results_{cls.LR}lr_{cls.EPOCH}epoch", 
                            f"joint_training/{num_dems}dems/{random_seed}")

    @classmethod
    def get_tensorboard_path(cls, num_dems, random_seed):
        return os.path.join(cls.BASE_LOG_PATH, "tensorboard", f"joint_training_{num_dems}dems_{random_seed}")

# Load the config when the module is imported
Config.load_config('./config.yaml')
