from train_model import train_imitation_agent
import os
from config import Config

if __name__ == "__main__":

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    n_dems_list = [5]#, 10, 20] # , 40 , 60, 100]
    random_seed_list = [ i for i in range(  5 )]
    
    Config.load_config_for_training("config.yaml")
    # Save a copy of the config file to the log directory
    import shutil
    
    # Ensure the base log directory exists
    os.makedirs(Config.BASE_LOG_PATH, exist_ok=True)
    
    # Copy the config file to the log directory
    config_copy_path = os.path.join(Config.BASE_LOG_PATH, 'config.yaml')
    shutil.copy2('config.yaml', config_copy_path)
    
    print(f"Config file copied to: {config_copy_path}")

    for seed in random_seed_list:
        # train
        for n_dems in n_dems_list:
            train_imitation_agent(n_dems, 2, seed, Config)
            #train_imitation_agent(n_dems, 1, seed)
            #train_imitation_agent(n_dems, 0, seed)