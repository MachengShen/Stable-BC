from test_model import test_imitation_agent
import time
import os

if __name__ == "__main__":

    os.environ["CUDA_VISIBLE_DEVICES"] = "2"

    n_dems_list = [
        5,
        # 10,
        # 20,
    ]  # , 40, 60, 100 ] # [ 5, 10, 15, 20, 25 ] # [ i for i in range( 1, 10 )]
    # random_seed_list = [i for i in range(1)]  # [ i for i in range( 10 )]
    random_seed_list = [8]
    test_name_list = ["training_region"]

    base_path_list = ["sim-quadrotor/logs/20240930-231316/results_0.0016439960878324687lr_300epoch"]#"sim-quadrotor/results_0.001lr_1000epoch/lamda_0.0001", 

    for base_path in base_path_list:
        for n_dems in n_dems_list:
            for test_name in test_name_list:
                for model_seed in random_seed_list:
                    success_rate_list = []
                    for rollout_seed in list(range(10)):

                        start_time = time.time()

                        success_rate = test_imitation_agent(
                            n_dems, 2, model_seed, test_name, rollout_seed=rollout_seed, base_path=base_path, early_return=True
                        )
                        # test_imitation_agent(n_dems, 1, seed, test_name, base_path)
                        # test_imitation_agent(n_dems, 0, seed, test_name, base_path)
                        success_rate_list.append(success_rate)
                        end_time = time.time()
                        iteration_time = (end_time - start_time) / 60
                        print(
                            "n_dems: ",
                            n_dems,
                            "model_seed: ",
                            model_seed,
                            "rollout_seed: ",
                            rollout_seed,
                            "test_name: ",
                            test_name,
                            "iteration_time: ",
                            iteration_time,
                            "success_rate: ",
                            success_rate,
                        )
                    print("Average success rate: ", sum(success_rate_list) / len(success_rate_list))
