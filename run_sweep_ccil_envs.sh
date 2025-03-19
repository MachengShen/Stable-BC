# Run a specific task
python sweep_ccil_envs.py --task pendulum_cont_100 --seeds 42

# Run all tasks
python sweep_ccil_envs.py --seeds 42 43 44

# Run only training for all tasks
python sweep_ccil_envs.py --mode train

# Run only evaluation for all tasks
python sweep_ccil_envs.py --mode eval