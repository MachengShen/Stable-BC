CONFIG_PATH=/cephfs/cjyai/joint_denoising_bc/ant-expert-v2_10/20241121-121053/config.yaml
CKPT_DIR=/cephfs/cjyai/joint_denoising_bc/ant-expert-v2_10/20241121-121053/results/joint_training/5dems/seed42/
python3 analyze_trajectories.py --config_path $CONFIG_PATH --checkpoint_dir $CKPT_DIR