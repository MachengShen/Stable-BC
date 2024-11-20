LOG_STAMP='hopper-expert-v2_25/20241120-202305/'
SEED='seed42'
CONFIG_PATH="/cephfs/cjyai/joint_denoising_bc/${LOG_STAMP}config.yaml"
CKPT_DIR="/cephfs/cjyai/joint_denoising_bc/${LOG_STAMP}results/joint_training/5dems/${SEED}/"

python eval_model.py \
    --config_path $CONFIG_PATH \
    --checkpoint_dir $CKPT_DIR \
    --num_eval_episodes 2