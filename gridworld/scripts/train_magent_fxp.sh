#!/bin/sh
algo="rmappo"
exp="fxp"
seed=0
wandb_name="${exp}-seed${seed}"
env="MAgentXP"
scenario="battle"
map_size=15
max_episode_length=200
num_agents=3
fp_interval=5000000
main_sp_prob=0.2
main_fsp_prob=0.4
main_xp_prob=0
main_fxp_prob=0.4
counter_sp_prob=0
counter_fsp_prob=0
counter_xp_prob=0
counter_fxp_prob=1.0
hidden_size=64
layer_N=1
num_env_steps=100000000
n_rollout_threads=100
episode_length=200
data_chunk_length=10
num_mini_batch=1
ppo_epoch=5
log_interval=100000
save_interval=20000

CUDA_VISIBLE_DEVICES=0 python train/train_magent_fxp.py \
--algorithm_name ${algo} --experiment_name ${exp} --seed ${seed} \
--env_name ${env} --scenario_name ${scenario} --map_size ${map_size} \
--max_episode_length ${max_episode_length} --num_agents ${num_agents} \
--fp_interval ${fp_interval} \
--main_sp_prob ${main_sp_prob} --main_fsp_prob ${main_fsp_prob} \
--main_xp_prob ${main_xp_prob} --main_fxp_prob ${main_fxp_prob} \
--counter_sp_prob ${counter_sp_prob} --counter_fsp_prob ${counter_fsp_prob} \
--counter_xp_prob ${counter_xp_prob} --counter_fxp_prob ${counter_fxp_prob} \
--hidden_size ${hidden_size} --layer_N ${layer_N} \
--num_env_steps ${num_env_steps} --n_rollout_threads ${n_rollout_threads} \
--episode_length ${episode_length} --data_chunk_length ${data_chunk_length} \
--num_mini_batch ${num_mini_batch} --ppo_epoch ${ppo_epoch} \
--log_interval ${log_interval} --save_interval ${save_interval} \
--user_name "zelaix" --wandb_name ${wandb_name}
