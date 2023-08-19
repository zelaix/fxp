#!/bin/sh
algo="rmappo"
exp="fsp"
seed=0
wandb_name="${exp}-seed${seed}"
env="MAgent"
scenario="battle"
map_size=15
max_episode_length=200
num_agents=3
sp_prob=0.2
fp_interval=1000000
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

CUDA_VISIBLE_DEVICES=0 python train/train_magent_fsp.py \
--algorithm_name ${algo} --experiment_name ${exp} --seed ${seed} \
--env_name ${env} --scenario_name ${scenario} --map_size ${map_size} \
--max_episode_length ${max_episode_length} --num_agents ${num_agents} \
--one_side --use_sp --sp_prob ${sp_prob} \
--use_population --fp_interval ${fp_interval} \
--hidden_size ${hidden_size} --layer_N ${layer_N} \
--num_env_steps ${num_env_steps} --n_rollout_threads ${n_rollout_threads} \
--episode_length ${episode_length} --data_chunk_length ${data_chunk_length} \
--num_mini_batch ${num_mini_batch} --ppo_epoch ${ppo_epoch} \
--log_interval ${log_interval} --save_interval ${save_interval} \
--user_name "zelaix" --wandb_name ${wandb_name}
