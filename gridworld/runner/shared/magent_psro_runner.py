from collections import defaultdict    
from copy import deepcopy
import imageio
import numpy as np
import time
import torch

from gridworld.runner.shared.base_runner import Runner
from gridworld.utils.meta_solver import UniformSolver, NashSolver


def _t2n(x):
    return x.detach().cpu().numpy()


class MAgentPSRORunner(Runner):

    def __init__(self, config):
        super(MAgentPSRORunner, self).__init__(config)
        self.num_oppo_policy = self.all_args.num_oppo_policy
        self.old_env_infos = defaultdict(list)

        self.psro_eval_episodes = self.all_args.psro_eval_episodes
        self.payoffs = np.zeros((self.num_oppo_policy, self.num_oppo_policy))
        self.solver_type = self.all_args.meta_solver
        if self.solver_type == "uniform":
            self.meta_solver = UniformSolver()
        elif self.solver_type == "nash":
            self.meta_solver = NashSolver()

        # convergence related
        self.iter_min_steps = self.all_args.iter_min_steps
        self.iter_max_steps = self.all_args.iter_max_steps
        self.mean_threshold = self.all_args.mean_threshold
        self.std_threshold = self.all_args.std_threshold
        self.iter_start_steps = 0
        self.win_rate_list = []

    def run(self):
        self.warmup()   

        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads

        for episode in range(episodes):
            if self.use_linear_lr_decay:
                self.trainer.policy.lr_decay(episode, episodes)

            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads
            if total_num_steps % self.log_interval == 0:
                # log env info
                self.log_info = True
                self.env_infos = defaultdict(list)
            else:
                self.log_info = False

            for step in range(self.episode_length):
                # Sample actions
                values, actions, action_log_probs, rnn_states, rnn_states_critic = self.collect(step)
                    
                # Obser reward and next obs
                obs, rewards, dones, infos = self.envs.step(actions)

                data = obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic 
                
                # insert data into buffer
                self.insert(data)                    

            # compute return and update network
            self.compute()
            train_infos = self.train()
            
            # post process
            # save model
            if (total_num_steps % self.save_interval == 0 or episode == episodes - 1):
                self.save()

            # log information
            if total_num_steps % self.log_interval == 0:
                end = time.time()
                print("\n Env {} Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\n"
                        .format(self.env_name,
                                self.algorithm_name,
                                self.experiment_name,
                                episode,
                                episodes,
                                total_num_steps,
                                self.num_env_steps,
                                int(total_num_steps / (end - start))))

                for idx_oppo in range(self.num_oppo_policy):
                    idx = np.array(self.env_infos[f"oppo_prob_{idx_oppo}"])
                    if np.sum(idx) == 0:
                        self.env_infos[f"win_rate_vs_{idx_oppo}"] = self.old_env_infos[f"win_rate_vs_{idx_oppo}"]
                        self.env_infos[f"draw_rate_vs_{idx_oppo}"] = self.old_env_infos[f"draw_rate_vs_{idx_oppo}"]
                        self.env_infos[f"lose_rate_vs_{idx_oppo}"] = self.old_env_infos[f"lose_rate_vs_{idx_oppo}"]
                    else:                        
                        self.env_infos[f"win_rate_vs_{idx_oppo}"] = np.array(self.env_infos["left_win_rate"])[idx]
                        self.env_infos[f"draw_rate_vs_{idx_oppo}"] = np.array(self.env_infos["draw_rate"])[idx]
                        self.env_infos[f"lose_rate_vs_{idx_oppo}"] = np.array(self.env_infos["right_win_rate"])[idx]
                self.old_env_infos = deepcopy(self.env_infos)

                self.log_train(train_infos, total_num_steps)
                self.log_env(self.env_infos, total_num_steps)

                if self.converged(total_num_steps - self.iter_start_steps):
                    self.save_model()
                    self.expand()
                    self.warmup()

            # # eval
            # if total_num_steps % self.eval_interval == 0 and self.use_eval:
            #     self.eval(total_num_steps)

    def warmup(self):
        # reset env
        obs = self.envs.reset()
        self.buffer.share_obs[0] = obs.copy()
        self.buffer.obs[0] = obs.copy()

        self.buffer.rnn_states[0, ...] = 0
        self.buffer.rnn_states_critic[0, ...] = 0
        self.buffer.masks[0, ...] = 1
        self.buffer.bad_masks[0, ...] = 1
        self.buffer.active_masks[0, ...] = 1
        self.buffer.available_actions[0, ...] = 1
    
    def save_model(self):
        print(f"saving iteration {self.num_oppo_policy} model")
        if self.use_single_network:
            policy_model = self.trainer.policy.model
            torch.save(policy_model.state_dict(), str(self.save_dir) + f"/model_{self.num_oppo_policy}.pt")
        else:
            policy_actor = self.trainer.policy.actor
            torch.save(policy_actor.state_dict(), str(self.save_dir) + f"/actor_{self.num_oppo_policy}.pt")
            policy_critic = self.trainer.policy.critic
            torch.save(policy_critic.state_dict(), str(self.save_dir) + f"/critic_{self.num_oppo_policy}.pt")

    def converged(self, iter_steps):
        self.win_rate_list.append(np.mean(self.env_infos["left_win_rate"]))
        if len(self.win_rate_list) > 10:
            self.win_rate_list.pop(0)
        else:
            # print(f"win rate list length {len(self.win_rate_list)} <= 10, continue...")
            return False

        if iter_steps < self.iter_min_steps:
            # print(f"iteration step {iter_steps} < min {self.iter_min_steps} steps, continue...")
            return False
        elif iter_steps >= self.iter_max_steps:
            self.iter_start_steps += iter_steps
            self.win_rate_list = []
            print(f"PSRO iteration {self.num_oppo_policy}: {iter_steps} steps >= max {self.iter_max_steps} steps.")
            print(f"Continue to next iteration.")
            return True
        
        mean = np.mean(self.win_rate_list)
        std = np.std(self.win_rate_list)
        if mean >= self.mean_threshold and std <= self.std_threshold:
            self.iter_start_steps += iter_steps
            self.win_rate_list = []
            print(f"PSRO iteration {self.num_oppo_policy}: win rate mean {mean} >= {self.mean_threshold}, std {std} <= {self.std_threshold}.")
            print(f"Continue to next iteration.")
            return True
        else:
            # print(f"win rate mean {mean} < {self.mean_threshold} or std {std} > {self.std_threshold}, continue...")
            return False

    def expand(self):
        # eval and update payoffs
        if self.solver_type == "uniform":
            # don't call eval_all() to save some time.
            self.payoffs = np.zeros((self.num_oppo_policy + 1, self.num_oppo_policy + 1))
        else:
            self.eval_all()
        self.num_oppo_policy += 1
        # solve meta-policy
        self.meta_policy, _ = self.meta_solver.solve(np.stack([self.payoffs, -self.payoffs]))
        print(f"meta policy set to {self.meta_policy}")
        # update envs
        update_kwargs = dict(update_population=True, oppo_sample_probs=self.meta_policy)
        self.envs.update([update_kwargs for _ in range(self.n_rollout_threads)])
        if self.solver_type == "uniform":
            pass
        else:
            eval_meta_policy = np.ones(self.num_oppo_policy) / self.num_oppo_policy
            update_kwargs = dict(update_population=True, oppo_sample_probs=eval_meta_policy)
            self.eval_envs.update([update_kwargs for _ in range(self.n_eval_rollout_threads)])


    @torch.no_grad()
    def collect(self, step):    
        values, actions, action_log_probs, rnn_states, rnn_states_critic = self.trainer.policy.get_actions(
            np.concatenate(self.buffer.share_obs[step]),
            np.concatenate(self.buffer.obs[step]),
            np.concatenate(self.buffer.rnn_states[step]),
            np.concatenate(self.buffer.rnn_states_critic[step]),
            np.concatenate(self.buffer.masks[step])
        )

        values = np.array(np.split(_t2n(values), self.n_rollout_threads))
        actions = np.array(np.split(_t2n(actions), self.n_rollout_threads))
        action_log_probs = np.array(np.split(_t2n(action_log_probs), self.n_rollout_threads))
        rnn_states = np.array(np.split(_t2n(rnn_states), self.n_rollout_threads))
        rnn_states_critic = np.array(np.split(_t2n(rnn_states_critic), self.n_rollout_threads))

        return values, actions, action_log_probs, rnn_states, rnn_states_critic

    def insert(self, data):
        obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic = data

        dones_env = np.all(dones, axis=1)
        rnn_states[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
        rnn_states_critic[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

        active_masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        active_masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)
        active_masks[dones_env == True] = np.ones(((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

        self.buffer.insert(
            share_obs=obs, 
            obs=obs, 
            rnn_states=rnn_states, 
            rnn_states_critic=rnn_states_critic,
            actions=actions, 
            action_log_probs=action_log_probs, 
            value_preds=values, 
            rewards=rewards, 
            masks=masks, 
            active_masks=active_masks, 
        )
        
        if self.log_info and any(dones_env):
            for idx_env in np.where(dones_env == True)[0]:
                for k, v in infos[idx_env].items():
                    self.env_infos[k].append(v)
    
    @torch.no_grad()
    def eval_all(self):
        eval_episodes = np.zeros(self.num_oppo_policy)
        eval_payoffs = defaultdict(list)

        eval_obs = self.eval_envs.reset()
        eval_rnn_states = np.zeros((self.n_eval_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
        eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)

        while True:
            self.trainer.prep_rollout()

            eval_actions, eval_rnn_states = self.trainer.policy.act(
                np.concatenate(eval_obs),
                np.concatenate(eval_rnn_states),
                np.concatenate(eval_masks),
                deterministic=False
            )

            eval_actions = np.array(np.split(_t2n(eval_actions), self.n_eval_rollout_threads))
            eval_rnn_states = np.array(np.split(_t2n(eval_rnn_states), self.n_eval_rollout_threads))

            # Obser reward and next obs
            eval_obs, eval_rewards, eval_dones, eval_infos = self.eval_envs.step(eval_actions)
            
            eval_dones_env = np.all(eval_dones, axis=1)
            eval_rnn_states[eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(), self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
            eval_masks = np.ones((self.all_args.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
            eval_masks[eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

            for eval_i in range(self.n_eval_rollout_threads):
                if eval_dones_env[eval_i]:
                    oppo_idx = eval_infos[eval_i]["oppo_idx"]
                    eval_episodes[oppo_idx] += 1
                    eval_payoffs[oppo_idx].append(int(eval_infos[eval_i]["left_win_rate"]) - int(eval_infos[eval_i]["right_win_rate"]))

            if np.min(eval_episodes) >= self.psro_eval_episodes:
                payoffs = np.zeros((self.num_oppo_policy + 1, self.num_oppo_policy + 1))
                payoffs[:-1, :-1] = deepcopy(self.payoffs)
                for oppo_i in range(self.num_oppo_policy):
                    payoffs[-1, oppo_i] = np.mean(eval_payoffs[oppo_idx])
                    payoffs[oppo_i, -1] = -payoffs[-1, oppo_i]
                self.payoffs = payoffs
                print(f"PSRO iteration {self.num_oppo_policy} payoffs:")
                print(self.payoffs)
                break

    @torch.no_grad()
    def eval(self, total_num_steps):
        eval_episode = 0
        eval_env_infos = defaultdict(list)

        eval_obs = self.eval_envs.reset()
        eval_rnn_states = np.zeros((self.n_eval_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
        eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)

        while True:
            self.trainer.prep_rollout()

            eval_actions, eval_rnn_states = self.trainer.policy.act(
                np.concatenate(eval_obs),
                np.concatenate(eval_rnn_states),
                np.concatenate(eval_masks),
                deterministic=self.all_args.eval_deterministic
            )
            
            eval_actions = np.array(np.split(_t2n(eval_actions), self.n_eval_rollout_threads))
            eval_rnn_states = np.array(np.split(_t2n(eval_rnn_states), self.n_eval_rollout_threads))

            # Obser reward and next obs
            eval_obs, eval_rewards, eval_dones, eval_infos = self.eval_envs.step(eval_actions)
            
            eval_dones_env = np.all(eval_dones, axis=1)
            eval_rnn_states[eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(), self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
            eval_masks = np.ones((self.all_args.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
            eval_masks[eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

            for eval_i in range(self.n_eval_rollout_threads):
                if eval_dones_env[eval_i]:
                    eval_episode += 1
                    for k, v in eval_infos[eval_i].items():
                        eval_env_infos[k].append(v)

            if eval_episode >= self.all_args.eval_episodes:
                self.log_env(eval_env_infos, total_num_steps)
                for k, v in eval_env_infos.items():
                    print(f"eval {k} is {np.mean(v)}.")
                break

    @torch.no_grad()
    def render(self):
        render_episode = 0
        render_env_infos = defaultdict(list)

        render_obs = self.envs.reset()

        render_rnn_states = np.zeros((self.n_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
        render_masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)

        while True:
            self.trainer.prep_rollout()

            render_actions, render_rnn_states = self.trainer.policy.act(
                np.concatenate(render_obs),
                np.concatenate(render_rnn_states),
                np.concatenate(render_masks),
                deterministic=self.all_args.render_deterministic
            )
            
            render_actions = np.array(np.split(_t2n(render_actions), self.n_rollout_threads))
            render_rnn_states = np.array(np.split(_t2n(render_rnn_states), self.n_rollout_threads))

            # Obser reward and next obs
            render_obs, render_rewards, render_dones, render_infos = self.envs.step(render_actions)
            images = self.envs.render(mode="rgb_array")
            
            render_dones_env = np.all(render_dones, axis=1)
            render_rnn_states[render_dones_env == True] = np.zeros(((render_dones_env == True).sum(), self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
            render_masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
            render_masks[render_dones_env == True] = np.zeros(((render_dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

            for render_i in range(self.n_rollout_threads):
                if render_dones_env[render_i]:
                    render_episode += 1
                    for k, v in render_infos[render_i].items():
                        if k == "frames":
                            # pass
                            imageio.mimsave(str(self.gif_dir) + f"/render{render_episode}.gif", v, duration=0.5)
                        else:
                            render_env_infos[k].append(v)

            if render_episode >= self.all_args.render_episodes:
                for k, v in render_env_infos.items():
                    print(f"eval {k} is {np.mean(v)}.")
                break
