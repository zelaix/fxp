from collections import defaultdict    
from copy import deepcopy
import imageio
import numpy as np
import time
import torch

from gridworld.runner.cross_play.base_runner import Runner


def _t2n(x):
    return x.detach().cpu().numpy()


class MAgentFXPRunner(Runner):

    def __init__(self, config):
        super(MAgentFXPRunner, self).__init__(config)
        self.update_kwargs = dict(update_main=True, update_counter=True)
        self.fp_interval = self.all_args.fp_interval
        self.main_pop_size = 0
        self.counter_pop_size = 0

    def run(self):
        self.save()
        self.main_envs.update([self.update_kwargs for _ in range(self.n_rollout_threads)])
        self.counter_envs.update([self.update_kwargs for _ in range(self.n_rollout_threads)])

        self.save_model("main")
        self.save_model("counter")
        main_pop_sample_probs = np.ones(self.main_pop_size) / self.main_pop_size
        counter_pop_sample_probs = np.ones(self.counter_pop_size) / self.counter_pop_size
        update_kwargs = dict(
            update_main_pop=True, 
            main_pop_sample_probs=main_pop_sample_probs,
            update_counter_pop=True,
            counter_pop_sample_probs=counter_pop_sample_probs,
        )
        self.main_envs.update([update_kwargs for _ in range(self.n_rollout_threads)])
        self.counter_envs.update([update_kwargs for _ in range(self.n_rollout_threads)])
        self.warmup()

        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads

        for episode in range(episodes):

            if self.use_linear_lr_decay:
                self.main_trainer.policy.lr_decay(episode, episodes)
                self.counter_trainer.policy.lr_decay(episode, episodes)

            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads
            if total_num_steps % self.log_interval == 0:
                # log env info
                self.log_info = True
                self.main_env_infos = defaultdict(list)
                self.counter_env_infos = defaultdict(list)
            else:
                self.log_info = False

            for step in range(self.episode_length):
                # Main.
                values, actions, action_log_probs, rnn_states, rnn_states_critic = self.collect(step, "main")
                obs, rewards, dones, infos = self.main_envs.step(actions)
                data = obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic 
                self.insert(data, "main")
                # Counter.
                values, actions, action_log_probs, rnn_states, rnn_states_critic = self.collect(step, "counter")
                obs, rewards, dones, infos = self.counter_envs.step(actions)
                data = obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic 
                self.insert(data, "counter")

            # compute return and update network
            self.compute()
            main_train_infos, counter_train_infos = self.train()
            
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

                self.update_env_info("main")
                self.update_env_info("counter")

                self.log_train(main_train_infos, counter_train_infos, total_num_steps)
                self.log_env(self.main_env_infos, self.counter_env_infos, total_num_steps)

            # post process
            # save model
            if (total_num_steps % self.save_interval == 0 or episode == episodes - 1):
                self.save()
                self.main_envs.update([self.update_kwargs for _ in range(self.n_rollout_threads)])
                self.counter_envs.update([self.update_kwargs for _ in range(self.n_rollout_threads)])

            # update population
            if total_num_steps % self.fp_interval == 0:
                self.save_model("main")
                self.save_model("counter")
                main_pop_sample_probs = np.ones(self.main_pop_size) / self.main_pop_size
                counter_pop_sample_probs = np.ones(self.counter_pop_size) / self.counter_pop_size
                update_kwargs = dict(
                    update_main_pop=True, 
                    main_pop_sample_probs=main_pop_sample_probs,
                    update_counter_pop=True,
                    counter_pop_sample_probs=counter_pop_sample_probs,
                )
                self.main_envs.update([update_kwargs for _ in range(self.n_rollout_threads)])
                self.counter_envs.update([update_kwargs for _ in range(self.n_rollout_threads)])
                self.warmup()

    def warmup(self):
        # reset main env
        obs = self.main_envs.reset()
        self.main_buffer.share_obs[0] = obs.copy()
        self.main_buffer.obs[0] = obs.copy()
        self.main_buffer.rnn_states[0, ...] = 0
        self.main_buffer.rnn_states_critic[0, ...] = 0
        self.main_buffer.masks[0, ...] = 1
        self.main_buffer.bad_masks[0, ...] = 1
        self.main_buffer.active_masks[0, ...] = 1
        self.main_buffer.available_actions[0, ...] = 1
        # reset counter env
        obs = self.counter_envs.reset()
        self.counter_buffer.share_obs[0] = obs.copy()
        self.counter_buffer.obs[0] = obs.copy()
        self.counter_buffer.rnn_states[0, ...] = 0
        self.counter_buffer.rnn_states_critic[0, ...] = 0
        self.counter_buffer.masks[0, ...] = 1
        self.counter_buffer.bad_masks[0, ...] = 1
        self.counter_buffer.active_masks[0, ...] = 1
        self.counter_buffer.available_actions[0, ...] = 1

    def save_model(self, name):
        if name == "main":
            print(f"saving {name}'s {self.main_pop_size} iteration model")
            if self.use_single_network:
                policy_model = self.main_trainer.policy.model
                torch.save(policy_model.state_dict(), str(self.save_dir) + f"/main/model_{self.main_pop_size}.pt")
            else:
                policy_actor = self.main_trainer.policy.actor
                torch.save(policy_actor.state_dict(), str(self.save_dir) + f"/main/actor_{self.main_pop_size}.pt")
                policy_critic = self.main_trainer.policy.critic
                torch.save(policy_critic.state_dict(), str(self.save_dir) + f"/main/critic_{self.main_pop_size}.pt")
            self.main_pop_size += 1
        elif name == "counter":
            print(f"saving {name}'s {self.counter_pop_size} iteration model")
            if self.use_single_network:
                policy_model = self.counter_trainer.policy.model
                torch.save(policy_model.state_dict(), str(self.save_dir) + f"/counter/model_{self.counter_pop_size}.pt")
            else:
                policy_actor = self.counter_trainer.policy.actor
                torch.save(policy_actor.state_dict(), str(self.save_dir) + f"/counter/actor_{self.counter_pop_size}.pt")
                policy_critic = self.counter_trainer.policy.critic
                torch.save(policy_critic.state_dict(), str(self.save_dir) + f"/counter/critic_{self.counter_pop_size}.pt")
            self.counter_pop_size += 1
        else:
            raise NotImplementedError

    def update_env_info(self, name):
        if name == "main":
            env_info = self.main_env_infos
        elif name == "counter":
            env_info = self.counter_env_infos
        else:
            raise NotImplementedError

        main_idx = np.array(env_info["main_prob"])
        env_info["win_rate_vs_main"] = np.array(env_info["left_win_rate"])[main_idx]
        env_info["draw_rate_vs_main"] = np.array(env_info["draw_rate"])[main_idx]
        env_info["lose_rate_vs_main"] = np.array(env_info["right_win_rate"])[main_idx]

        counter_idx = np.array(env_info["counter_prob"])
        env_info["win_rate_vs_counter"] = np.array(env_info["left_win_rate"])[counter_idx]
        env_info["draw_rate_vs_counter"] = np.array(env_info["draw_rate"])[counter_idx]
        env_info["lose_rate_vs_counter"] = np.array(env_info["right_win_rate"])[counter_idx]

        if (name == "main" and self.all_args.main_fsp_prob > 0) or (name == "counter" and self.all_args.counter_fxp_prob > 0):
            for i in range(self.main_pop_size):
                idx = np.array(env_info[f"main{i}_prob"])
                env_info[f"win_rate_vs_main{i}"] = np.array(env_info["left_win_rate"])[idx]
                env_info[f"draw_rate_vs_main{i}"] = np.array(env_info["draw_rate"])[idx]
                env_info[f"lose_rate_vs_main{i}"] = np.array(env_info["right_win_rate"])[idx]

        if (name == "main" and self.all_args.main_fxp_prob > 0) or (name == "counter" and self.all_args.counter_fsp_prob > 0):
            for i in range(self.counter_pop_size):
                idx = np.array(env_info[f"counter{i}_prob"])
                env_info[f"win_rate_vs_counter{i}"] = np.array(env_info["left_win_rate"])[idx]
                env_info[f"draw_rate_vs_counter{i}"] = np.array(env_info["draw_rate"])[idx]
                env_info[f"lose_rate_vs_counter{i}"] = np.array(env_info["right_win_rate"])[idx]

        if name == "main":
            self.main_env_infos = env_info
        elif name == "counter":
            self.counter_env_infos = env_info

    @torch.no_grad()
    def collect(self, step, name):
        if name == "main":
            values, actions, action_log_probs, rnn_states, rnn_states_critic = self.main_trainer.policy.get_actions(
                np.concatenate(self.main_buffer.share_obs[step]),
                np.concatenate(self.main_buffer.obs[step]),
                np.concatenate(self.main_buffer.rnn_states[step]),
                np.concatenate(self.main_buffer.rnn_states_critic[step]),
                np.concatenate(self.main_buffer.masks[step])
            )
        elif name == "counter":
            values, actions, action_log_probs, rnn_states, rnn_states_critic = self.counter_trainer.policy.get_actions(
                np.concatenate(self.counter_buffer.share_obs[step]),
                np.concatenate(self.counter_buffer.obs[step]),
                np.concatenate(self.counter_buffer.rnn_states[step]),
                np.concatenate(self.counter_buffer.rnn_states_critic[step]),
                np.concatenate(self.counter_buffer.masks[step])
            )
        else:
            raise NotImplementedError(name)

        values = np.array(np.split(_t2n(values), self.n_rollout_threads))
        actions = np.array(np.split(_t2n(actions), self.n_rollout_threads))
        action_log_probs = np.array(np.split(_t2n(action_log_probs), self.n_rollout_threads))
        rnn_states = np.array(np.split(_t2n(rnn_states), self.n_rollout_threads))
        rnn_states_critic = np.array(np.split(_t2n(rnn_states_critic), self.n_rollout_threads))

        return values, actions, action_log_probs, rnn_states, rnn_states_critic

    def insert(self, data, name):
        obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic = data

        dones_env = np.all(dones, axis=1)
        rnn_states[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
        rnn_states_critic[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

        active_masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        active_masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)
        active_masks[dones_env == True] = np.ones(((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

        if name == "main":
            self.main_buffer.insert(
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
                        self.main_env_infos[k].append(v)
        elif name == "counter":
            self.counter_buffer.insert(
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
                        self.counter_env_infos[k].append(v)
        else:
            raise NotImplementedError(name)
    
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
