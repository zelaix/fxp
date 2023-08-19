from copy import deepcopy
from pettingzoo import magent
import gym
import numpy as np
import torch
import wandb


def _t2n(x):
    return x.detach().cpu().numpy()


class StateObsWrapper(gym.ObservationWrapper):
    """
    obs: 14 for 2 vs 2
        2: one-hot id
        3: self info (position, hp)
        3: teammate info (position, hp)
        2x3: opponent info (position, hp)
    """
    def __init__(self, env):
        super().__init__(env)
        self.__n_half_agents = env.unwrapped.num_agents // 2
        self.__map_size = env.state().shape[0]

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        return self.observation(obs)

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        return self.observation(observation), reward, done, info

    def observation(self, observation):
        # If dead, then position = (0, 0), hp = 0.
        red_states = np.zeros((self.__n_half_agents, 3))
        blue_states = np.zeros((self.__n_half_agents, 3))
        for k in observation.keys():  # Alive.
            team, idx = k.split("_")
            idx = int(idx)
            if team == "red":
                red_states[idx, :2] = observation[k][0, 0, -2:] * self.__map_size  # x, y.
                red_states[idx, -1] = observation[k][6, 6, 2]  # hp.
            elif team == "blue":
                blue_states[idx, :2] = self.__map_size - 1 - self.__map_size * observation[k][0, 0, -2:]  # x, y.
                blue_states[idx, -1] = observation[k][6, 6, 2]  # hp.
            else:
                raise ValueError(f"Unexpected key {k} in observation dict.")

        red_obs = [self.get_obs(red_states, blue_states, idx) for idx in range(self.__n_half_agents)]
        blue_obs = [self.get_obs(blue_states, red_states, idx) for idx in range(self.__n_half_agents)]

        return red_obs + blue_obs

    def get_obs(self, self_states, opponent_states, idx):
        self_states = deepcopy(self_states)
        opponent_states = deepcopy(opponent_states)
        one_hot_id = np.zeros(self.__n_half_agents)
        one_hot_id[idx] = 1
        order = (np.arange(self.__n_half_agents) + idx) % self.__n_half_agents
        self_states = np.concatenate(self_states[order])
        opponent_states[:, :2] = self.__map_size - 1 - opponent_states[:, :2]
        opponent_states = np.concatenate(opponent_states[order])
        return np.concatenate([one_hot_id, self_states, opponent_states])


class MixedRewardWrapper(gym.RewardWrapper):
    """
    In team: cooperate and share reward.
    Between team: competitive and zero-sum.
    Final reward:
        + 0.1: self or teammate attacks an opponent.
        + 10: self of teammate kills an opponent.
        - 0.1: self or teammate is attacked.
        - 10: self or teammate is killed.
    """
    def __init__(self, env):
        super().__init__(env)
        self.__n_half_agents = env.unwrapped.num_agents // 2

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        return observation, self.reward(reward), done, info

    def reward(self, reward):
        red_reward = 0
        blue_reward = 0
        for k, v in reward.items():
            if "red" in k:
                red_reward += v
            elif "blue" in k:
                blue_reward += v
            else:
                raise ValueError(f"Unexpected key {k} in reward dict.")
        return [[red_reward - blue_reward]] * self.__n_half_agents + [[blue_reward - red_reward]] * self.__n_half_agents


class HPRewardWrapper(gym.RewardWrapper):
    """
    In team: cooperate and share reward.
    Between team: competitive and zero-sum.
    Final reward:
        * self_left_hp - oppo_left_hp
    """
    def __init__(self, env):
        super().__init__(env)
        print("using hp reward for test!")
        self.__n_half = env.unwrapped.num_agents // 2
        self.red_prev_hp = self.__n_half
        self.blue_prev_hp = self.__n_half

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        red_hp = np.sum([obs[self.__n_half + 2] for obs in observation[:self.__n_half]])
        blue_hp = np.sum([obs[self.__n_half + 2] for obs in observation[self.__n_half:]])
        self.red_hp_change = red_hp - self.red_prev_hp
        self.blue_hp_change = blue_hp - self.blue_prev_hp
        self.red_prev_hp = red_hp
        self.blue_prev_hp = blue_hp
        # print(f"red hp change: {self.red_hp_change}")
        # print(f"blue hp change: {self.blue_hp_change}")
        return observation, self.reward(reward), done, info

    def reward(self, reward):
        red_reward = self.red_hp_change - self.blue_hp_change
        blue_reward = self.blue_hp_change - self.red_hp_change
        return [[red_reward]] * self.__n_half + [[blue_reward]] * self.__n_half


class RotateActionWrapper(gym.ActionWrapper):
    """
    Move:                 Attack:
            4,
        1,  5,  9,            13, 16, 18,
    0,  2,  6, 10, 12,        14,     19,
        3,  7, 11,            15, 17, 20,
            8,
    """
    def __init__(self, env):
        super().__init__(env)
        self.__all_agents = env.unwrapped.agents
        self.__convert_action = [12 - i for i in range(13)] + [20 - j for j in range(8)]

    def action(self, action):
        magent_action = {}
        for k, a in zip(self.__all_agents, action):
            if a is None:
                continue
            elif "red" in k:
                magent_action[k] = a
            elif "blue" in k:
                magent_action[k] = self.__convert_action[a]
            else:
                raise ValueError(f"Unexpected key {k} in env agents dict.")
        return magent_action


class LeftRightWrapper(gym.Wrapper):

    def __init__(self, env):
        super().__init__(env)
        self.__map_size = env.unwrapped.map_size
        self.__all_agents = env.unwrapped.agents
        self.__n_half_agents = env.unwrapped.num_agents // 2

        self.__step_count = 0
        self.__episode_return = np.zeros((2 * self.__n_half_agents, 1))
        if np.random.randint(2) == 0:
            self.__left_team = "blue"
        else:
            self.__left_team = "red"

    def reset(self, **kwargs):
        self.__step_count = 0
        self.__episode_return[:] = 0
        self.__left_team = "red" if self.__left_team == "blue" else "blue"

        if self.__map_size == 13:
            _ = self.env.reset()
            _ = self.env.step([10, 10, 6, 6])
            obs, _, _, _ = self.env.step([7] * 2 + [5] * 2)
        if self.__map_size == 15:
            _ = self.env.reset()
            _ = self.env.step([10, 10, 10, 6, 6, 6])
            obs, _, _, _ = self.env.step([7] * 2 + [5] * 2)

        if self.__left_team == "blue":
            obs = obs[-self.__n_half_agents:] + obs[:self.__n_half_agents]
        return obs
        
    def step(self, action):
        action = list(action[:, 0])
        if self.__left_team == "blue":
            action = action[-self.__n_half_agents:] + action[:self.__n_half_agents]
        obs, reward, done, info = self.env.step(action)
        done = [done[k] if k in done.keys() else True for k in self.__all_agents]
        if self.__left_team == "blue":
            obs = obs[-self.__n_half_agents:] + obs[:self.__n_half_agents]
            reward = reward[-self.__n_half_agents:] + reward[:self.__n_half_agents]
            done = done[-self.__n_half_agents:] + done[:self.__n_half_agents]

        self.__step_count += 1
        self.__episode_return += np.array(reward)

        if all(done):
            hp_idx = 4 + 3 * np.arange(self.__n_half_agents)
            left_hp = obs[0][hp_idx]
            left_survival = np.sum(left_hp > 0)
            left_final_hp = np.sum(left_hp)
            right_hp = obs[-1][hp_idx]
            right_survival = np.sum(right_hp > 0)
            right_final_hp = np.sum(right_hp)
            info = dict(
                episode_length=self.__step_count,
                left_return=self.__episode_return[0, 0],
                right_return=self.__episode_return[-1, 0],
                left_survival=left_survival,
                left_final_hp=left_final_hp,
                right_survival=right_survival,
                right_final_hp=right_final_hp,
                left_win_rate=left_survival > right_survival,
                draw_rate=left_survival == right_survival,
                right_win_rate=left_survival < right_survival,
            )
        else:
            info = dict(
                episode_length=self.__step_count,
                left_return=self.__episode_return[0, 0],
                right_return=self.__episode_return[-1, 0],
            )
                
        return obs, reward, done, info


class MAgentEnv:

    def __init__(self, args, is_eval=False):
        self.one_side = args.one_side
        self.use_sp = args.use_sp
        self.use_population = args.use_population
        self.use_render = args.use_render
        self.num_agents = args.num_agents
        self.total_agents = self.num_agents * 2 if self.one_side else self.num_agents
        if args.scenario_name == "battle":
            env = magent.battle_v4.parallel_env(
                map_size=args.map_size, 
                max_cycles=args.max_episode_length + 2, 
                minimap_mode=True, 
                step_reward=0, 
                dead_penalty=-5, 
                attack_penalty=0, 
                attack_opponent_reward=0.1,
            )
        else:
            raise NotImplementedError(f"{args.scenario_name} is not supported.")
        env = RotateActionWrapper(MixedRewardWrapper(StateObsWrapper(env)))
        # env = RotateActionWrapper(HPRewardWrapper(StateObsWrapper(env)))
        self.env = LeftRightWrapper(env)

        # make policy
        self.args = args
        if args.use_wandb:
            self.model_dir = str(args.run_dir) + "/" + "/".join(wandb.run.dir.split("/")[-3:])
        else:
            self.model_dir = str(args.run_dir) + "/models"
        if self.one_side:
            if "mappo" in args.algorithm_name:
                if args.use_single_network:
                    from gridworld.algorithms.r_mappo_single.algorithm.rMAPPOPolicy import R_MAPPOPolicy as Policy
                else:
                    from gridworld.algorithms.r_mappo.algorithm.rMAPPOPolicy import R_MAPPOPolicy as Policy
            elif "mappg" in args.algorithm_name:
                if args.use_single_network:
                    from gridworld.algorithms.r_mappg_single.algorithm.rMAPPGPolicy import R_MAPPGPolicy as Policy
                else:
                    from gridworld.algorithms.r_mappg.algorithm.rMAPPGPolicy import R_MAPPGPolicy as Policy
            else:
                raise NotImplementedError

            self.Policy = Policy
            self.recurrent_N = args.recurrent_N
            self.hidden_size = args.hidden_size
            self.sp_prob = args.sp_prob
            if self.use_sp:
                assert self.sp_prob > 0
                self.sp_policy = self.Policy(
                    args, 
                    self.observation_space[0],
                    self.share_observation_space[0],
                    self.action_space[0],
                    device=torch.device("cpu"),
                )

            if self.use_population:
                assert self.sp_prob < 1
                self.num_oppo_policy = args.num_oppo_policy
                if is_eval:
                    self.oppo_sample_probs = np.ones(args.num_oppo_policy) / args.num_oppo_policy
                else:
                    self.oppo_sample_probs = np.array(args.oppo_sample_probs)
                self.oppo_population = []
                for model_dir in args.oppo_model_dir:
                    if model_dir == "random":
                        self.oppo_population.append("random")
                    else:
                        oppo_policy = self.Policy(
                            args, 
                            self.observation_space[0],
                            self.share_observation_space[0],
                            self.action_space[0],
                            device=torch.device("cpu"),
                        )
                        if args.use_single_network:
                            policy_model_state_dict = torch.load(str(model_dir) + "/model.pt", map_location=torch.device("cpu"))
                            oppo_policy.model.load_state_dict(policy_model_state_dict)
                        else:
                            policy_actor_state_dict = torch.load(str(model_dir) + "/actor.pt", map_location=torch.device("cpu"))
                            oppo_policy.actor.load_state_dict(policy_actor_state_dict)
                            policy_critic_state_dict = torch.load(str(model_dir) + "/critic.pt", map_location=torch.device("cpu"))
                            oppo_policy.critic.load_state_dict(policy_critic_state_dict)
                        self.oppo_population.append(oppo_policy)

    @property
    def action_space(self):
        return [gym.spaces.Discrete(21)] * self.num_agents

    @property
    def observation_space(self):
        return [[self.total_agents // 2 + 3 * self.total_agents]] * self.num_agents

    @property
    def share_observation_space(self):
        return [[self.total_agents // 2 + 3 * self.total_agents]] * self.num_agents

    def reset(self):
        obs = self.env.reset()

        if self.one_side:
            self.self_play = (np.random.random() <= self.sp_prob)
            if self.self_play:
                self.oppo_idx = -1
                oppo_obs = np.array(obs[self.num_agents:])
                self.oppo_rnn_state = np.zeros((self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
                oppo_mask = np.ones((self.num_agents, 1), dtype=np.float32)
                oppo_action, oppo_rnn_state = self.sp_policy.act(
                    oppo_obs,
                    self.oppo_rnn_state,
                    oppo_mask,
                    deterministic=False,
                )
                self.oppo_action = _t2n(oppo_action)
                self.oppo_rnn_state = _t2n(oppo_rnn_state)
            else:  # vs. population
                self.oppo_idx = np.random.choice(self.num_oppo_policy, p=self.oppo_sample_probs)
                if self.oppo_population[self.oppo_idx] == "random":
                    self.oppo_action = np.random.randint(21, size=(self.num_agents, 1))
                else:
                    oppo_obs = np.array(obs[self.num_agents:])
                    self.oppo_rnn_state = np.zeros((self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
                    oppo_mask = np.ones((self.num_agents, 1), dtype=np.float32)
                    oppo_action, oppo_rnn_state = self.oppo_population[self.oppo_idx].act(
                        oppo_obs,
                        self.oppo_rnn_state,
                        oppo_mask,
                        deterministic=False,
                    )
                    self.oppo_action = _t2n(oppo_action)
                    self.oppo_rnn_state = _t2n(oppo_rnn_state)

            obs = obs[:self.num_agents]

        if self.use_render:
            self.frames = [self.render()]

        return obs
    
    def update(self, update_sp=False, update_population=False, oppo_sample_probs=None):
        if update_sp:
            if self.args.use_single_network:
                policy_model_state_dict = torch.load(self.model_dir + f"/model.pt", map_location=torch.device("cpu"))
                self.sp_policy.model.load_state_dict(policy_model_state_dict)
            else:
                policy_actor_state_dict = torch.load(self.model_dir + f"/actor.pt", map_location=torch.device("cpu"))
                self.sp_policy.actor.load_state_dict(policy_actor_state_dict)
                policy_critic_state_dict = torch.load(self.model_dir + f"/critic.pt", map_location=torch.device("cpu"))
                self.sp_policy.critic.load_state_dict(policy_critic_state_dict)

        if update_population:
            new_policy = self.Policy(
                self.args, 
                self.observation_space[0],
                self.share_observation_space[0],
                self.action_space[0],
                device=torch.device("cpu"),
            )
            if self.args.use_single_network:
                policy_model_state_dict = torch.load(self.model_dir + f"/model_{self.num_oppo_policy}.pt", map_location=torch.device("cpu"))
                new_policy.model.load_state_dict(policy_model_state_dict)
            else:
                policy_actor_state_dict = torch.load(self.model_dir + f"/actor_{self.num_oppo_policy}.pt", map_location=torch.device("cpu"))
                new_policy.actor.load_state_dict(policy_actor_state_dict)
                policy_critic_state_dict = torch.load(self.model_dir + f"/critic_{self.num_oppo_policy}.pt", map_location=torch.device("cpu"))
                new_policy.critic.load_state_dict(policy_critic_state_dict)
            self.oppo_population.append(new_policy)
            self.num_oppo_policy += 1

        if oppo_sample_probs is not None:
            self.oppo_sample_probs = np.array(oppo_sample_probs)

        return True

    def step(self, action):
        if self.one_side:
            action = np.concatenate([action, self.oppo_action])
        obs, reward, done, info = self.env.step(action)
        if self.one_side:
            if all(done):
                if self.use_sp:
                    info["sp_prob"] = self.self_play
                if self.use_population:
                    for i in range(self.num_oppo_policy):
                        info[f"oppo_prob_{i}"] = (i == self.oppo_idx)
                    info["oppo_idx"] = self.oppo_idx
            if self.self_play:
                oppo_obs = np.array(obs[self.num_agents:])
                oppo_mask = np.ones((self.num_agents, 1), dtype=np.float32)
                oppo_action, oppo_rnn_state = self.sp_policy.act(
                    oppo_obs,
                    self.oppo_rnn_state,
                    oppo_mask,
                    deterministic=False,
                )
                self.oppo_action = _t2n(oppo_action)
                self.oppo_rnn_state = _t2n(oppo_rnn_state)
            else:  # vs. population.
                if self.oppo_population[self.oppo_idx] == "random":
                    self.oppo_action = np.random.randint(21, size=(self.num_agents, 1))
                else:
                    oppo_obs = np.array(obs[self.num_agents:])
                    oppo_mask = np.ones((self.num_agents, 1), dtype=np.float32)
                    oppo_action, oppo_rnn_state = self.oppo_population[self.oppo_idx].act(
                        oppo_obs,
                        self.oppo_rnn_state,
                        oppo_mask,
                        deterministic=False,
                    )
                    self.oppo_action = _t2n(oppo_action)
                    self.oppo_rnn_state = _t2n(oppo_rnn_state)

            obs = obs[:self.num_agents]
            reward = reward[:self.num_agents]
            done = done[:self.num_agents]

        if self.use_render:
            self.frames.append(self.render())
            if all(done):
                info["frames"] = deepcopy(self.frames)

        return obs, reward, done, info

    def seed(self, seed=None):
        seed = seed or 0
        self.env.seed(seed)
        np.random.seed(seed)

    def close(self):
        self.env.close()

    def render(self, mode="rgb_array"):
        assert mode == "rgb_array", f"mode should be rgb_array, but got {mode}"
        # bug in magent, render 3 times to get the right image
        self.env.render(mode="rgb_array")
        self.env.render(mode="rgb_array")
        return self.env.render(mode="rgb_array")


class MAgentXPEnv:

    def __init__(self, 
                 args, 
                 main_prob,
                 counter_prob,
                 main_pop_prob,
                 counter_pop_prob,
                 is_eval=False):
        self.args = args        
        self.use_render = args.use_render
        self.num_agents = args.num_agents
        self.total_agents = self.num_agents * 2
        if args.scenario_name == "battle":
            env = magent.battle_v4.parallel_env(
                map_size=args.map_size, 
                max_cycles=args.max_episode_length + 2, 
                minimap_mode=True, 
                step_reward=0, 
                dead_penalty=-5, 
                attack_penalty=0, 
                attack_opponent_reward=0.1,
            )
        else:
            raise NotImplementedError(f"{args.scenario_name} is not supported.")
        env = RotateActionWrapper(MixedRewardWrapper(StateObsWrapper(env)))
        self.env = LeftRightWrapper(env)

        sum_prob = np.float32(main_prob + counter_prob + main_pop_prob + counter_pop_prob)
        main_prob = np.float32(main_prob) / sum_prob
        counter_prob = np.float32(counter_prob) / sum_prob
        main_pop_prob = np.float32(main_pop_prob) / sum_prob
        counter_pop_prob = np.float32(counter_pop_prob) / sum_prob
        # print(main_prob, counter_prob, main_pop_prob, counter_pop_prob)
        # print(main_prob + counter_prob + main_pop_prob + counter_pop_prob)
        assert main_prob + counter_prob + main_pop_prob + counter_pop_prob == 1
        self.main_prob = main_prob
        self.counter_prob = counter_prob
        self.main_pop_prob = main_pop_prob
        self.counter_pop_prob = counter_pop_prob
        # make policy
        if args.use_wandb:
            self.main_model_dir = str(args.run_dir) + "/" + "/".join(wandb.run.dir.split("/")[-3:]) + "/main"
            self.counter_model_dir = str(args.run_dir) + "/" + "/".join(wandb.run.dir.split("/")[-3:]) + "/counter"
        else:
            self.main_model_dir = str(args.run_dir) + "/models/main"
            self.counter_model_dir = str(args.run_dir) + "/models/counter"
        if "mappo" in args.algorithm_name:
            if args.use_single_network:
                from gridworld.algorithms.r_mappo_single.algorithm.rMAPPOPolicy import R_MAPPOPolicy as Policy
            else:
                from gridworld.algorithms.r_mappo.algorithm.rMAPPOPolicy import R_MAPPOPolicy as Policy
        elif "mappg" in args.algorithm_name:
            if args.use_single_network:
                from gridworld.algorithms.r_mappg_single.algorithm.rMAPPGPolicy import R_MAPPGPolicy as Policy
            else:
                from gridworld.algorithms.r_mappg.algorithm.rMAPPGPolicy import R_MAPPGPolicy as Policy
        else:
            raise NotImplementedError

        self.Policy = Policy
        self.recurrent_N = args.recurrent_N
        self.hidden_size = args.hidden_size
        if self.main_prob > 0:
            self.main_policy = self.Policy(
                args, 
                self.observation_space[0],
                self.share_observation_space[0],
                self.action_space[0],
                device=torch.device("cpu"),
            )
        if self.counter_prob > 0:
            self.counter_policy = self.Policy(
                args, 
                self.observation_space[0],
                self.share_observation_space[0],
                self.action_space[0],
                device=torch.device("cpu"),
            )
        if self.main_pop_prob > 0:
            self.main_pop_size = args.main_pop_size
            if is_eval:
                self.main_pop_sample_probs = np.ones(args.main_pop_size) / args.main_pop_size
            else:
                self.main_pop_sample_probs = np.array(args.main_pop_sample_probs)
            self.main_population = []
        if self.counter_pop_prob > 0:
            self.counter_pop_size = args.counter_pop_size
            if is_eval:
                self.counter_pop_sample_probs = np.ones(args.counter_pop_size) / args.counter_pop_size
            else:
                self.counter_pop_sample_probs = np.array(args.counter_pop_sample_probs)
            self.counter_population = []

    @property
    def action_space(self):
        return [gym.spaces.Discrete(21)] * self.num_agents

    @property
    def observation_space(self):
        return [[self.num_agents + 3 * self.total_agents]] * self.num_agents

    @property
    def share_observation_space(self):
        return [[self.num_agents + 3 * self.total_agents]] * self.num_agents

    def reset(self):
        self.select_opponent()

        obs = self.env.reset()
        self_obs = obs[:self.num_agents]
        self.oppo_obs = np.array(obs[self.num_agents:])
        self.oppo_rnn_state = np.zeros((self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)

        if self.use_render:
            self.frames = [self.render()]

        return self_obs
    
    def update(self, 
               update_main=False, 
               update_counter=False, 
               update_main_pop=False, 
               main_pop_sample_probs=None,
               update_counter_pop=False,
               counter_pop_sample_probs=None):
        if update_main and self.main_prob > 0:
            if self.args.use_single_network:
                policy_model_state_dict = torch.load(self.main_model_dir + f"/model.pt", map_location=torch.device("cpu"))
                self.main_policy.model.load_state_dict(policy_model_state_dict)
            else:
                policy_actor_state_dict = torch.load(self.main_model_dir + f"/actor.pt", map_location=torch.device("cpu"))
                self.main_policy.actor.load_state_dict(policy_actor_state_dict)
                policy_critic_state_dict = torch.load(self.main_model_dir + f"/critic.pt", map_location=torch.device("cpu"))
                self.main_policy.critic.load_state_dict(policy_critic_state_dict)

        if update_counter and self.counter_prob > 0:
            if self.args.use_single_network:
                policy_model_state_dict = torch.load(self.counter_model_dir + f"/model.pt", map_location=torch.device("cpu"))
                self.counter_policy.model.load_state_dict(policy_model_state_dict)
            else:
                policy_actor_state_dict = torch.load(self.counter_model_dir + f"/actor.pt", map_location=torch.device("cpu"))
                self.counter_policy.actor.load_state_dict(policy_actor_state_dict)
                policy_critic_state_dict = torch.load(self.counter_model_dir + f"/critic.pt", map_location=torch.device("cpu"))
                self.counter_policy.critic.load_state_dict(policy_critic_state_dict)

        if update_main_pop and self.main_pop_prob > 0:
            new_policy = self.Policy(
                self.args, 
                self.observation_space[0],
                self.share_observation_space[0],
                self.action_space[0],
                device=torch.device("cpu"),
            )
            if self.args.use_single_network:
                policy_model_state_dict = torch.load(self.main_model_dir + f"/model_{self.main_pop_size}.pt", map_location=torch.device("cpu"))
                new_policy.model.load_state_dict(policy_model_state_dict)
            else:
                policy_actor_state_dict = torch.load(self.main_model_dir + f"/actor_{self.main_pop_size}.pt", map_location=torch.device("cpu"))
                new_policy.actor.load_state_dict(policy_actor_state_dict)
                policy_critic_state_dict = torch.load(self.main_model_dir + f"/critic_{self.main_pop_size}.pt", map_location=torch.device("cpu"))
                new_policy.critic.load_state_dict(policy_critic_state_dict)
            self.main_population.append(new_policy)
            self.main_pop_size += 1

        if update_counter_pop and self.counter_pop_prob > 0:
            new_policy = self.Policy(
                self.args, 
                self.observation_space[0],
                self.share_observation_space[0],
                self.action_space[0],
                device=torch.device("cpu"),
            )
            if self.args.use_single_network:
                policy_model_state_dict = torch.load(self.counter_model_dir + f"/model_{self.counter_pop_size}.pt", map_location=torch.device("cpu"))
                new_policy.model.load_state_dict(policy_model_state_dict)
            else:
                policy_actor_state_dict = torch.load(self.counter_model_dir + f"/actor_{self.counter_pop_size}.pt", map_location=torch.device("cpu"))
                new_policy.actor.load_state_dict(policy_actor_state_dict)
                policy_critic_state_dict = torch.load(self.counter_model_dir + f"/critic_{self.counter_pop_size}.pt", map_location=torch.device("cpu"))
                new_policy.critic.load_state_dict(policy_critic_state_dict)
            self.counter_population.append(new_policy)
            self.counter_pop_size += 1

        if main_pop_sample_probs is not None:
            self.main_pop_sample_probs = np.array(main_pop_sample_probs)

        if counter_pop_sample_probs is not None:
            self.counter_pop_sample_probs = np.array(counter_pop_sample_probs)

        return True

    def step(self, action):
        oppo_action = self.opponent_act()
        action = np.concatenate([action, oppo_action])

        obs, reward, done, info = self.env.step(action)
        self_obs = obs[:self.num_agents]
        self_reward = reward[:self.num_agents]
        self_done = done[:self.num_agents]
        self.oppo_obs = np.array(obs[self.num_agents:])

        if all(done):
            info["main_prob"] = (self.main_idx == -1)
            info["counter_prob"] = (self.counter_idx == -1)
            if self.main_pop_prob > 0:
                for i in range(self.main_pop_size):
                    info[f"main{i}_prob"] = (i == self.main_idx)
                info["main_idx"] = self.main_idx
            if self.counter_pop_prob > 0:
                for i in range(self.counter_pop_size):
                    info[f"counter{i}_prob"] = (i == self.counter_idx)
                info["counter_idx"] = self.counter_idx

        if self.use_render:
            self.frames.append(self.render())
            if all(done):
                info["frames"] = deepcopy(self.frames)

        return self_obs, self_reward, self_done, info

    def select_opponent(self):
        prob = np.random.random()
        if prob < self.main_prob:
            self.main_idx = -1
            self.counter_idx = -2
            self.oppo_policy = self.main_policy
        elif prob < self.main_prob + self.counter_prob:
            self.main_idx = -2
            self.counter_idx = -1
            self.oppo_policy = self.counter_policy
        elif prob < self.main_prob + self.counter_prob + self.main_pop_prob:
            self.main_idx = np.random.choice(self.main_pop_size, p=self.main_pop_sample_probs)
            self.counter_idx = -2
            self.oppo_policy = self.main_population[self.main_idx]
        else:
            self.main_idx = -2
            self.counter_idx = np.random.choice(self.counter_pop_size, p=self.counter_pop_sample_probs)
            self.oppo_policy = self.counter_population[self.counter_idx]

    @torch.no_grad()
    def opponent_act(self):
        oppo_action, oppo_rnn_state = self.oppo_policy.act(
            self.oppo_obs,
            self.oppo_rnn_state,
            np.ones((self.num_agents, 1), dtype=np.float32),
            deterministic=False,
        )
        self.oppo_rnn_state = _t2n(oppo_rnn_state)
        return _t2n(oppo_action)

    def seed(self, seed=None):
        seed = seed or 0
        self.env.seed(seed)
        np.random.seed(seed)

    def close(self):
        self.env.close()

    def render(self, mode="rgb_array"):
        assert mode == "rgb_array", f"mode should be rgb_array, but got {mode}"
        # bug in magent, render 3 times to get the right image
        self.env.render(mode="rgb_array")
        self.env.render(mode="rgb_array")
        return self.env.render(mode="rgb_array")
