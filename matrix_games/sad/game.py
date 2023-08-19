from gym.spaces import Discrete
import numpy as np


class SADGame(object):
    def __init__(self, team_players=3, num_actions=5, episode_length=1, attack_cost=1, attack_force=3):
        # 1...A for gaining actions, 0 for none, A+1 for attack, A+2 for defend
        self.n_player = team_players
        self.n_action = num_actions
        self.L = episode_length
        self.attack_cost = attack_cost

        self.attack_action = self.n_action + 1
        self.defend_action = self.n_action + 2
        self.attack_force = attack_force

        self.reward = np.zeros(self.n_action + 3)
        for i in range(1, 1 + self.n_action):
            self.reward[i] = 1 + i ** 2 / self.n_action ** 2

        self.observation_space = [self.n_player for _ in range(2 * self.n_player)]
        self.action_space = [Discrete(self.n_action + 3) for _ in range(self.n_player)]
    
    def reward(self, actions):
        # actions [2 * N] \in {0, 1, 2, ..., n_action + 2}
        my_action, oppo_action = actions[:self.n_player], actions[self.n_player:]
        def gain(actions):
            A = []
            for a in actions:
                if a <= self.n_action:
                    A.append(a)
            if len(A) == 0:
                return 0
            base = 0 if (np.max(A) - np.min(A) > 1) else np.min(A)
            R = 0
            for a in A:
                if a <= base + 1:
                    R += self.reward[a]
            return R
        my_attacked = np.sum(oppo_action == self.attack_action) >= self.attack_force and np.all(my_action != self.defend_action)
        oppo_attacked = np.sum(my_action == self.attack_action) >= self.attack_force and np.all(oppo_action != self.defend_action)
        r_my = gain(my_action) * (1 - my_attacked)
        r_oppo = gain(oppo_action) * (1 - oppo_attacked)
        cost_my = self.attack_cost * np.sum(actions[:self.n_player] == self.attack_action)
        cost_oppo = self.attack_cost * np.sum(actions[self.n_player:] == self.attack_action)
        return r_my - cost_my - r_oppo + cost_oppo

    def get_attack_prob(self, p_):
        # p: [B, N_player], return prob_of_attack
        if len(p_.shape) == 1:
            p = np.copy([p_])
        else:
            p = np.copy(p_)
        dp = np.zeros((p.shape[0], self.n_player + 1, self.attack_force))
        dp[:, 0, 0] = np.ones(p.shape[0])
        for i in range(self.n_player):
            for j in range(self.attack_force):
                dp[:, i + 1, j] += dp[:, i, j] * (1 - p[:, i])
                if j > 0:
                    dp[:, i + 1, j] += dp[:, i, j - 1] * p[:, i]
        return 1 - np.sum(dp[:, self.n_player, :], 1)

    def get_value(self, p_a):
        # p_a [2 * N, n_action + 3]
        my_attacked = self.get_attack_prob(p_a[self.n_player:, self.attack_action])[0]
        oppo_attacked = self.get_attack_prob(p_a[:self.n_player, self.attack_action])[0]
        
        def get_oneside_reward(p_a, p_attack):
            # f[N][min][defend][cond1][cond2], g
            # f = p, g = f * \sum r
            f = np.zeros((self.n_player + 1, self.n_action + 1, 2, 2, 2))
            for j in range(1 + self.n_action):
                f[0][j][0][0][0] = 1
            g = np.zeros((self.n_player + 1, self.n_action + 1, 2, 2, 2))
            for i in range(self.n_player):
                for j in range(1 + self.n_action):
                    for x in range(2):
                        for y in range(2):
                            for z in range(2):
                                for a in range(self.n_action + 3):
                                    nx, ny, nz = x, y, z
                                    if j > 0:
                                        if a <= self.n_action and (a < j or a > j + 1): continue
                                        if a == j: ny = nz = 1
                                    elif j == 0:
                                        if a == 1: ny = 1
                                        if a >= 3 and a <= self.n_action or a == 0: nz = 1
                                    if a == self.defend_action: nx = 1
                                    f[i + 1][j][nx][ny][nz] += f[i][j][x][y][z] * p_a[i][a]
                                    r = 0
                                    if a <= self.n_action and a >= j and a <= j + 1:
                                        r = self.reward[a]
                                    g[i + 1][j][nx][ny][nz] += (g[i][j][x][y][z] + f[i][j][x][y][z] * r) * p_a[i][a]
            R = 0
            for j in range(1 + self.n_action):
                R += g[self.n_player][j][0][1][1] * (1 - p_attack) + g[self.n_player][j][1][1][1]
            return R
        value = get_oneside_reward(p_a[:self.n_player], my_attacked)
        value -= get_oneside_reward(p_a[self.n_player:], oppo_attacked)
        value -= self.attack_cost * p_a[:self.n_player, self.attack_action].sum()
        value += self.attack_cost * p_a[self.n_player:, self.attack_action].sum()
        return value

    def batch_get_value(self, p_a_):
        # p_a [B, 2 * N, n_action + 3] or [2 * N, n_action + 3]
        if len(p_a_.shape) == 2:
            p_a = np.copy([p_a_])
        else:
            p_a = np.copy(p_a_)
        assert len(p_a.shape) == 3 and p_a.shape[1] == self.n_player * 2 and p_a.shape[2] == self.n_action + 3
        
        my_attacked = self.get_attack_prob(p_a[:, self.n_player:, self.attack_action]) # B
        oppo_attacked = self.get_attack_prob(p_a[:, :self.n_player, self.attack_action]) # B
        
        def get_oneside_reward(p_a, p_attack):
            # f[B][N][min][defend][cond1][cond2], g
            # f = p, g = f * \sum r
            f = np.zeros((p_a.shape[0], self.n_player + 1, self.n_action + 1, 2, 2, 2))
            for j in range(1 + self.n_action):
                f[:, 0, j, 0, 0, 0] = 1
            g = np.zeros_like(f)
            for i in range(self.n_player):
                for j in range(1 + self.n_action):
                    for x in range(2):
                        for y in range(2):
                            for z in range(2):
                                for a in range(self.n_action + 3):
                                    nx, ny, nz = x, y, z
                                    if j > 0:
                                        if a <= self.n_action and (a < j or a > j + 1): continue
                                        if a == j: ny = nz = 1
                                    elif j == 0:
                                        if a == 1: ny = 1
                                        if a >= 3 and a <= self.n_action or a == 0: nz = 1
                                    if a == self.defend_action: nx = 1
                                    f[:, i + 1, j, nx, ny, nz] += f[:, i, j, x ,y ,z] * p_a[:, i, a]
                                    r = 0
                                    if a <= self.n_action and a >= j and a <= j + 1:
                                        r = self.reward[a]
                                    g[:, i + 1, j, nx, ny, nz] += (g[:, i, j, x ,y ,z] + f[:, i, j, x ,y ,z] * r) * p_a[:, i, a]
            R = np.zeros((p_a.shape[0],))
            for j in range(1 + self.n_action):
                R += g[:, self.n_player, j, 0, 1, 1] * (1 - p_attack) + g[:, self.n_player, j, 1, 1, 1]
            return R # [B]
        
        value = get_oneside_reward(p_a[:, :self.n_player], my_attacked)
        value -= get_oneside_reward(p_a[:, self.n_player:], oppo_attacked)
        value -= self.attack_cost * p_a[:, :self.n_player, self.attack_action].sum(-1)
        value += self.attack_cost * p_a[:, self.n_player:, self.attack_action].sum(-1)
        return value


if __name__ == "__main__":
    n_player = 6
    n_action = 7

    np.random.seed(0)
    game = SADGame(n_player, n_action, attack_force=2)
    list_p = []
    for _ in range(10):
        p = np.random.rand(n_player * 2, n_action + 3)
        p /= p.sum(1, keepdims=True)
        print(game.get_value(p))
        list_p.append(p)
    print(game.batch_get_value(np.stack(list_p)))
