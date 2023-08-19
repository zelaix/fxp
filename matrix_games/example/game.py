from gym.spaces import Discrete
import numpy as np


class ExampleGame(object):
    def __init__(self, team_players=3, num_actions=2, episode_length=1, C=0.1, eps=0.01):
        self.n_player = team_players
        self.n_action = num_actions
        self.L = episode_length
        assert self.n_action == 2
        assert self.L == 1
        self.C, self.eps = C, eps

        self.observation_space = [self.n_player for _ in range(2 * self.n_player)]
        self.action_space = [Discrete(self.n_action) for _ in range(self.n_player)]

    def get_value(self, p_a):
        return self.batch_get_value(p_a)[0]
    
    def batch_get_value(self, p_a_):
        # p_a [B, 2 * N, n_action] or [2 * N, n_action]
        if len(p_a_.shape) == 2:
            p_a = np.copy([p_a_])
        else:
            p_a = np.copy(p_a_)
        assert len(p_a.shape) == 3 and p_a.shape[1] == self.n_player * 2 and p_a.shape[2] == self.n_action
        
        u = p_a[:, :self.n_player, 1].sum(1) - p_a[:, self.n_player:, 1].sum(1)
        my_p0, oppo_p0 = np.prod(p_a[:, :self.n_player, 0], 1), np.prod(p_a[:, self.n_player:, 0], 1)
        my_p1, oppo_p1 = np.prod(p_a[:, :self.n_player, 1], 1), np.prod(p_a[:, self.n_player:, 1], 1)
        u += my_p0 * ((1 + self.eps) * p_a[:, self.n_player:, 1].sum(1))
        u -= oppo_p0 * ((1 + self.eps) * p_a[:, :self.n_player, 1].sum(1))
        u += my_p0 * oppo_p1 * (self.C - self.n_player * self.eps)
        u -= oppo_p0 * my_p1 * (self.C - self.n_player * self.eps)
        return u


if __name__ == "__main__":
    np.random.seed(0)
    N, A = 5, 2
    game = ExampleGame(N, A)
    p_0 = np.eye(A)[np.repeat(0, N)]
    p_1 = np.eye(A)[np.repeat(1, N)]
    p_rand = np.eye(A)[np.random.randint(0, A, N)]

    print(game.get_value(np.concatenate([p_0, p_1])))
    print(game.get_value(np.concatenate([p_0, p_rand])))
    print(game.get_value(np.concatenate([p_1, p_rand])))
