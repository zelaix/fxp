import numpy as np


class TeamRPSGame(object):
    def __init__(self):
        self.n_player = 2
        self.n_action = 2

    def get_value(self, p_a):
        return self.batch_get_value(p_a)[0]
    
    def batch_get_value(self, p_a_):
        # p_a [B, 2 * n_player, n_action] or [2 * n_player, n_action]
        if len(p_a_.shape) == 2:
            p_a = np.copy([p_a_])
        else:
            p_a = np.copy(p_a_)
        assert len(p_a.shape) == 3 and p_a.shape[1] == self.n_player * 2 and p_a.shape[2] == self.n_action

        my_rock_p = np.prod(p_a[:, :self.n_player, 0], 1)
        my_scissors_p = np.prod(p_a[:, :self.n_player, 1], 1)
        my_paper_p = np.ones_like(my_rock_p) - my_rock_p - my_scissors_p
        oppo_rock_p = np.prod(p_a[:, self.n_player:, 0], 1)
        oppo_scissors_p = np.prod(p_a[:, self.n_player:, 1], 1)
        oppo_paper_p = np.ones_like(oppo_rock_p) - oppo_rock_p - oppo_scissors_p

        u = my_rock_p * (oppo_scissors_p - oppo_paper_p)
        u += my_paper_p * (oppo_rock_p - oppo_scissors_p)
        u += my_scissors_p * (oppo_paper_p - oppo_rock_p)
        return u


if __name__ == "__main__":
    np.random.seed(0)
    game = TeamRPSGame()
    p_rock = np.array([[1, 0], [1, 0]])
    p_paper = np.array([[1, 0], [0, 1]])
    p_scissors = np.array([[0, 1], [0, 1]])

    print(game.get_value(np.concatenate([p_rock, p_rock])))
    print(game.get_value(np.concatenate([p_rock, p_paper])))
    print(game.get_value(np.concatenate([p_rock, p_scissors])))
    print(game.get_value(np.concatenate([p_paper, p_rock])))
    print(game.get_value(np.concatenate([p_paper, p_paper])))
    print(game.get_value(np.concatenate([p_paper, p_scissors])))
    print(game.get_value(np.concatenate([p_scissors, p_rock])))
    print(game.get_value(np.concatenate([p_scissors, p_paper])))
    print(game.get_value(np.concatenate([p_scissors, p_scissors])))
