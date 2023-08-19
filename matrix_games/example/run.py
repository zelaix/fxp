import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import os
import torch

from matrix_games.example.game import ExampleGame


pi0_gen_mode = "uniform"

def softmax(x, d=-1):
    y = np.exp(np.clip(x, -100, 100))
    return y / y.sum(d, keepdims=True)

def soft_BR(q, alpha=0, noise_scale=0):
    # def q: [**shape, A]
    noise = noise_scale * np.abs(q).mean(-1, keepdims=True) * np.random.uniform(-1, 1, q.shape)
    if alpha < 1e-3:
        return np.eye(q.shape[-1])[np.argmax((q + noise), -1)]
    else:
        Q = q + noise
        Q -= Q.max(-1, keepdims=True)
        return softmax(Q / alpha)

def get_pi0(N, A, mode="rand"):
    if mode == "rand":
        pi_0 = np.random.rand(N, A)
    elif mode == "uniform":
        pi_0 = np.ones((N, A))
    else:
        raise NotImplementedError
    pi_0 /= pi_0.sum(-1, keepdims=1)
    return pi_0

def isclose(pi, mu, eps=1e-4, return_value=False):
    if return_value:
        value = np.max(((pi - mu) ** 2).sum(-1))
        return value < eps, value
    return np.max(((pi - mu) ** 2).sum(-1)) < eps

class EvalLogger(object):
    def __init__(self, game):
        self.game = game
        self.N = game.n_player
        self.A = game.n_action
        self.T, self.q = 0, []
        self.log_mode = 1 # p0_expl, p1_expl

        self.end_point = []
    
    def eval(self, pi):
        N, A = self.N, self.A
        p0_pi = np.eye(self.A)[np.repeat(0, N)]
        p1_pi = np.eye(self.A)[np.repeat(1, N)]

        p0_expl = game.get_value(np.concatenate([p0_pi, pi], 0))
        p1_expl = game.get_value(np.concatenate([p1_pi, pi], 0))
        return p0_expl, p1_expl
    
    def log(self, pi):
        p0_expl, p1_expl = self.eval(pi)
        if self.log_mode == 1:
            self.q.append(np.array([max(0, p0_expl)]))
        elif self.log_mode == 3:
            self.q.append(np.array([max(0, p0_expl), max(0, p1_expl)]))
        else:
            raise NotImplementedError
        self.T += 1
    
    def batch_eval(self, pi_list):
        N, A = self.N, self.A
        p0_pi = np.eye(self.A)[np.repeat(0, N)]
        p1_pi = np.eye(self.A)[np.repeat(1, N)]

        expl_list = []
        for expl_policy in [p0_pi, p1_pi]:
            eval_list = []
            for pi in pi_list:
                eval_list.append(np.concatenate([expl_policy, pi], 0))
            eval_res = game.batch_get_value(np.stack(eval_list))
            expl_list.append(eval_res)
        return expl_list[0], expl_list[1]
    
    def log_meta(self, meta_pi):
        pi_list, coeff = meta_pi
        p0_expl, p1_expl = self.batch_eval(pi_list)
        p0_expl = (p0_expl * coeff).sum()
        p1_expl = (p1_expl * coeff).sum()
        if self.log_mode == 1:
            self.q.append(np.array([max(0, p0_expl)]))
        elif self.log_mode == 2:
            self.q.append(np.array([max(0, p0_expl), max(0, p1_expl)]))
        else:
            raise NotImplementedError
        self.T += 1
    
    def terminate(self):
        self.end_point.append(self.T)
    
    def paint(self, ax):
        if self.T == 0:
            return
        data = np.stack(self.q, 1)
        color = ["red", "green"]
        label = ["p0", "p1"]
        for i in range(1, self.log_mode):
            data[i] += data[i - 1]
        for i in range(0, self.log_mode):
            last = np.zeros_like(data[i]) if i == 0 else data[i - 1]
            bg = 0
            for ed in self.end_point:
                ax.plot(np.arange(self.T)[bg:ed], data[i][bg:ed], color=color[i])
                ax.fill_between(np.arange(self.T)[bg:ed], last[bg:ed], data[i][bg:ed], color=color[i], alpha=0.3)
                bg = ed
    
    def save(self, name):
        data = {
            "T": self.T,
            "log_mode": self.log_mode,
            "end_point": self.end_point,
            "q": self.q,
        }
        torch.save(data, f"models/{name}.pt")

def SP(pi, game, n_iters=100, lr=1e-1, alpha=0, noise=0, logger=None):
    # def pi: [N, A]
    N, A = pi.shape[0], pi.shape[1]
    converge = 0
    for i in range(n_iters):
        p = np.repeat(pi, 2, 0)
        p_a_list = []
        for x in range(N):
            for a in range(A):
                p_a = np.copy(p)
                p_a[x] = np.eye(A)[a]
                p_a_list.append(p_a)
        Q = game.batch_get_value(np.stack(p_a_list)).reshape(N, A)
        sol = soft_BR(Q, alpha, noise / (i + 2))
        batch_lr = lr * (np.ones(N) + np.eye(N)[i % N]).reshape(N, 1)
        pi = (1 - batch_lr) * pi + batch_lr * sol
        if logger:
            logger.log(pi)
        if isclose(pi, sol):
            converge += 1
            if converge >= max(5, 0.1 * i):
                break
        else:
            converge = 0
    if logger:
        logger.terminate()
    return pi

def FP(pi, game, n_iters=100, lr=1e-1, alpha=0, noise=0, logger=None):
    # def pi: [N, A]
    N, A = pi.shape[0], pi.shape[1]
    pi_list = []
    converge = 0
    for i in range(n_iters):
        p_a_list = []
        sample_list = np.random.choice(i, min(i, 20), replace=False)
        for j in range(len(sample_list) + 1):
            if j < len(sample_list):
                p = np.concatenate([pi, pi_list[j]], 0)
            else:
                p = np.concatenate([pi, pi], 0)
            for x in range(N):
                for a in range(A):
                    p_a = np.copy(p)
                    p_a[x] = np.eye(A)[a]
                    p_a_list.append(p_a)
        Q_list = game.batch_get_value(np.stack(p_a_list)).reshape(len(sample_list) + 1, N, A)
        Q = 1 / (i + 1) * Q_list[-1]
        if i > 0:
            Q += i / (i + 1) * Q_list[:-1].mean(0)
        sol = soft_BR(Q, alpha, noise / (i + 2))
        batch_lr = lr * (np.ones(N) + np.eye(N)[i % N]).reshape(N, 1)
        pi = (1 - batch_lr) * pi + batch_lr * sol

        pi_list.append(pi)
        if logger:
            logger.log_meta(meta_pi=(pi_list, np.ones(i + 1) / (i + 1)))
        avg_pi = np.stack(pi_list).mean(0)
        if isclose(avg_pi, sol):
            converge += 1
            if converge >= max(5, 0.1 * i):
                break
        else:
            converge = 0
    if logger:
        logger.terminate()
    return avg_pi

def FSP(pi, game, eta=0.3, n_iters=100, lr=1e-1, alpha=0, noise=0, logger=None):
    # def pi: [N, A]
    N, A = pi.shape[0], pi.shape[1]
    pi_list = []
    for i in range(n_iters):
        p_a_list = []
        sample_list = np.random.choice(i, min(i, 20), replace=False)
        for j in range(len(sample_list) + 1):
            if j < len(sample_list):
                p = np.concatenate([pi, pi_list[j]], 0)
            else:
                p = np.concatenate([pi, pi], 0)
            for x in range(N):
                for a in range(A):
                    p_a = np.copy(p)
                    p_a[x] = np.eye(A)[a]
                    p_a_list.append(p_a)
        Q_list = game.batch_get_value(np.stack(p_a_list)).reshape(len(sample_list) + 1, N, A)
        Q = eta * Q_list[-1]
        if i > 0:
            Q += (1 - eta) * np.mean(Q_list[:-1], 0)
        sol = soft_BR(Q, alpha, noise / (i + 2))
        batch_lr = lr * (np.ones(N) + np.eye(N)[i % N]).reshape(N, 1)
        pi = (1 - batch_lr) * pi + batch_lr * sol
        pi_list.append(pi)
        if logger:
            logger.log_meta(meta_pi=(pi_list, np.ones(i + 1) / (i + 1)))
        avg_pi = np.stack(pi_list).mean(0)
        if isclose(avg_pi, sol):
            converge += 1
            if converge >= max(5, 0.1 * i):
                break
        else:
            converge = 0
    if logger:
        logger.terminate()
    return avg_pi

def FoReL(pi, game, n_iters=100, lr=1e-1, alpha=0, noise=0, logger=None):
    # def pi: [N, A]
    N, A = pi.shape[0], pi.shape[1]
    converge = 0
    Q_sum = np.zeros_like(pi)
    for i in range(n_iters):
        p = np.repeat(pi, 2, 0)
        V = game.get_value(p)
        p_a_list = []
        for x in range(N):
            for a in range(A):
                p_a = np.copy(p)
                p_a[x] = np.eye(A)[a]
                p_a_list.append(p_a)
        Q = game.batch_get_value(np.stack(p_a_list)).reshape(N, A)
        Regret = Q - V
        Regret /= Regret.std(-1, keepdims=True)
        Q_sum += lr * Regret / (i + 1) ** 0.5
        Q_sum = np.clip(Q_sum, -4, 4)
        last_pi = np.copy(pi)
        pi = softmax(Q_sum)
        if isclose(pi, last_pi):
            converge += 1
            # if converge >= max(5, 0.1 * i):
            #     break
        if logger:
            logger.log(pi)
    if logger:
        logger.terminate()
    return pi

def Replicator(pi, game, n_iters=100, lr=1e-1, alpha=0, noise=0, logger=None):
    # def pi: [N, A]
    N, A = pi.shape[0], pi.shape[1]
    converge = 0
    for i in range(n_iters):
        p = np.repeat(pi, 2, 0)
        V = game.get_value(p)
        p_a_list = []
        for x in range(N):
            for a in range(A):
                p_a = np.copy(p)
                p_a[x] = np.eye(A)[a]
                p_a_list.append(p_a)
        Q = game.batch_get_value(np.stack(p_a_list)).reshape(N, A)
        Regret = Q - V
        Regret /= np.max(np.abs(Regret)) + 1e-9
        last_pi = np.copy(pi)
        pi += lr * pi * Regret
        pi = (pi + 1e-4) / (pi + 1e-4).sum(-1, keepdims=True)
        if isclose(pi, last_pi):
            converge += 1
            # if converge >= max(50, 0.5 * i):
            #     break
        if logger:
            logger.log(pi)
    if logger:
        logger.terminate()
    return pi

def MWU(pi, game, n_iters=100, lr=1e-1, alpha=0, noise=0, logger=None):
    # def pi: [N, A]
    N, A = pi.shape[0], pi.shape[1]
    converge = 0
    for i in range(n_iters):
        p = np.repeat(pi, 2, 0)
        p_a_list = []
        for x in range(N):
            for a in range(A):
                p_a = np.copy(p)
                p_a[x] = np.eye(A)[a]
                p_a_list.append(p_a)
        Q = game.batch_get_value(np.stack(p_a_list)).reshape(N, A)
        last_pi = np.copy(pi)
        pi = pi * soft_BR(Q, alpha=1./lr)
        pi = (pi + 1e-4) / (pi + 1e-4).sum(-1, keepdims=True)
        if isclose(pi, last_pi):
            converge += 1
            # if converge >= max(50, 0.5 * i):
            #     break
        if logger:
            logger.log(pi)
    if logger:
        logger.terminate()
    return pi

def CFR(pi, game, n_iters=100, lr=1e-1, alpha=0, noise=0, logger=None):
    # def pi: [N, A]
    N, A = pi.shape[0], pi.shape[1]
    converge = 0
    Regret_sum = np.zeros_like(pi)
    pi_list = []
    for i in range(n_iters):
        p = np.repeat(pi, 2, 0)
        V = game.get_value(p)
        p_a_list = []
        for x in range(N):
            for a in range(A):
                p_a = np.copy(p)
                p_a[x] = np.eye(A)[a]
                p_a_list.append(p_a)
        Q = game.batch_get_value(np.stack(p_a_list)).reshape(N, A)
        Regret = Q - V
        Regret_sum += lr * Regret
        Regret_sum = np.clip(Regret_sum, -1, 1)
        Regret_positive = np.clip(Regret_sum, 0, 1e9) + 1e-9
        pi = Regret_positive / Regret_positive.sum(-1, keepdims=True)
        pi_list.append(pi)
        avg_pi = np.stack(pi_list).mean(0)
        if isclose(avg_pi, pi):
            converge += 1
            # if converge >= max(5, 0.1 * i):
            #     break
        if logger:
            logger.log_meta(meta_pi=(pi_list, np.ones(i + 1) / (i + 1)))
    if logger:
        logger.terminate()
    return pi

def PSRO_uniform(pi, game, m, n_iters=100, lr=1e-1, alpha=0, noise=0, reset=True, logger=None):
    # def pi: [N, A]
    N, A = pi.shape[0], pi.shape[1]
    pi_list = [pi]
    br = np.copy(pi)
    if logger:
        pi_logger, meta_logger = logger
    for i in range(1, m + 1):
        if reset:
            br = get_pi0(N, A, pi0_gen_mode)
        converge = 0
        for it in range(n_iters):
            p_a_list = []
            for j in range(i):
                p = np.concatenate([br, pi_list[j]], 0)
                for x in range(N):
                    for a in range(A):
                        p_a = np.copy(p)
                        p_a[x] = np.eye(A)[a]
                        p_a_list.append(p_a)
            Q = game.batch_get_value(np.stack(p_a_list)).reshape(i, N, A)
            Q = np.mean(Q, 0)
            sol = soft_BR(Q, alpha, noise / (it + 2))
            batch_lr = lr * (np.ones(N) + np.eye(N)[it % N]).reshape(N, 1)
            br = (1 - batch_lr) * br + batch_lr * sol
            if logger:
                pi_logger.log(br)
                meta_logger.log_meta(meta_pi=(pi_list, np.ones(i) / i))
            if isclose(br, sol):
                converge += 1
                if converge >= max(5, 0.1 * it) and it >= 50:
                    break
            else:
                converge = 0
        pi_list.append(br)
        pi = np.mean(np.stack(pi_list), 0)
        if logger:
            if reset:
                pi_logger.terminate()
            meta_logger.terminate()
    if not reset:
        pi_logger.terminate()
    return pi

def matrixNE(payoff_mat):
    n = payoff_mat.shape[0]
    p, q = np.ones(n) / n, np.ones(n) / n
    for i in range(1, 1000):
        a = np.eye(n)[np.argmax(payoff_mat @ q)]
        b = np.eye(n)[np.argmin(p @ payoff_mat)]
        p = p * i / (i + 1) + a / (i + 1)
        q = q * i / (i + 1) + b / (i + 1)
    return p, q

def PSRO_NE(pi, game, m, n_iters=100, lr=1e-1, alpha=0, noise=0, reset=True, logger=None):
    # def pi: [N, A]
    N, A = pi.shape[0], pi.shape[1]
    pi_list = [pi]
    coeff = np.ones(1)
    meta_pi = (pi_list, coeff)
    payoff_mat = np.zeros((m + 1, m + 1))
    br = np.copy(pi)
    if logger:
        pi_logger, meta_logger = logger
    for i in range(1, m + 1):
        if reset:
            br = get_pi0(N, A, pi0_gen_mode)
        converge = 0
        for it in range(n_iters):
            p_a_list = []
            for j in range(i):
                p = np.concatenate([br, pi_list[j]], 0)
                for x in range(N):
                    for a in range(A):
                        p_a = np.copy(p)
                        p_a[x] = np.eye(A)[a]
                        p_a_list.append(p_a)
            Q = game.batch_get_value(np.stack(p_a_list)).reshape(i, N, A)
            Q = (Q * coeff.reshape(-1, 1, 1)).sum(0)
            sol = soft_BR(Q, alpha, noise / (it + 2))
            batch_lr = lr * (np.ones(N) + np.eye(N)[it % N]).reshape(N, 1)
            br = (1 - batch_lr) * br + batch_lr * sol
            if logger:
                pi_logger.log(br)
                meta_logger.log_meta(meta_pi)
            if isclose(br, sol):
                converge += 1
                if converge >= max(5, 0.1 * it) and it >= 50:
                    break
            else:
                converge = 0
        pi_list.append(br)
        battles = []
        for j in range(i):
            battles.append(np.concatenate([br, pi_list[j]], 0))
        br_vs_past = game.batch_get_value(np.stack(battles))
        payoff_mat[i, :i] = br_vs_past
        payoff_mat[:i, i] = -br_vs_past
        coeff, _ = matrixNE(payoff_mat[:i + 1, :i + 1])
        meta_pi = (pi_list, coeff)
        pi = (coeff.reshape(-1, 1, 1) * np.stack(pi_list)).sum(0)
        if logger:
            if reset:
                pi_logger.terminate()
            meta_logger.terminate()
    if not reset:
        pi_logger.terminate()
    return pi

def ODO(pi, game, m, mu, n_iters=100, lr=1e-1, alpha=0, noise=0, reset=True, logger=None):
    # def pi: [N, A]
    N, A = pi.shape[0], pi.shape[1]
    pi_list = [pi]
    coeff = np.ones(1)
    meta_pi = (pi_list, coeff)
    payoff_mat = np.zeros((m + 1, m + 1))
    br = np.copy(pi)
    if logger:
        pi_logger, meta_logger = logger
    for i in range(1, m + 1):
        if reset:
            br = get_pi0(N, A, pi0_gen_mode)
        converge = 0
        for it in range(n_iters):
            p_a_list = []
            for j in range(len(pi_list)):
                p = np.concatenate([br, pi_list[j]], 0)
                for x in range(N):
                    for a in range(A):
                        p_a = np.copy(p)
                        p_a[x] = np.eye(A)[a]
                        p_a_list.append(p_a)
            Q = game.batch_get_value(np.stack(p_a_list)).reshape(len(pi_list), N, A)
            Q = (Q * coeff.reshape(-1, 1, 1)).sum(0)
            sol = soft_BR(Q, alpha, noise / (it + 2))
            batch_lr = lr * (np.ones(N) + np.eye(N)[it % N]).reshape(N, 1)
            br = (1 - batch_lr) * br + batch_lr * sol
            if logger:
                pi_logger.log(br)
                meta_logger.log_meta(meta_pi)
            if isclose(br, sol):
                converge += 1
                if converge >= max(5, 0.1 * it) and it >= 50:
                    break
            else:
                converge = 0
        
        k = None
        for j in range(len(pi_list)):
            if isclose(br, pi_list[j]):
                k = j
                break
        if k is None:
            k = len(pi_list)
            pi_list.append(br)
            battles = []
            for j in range(k):
                battles.append(np.concatenate([br, pi_list[j]], 0))
            br_vs_past = game.batch_get_value(np.stack(battles))
            payoff_mat[k, :k] = br_vs_past
            payoff_mat[:k, k] = -br_vs_past
            coeff = np.ones(k + 1) / (k + 1)
        
        print("weight:", payoff_mat[:len(coeff), :len(coeff)] @ coeff)
        print("softmax:", softmax(mu * payoff_mat[:len(coeff), :len(coeff)] @ coeff))
        coeff = coeff * softmax(mu * payoff_mat[:len(coeff), :len(coeff)] @ coeff)
        print("coeff", np.round(coeff, 2))
        print()
        coeff /= coeff.sum()
        meta_pi = (pi_list, coeff)
        pi = (coeff.reshape(-1, 1, 1) * np.stack(pi_list)).sum(0)
        if logger:
            if reset:
                pi_logger.terminate()
            meta_logger.terminate()
    if not reset:
        pi_logger.terminate()
    return pi

def FXP(pi, game, m, eta=0.3, n_iters=100, lr=1e-1, alpha=0, noise=0, logger=None, reset=False):
    # def pi: [N, A]
    pi_0_copy = pi.copy()
    N, A = pi.shape[0], pi.shape[1]
    pi_list = []
    mu_0 = get_pi0(N, A, pi0_gen_mode)
    mu_list = [mu_0]
    all_list = [mu_0]
    payoff_mat_all = np.zeros((2 * m + 1, 2 * m + 1))
    payoff_mat_mc = np.zeros((m, m + 1))
    meta_pi = (all_list, np.ones(1))
    if logger:
        pi_logger, ct_logger, meta_logger = logger
    for i in range(1, m + 1):
        if reset:
            pi = pi_0_copy.copy()
        coeff, _ = matrixNE(payoff_mat_all[:2*i-1, :2*i-1])
        meta_pi = (all_list, np.copy(coeff))

        converge = 0
        couter_iters = n_iters
        for it in range(n_iters):
            p_a_list = []
            for j in range(i * 2):
                if j < i * 2 - 1:
                    p = np.concatenate([pi, all_list[j]], 0)
                else:
                    p = np.concatenate([pi, pi], 0)
                for x in range(N):
                    for a in range(A):
                        p_a = np.copy(p)
                        p_a[x] = np.eye(A)[a]
                        p_a_list.append(p_a)
            Q = game.batch_get_value(np.stack(p_a_list)).reshape(i * 2, N, A)
            Q = eta * Q[-1] + (1 - eta) * (Q[:-1] * coeff.reshape(-1, 1, 1)).sum(0)
            sol = soft_BR(Q, alpha, noise / (it + 2))
            batch_lr = lr * (np.ones(N) + np.eye(N)[it % N]).reshape(N, 1)
            pi = (1 - batch_lr) * pi + batch_lr * sol
            if logger:
                pi_logger.log(pi)
                meta_logger.log_meta(meta_pi)
            if isclose(pi, sol):
                converge += 1
                if converge >= max(5, 0.1 * it) and it >= 50:
                    couter_iters = it + 1
                    break
            else:
                converge = 0

        pi_list.append(pi)
        battles = []
        for j in range(i):
            battles.append(np.concatenate([pi, mu_list[j]], 0))
        pi_vs_past_mu = game.batch_get_value(np.stack(battles))
        payoff_mat_mc[i-1, :i] = pi_vs_past_mu
        
        all_list.append(pi)
        battles = []
        for j in range(2 * i - 1):
            battles.append(np.concatenate([pi, all_list[j]], 0))
        pi_vs_all = game.batch_get_value(np.stack(battles))
        payoff_mat_all[2*i-1, :2*i-1] = pi_vs_all
        payoff_mat_all[:2*i-1, 2*i-1] = -pi_vs_all

        if logger:
            pi_logger.terminate()
            meta_logger.terminate()
        coeff, _ = matrixNE(payoff_mat_all[:2*i, :2*i])
        meta_pi = (all_list, np.copy(coeff))

        coeff, _ = matrixNE(payoff_mat_mc[:i, :i])
        mu = get_pi0(N, A, pi0_gen_mode)

        converge = 0
        for it in range(couter_iters):
            p_a_list = []
            for j in range(i):
                p = np.concatenate([mu, pi_list[j]], 0)
                for x in range(N):
                    for a in range(A):
                        p_a = np.copy(p)
                        p_a[x] = np.eye(A)[a]
                        p_a_list.append(p_a)
            Q = game.batch_get_value(np.stack(p_a_list)).reshape(i, N, A)
            Q = (Q * coeff.reshape(-1, 1, 1)).sum(0)
            sol = soft_BR(Q, alpha, noise / (it + 2))
            batch_lr = lr * (np.ones(N) + np.eye(N)[it % N]).reshape(N, 1)
            mu = (1 - batch_lr) * mu + batch_lr * sol
            if logger:
                ct_logger.log(mu)
                meta_logger.log_meta(meta_pi)
            if isclose(mu, sol):
                converge += 1
                if converge >= max(5, 0.1 * it):
                    break
            else:
                converge = 0
        
        mu_list.append(mu)
        battles = []
        for j in range(i):
            battles.append(np.concatenate([pi_list[j], mu], 0))
        past_vs_mu = game.batch_get_value(np.stack(battles))
        payoff_mat_mc[:i, i] = past_vs_mu

        all_list.append(mu)
        battles = []
        for j in range(2 * i):
            battles.append(np.concatenate([mu, all_list[j]], 0))
        mu_vs_all = game.batch_get_value(np.stack(battles))
        payoff_mat_all[2*i, :2*i] = mu_vs_all
        payoff_mat_all[:2*i, 2*i] = -mu_vs_all
        
        if logger:
            ct_logger.terminate()
            meta_logger.terminate()
    if logger:
        pi_logger.terminate()
    
    coeff, _ = matrixNE(payoff_mat_all)
    single_meta_pi = (coeff.reshape(-1, 1, 1) * np.stack(all_list, 0)).sum(0)
    return pi, single_meta_pi


if __name__ == "__main__":
    to_run = ["SP", "FP", "FSP", "FoReL", "Replicator", "MWU", "CFR", "FXP", "FXP_c", "FXP_meta", "FXP_reset", "FXP_reset_c", "FXP_reset_meta", "PSRO_uniform", "PSRO_uniform_meta", "PSRO_uniform_warm", "PSRO_uniform_warm_meta", "PSRO_NE", "PSRO_NE_meta", "PSRO_NE_warm", "PSRO_NE_warm_meta", "ODO", "ODO_meta", "ODO_warm", "ODO_warm_meta"]
    # to_run = ["FXP", "FXP_c", "FXP_meta"]
    # to_run = ["SP", "FP", "FSP", "FXP_reset", "FXP_reset_c", "FXP_reset_meta"] #, "FXP_c", "FXP_meta", "PSRO_uniform", "PSRO_uniform_meta", "PSRO_uniform_warm", "PSRO_uniform_warm_meta", "PSRO_NE", "PSRO_NE_meta", "PSRO_NE_warm", "PSRO_NE_warm_meta"]
    np.random.seed(0)

    N, A = 3, 2
    game = ExampleGame(N, A, episode_length=1, C=1.5, eps=0.1)
    pi_0 = get_pi0(N, A, pi0_gen_mode)

    nCol, nRow = 5, max(5, (len(to_run) + 2) // 5)
    fig, axes = plt.subplots(ncols=nCol, nrows=nRow, figsize=(32, 32))

    global_n_inter = 1000
    global_lr = 0.1
    global_alpha = 0

    if not os.path.exists("models"):
        os.makedirs("models")
    if not os.path.exists("figs"):
        os.makedirs("figs")

    if "SP" in to_run:
        print("start SP")
        pi_logger = EvalLogger(game)
        SP_pi = SP(np.copy(pi_0), game, n_iters=global_n_inter, lr=global_lr, alpha=global_alpha, noise=0, logger=pi_logger)
        k = to_run.index("SP")
        pi_logger.save("SP")
        pi_logger.paint(axes[k // nCol, k % nCol])
        print(np.round(SP_pi, 2))
        print()

    if "FP" in to_run:
        print("start FP")
        pi_logger = EvalLogger(game)
        FP_pi = FP(np.copy(pi_0), game, n_iters=global_n_inter, lr=global_lr, alpha=global_alpha, noise=0, logger=pi_logger)
        k = to_run.index("FP")
        pi_logger.save("FP")
        pi_logger.paint(axes[k // nCol, k % nCol])
        print(np.round(FP_pi, 2))
        print()

    if "FSP" in to_run:
        print("start FSP")
        pi_logger = EvalLogger(game)
        FSP_pi = FSP(np.copy(pi_0), game, eta=0.99, n_iters=global_n_inter, lr=global_lr, alpha=global_alpha, noise=0, logger=pi_logger)
        k = to_run.index("FSP")
        pi_logger.save("FSP")
        pi_logger.paint(axes[k // nCol, k % nCol])
        print(np.round(FSP_pi, 2))
        print()
    
    if "Replicator" in to_run:
        print("start Replicator")
        pi_logger = EvalLogger(game)
        Replicator_pi = Replicator(np.copy(pi_0), game, n_iters=global_n_inter, lr=0.8, alpha=0, noise=0, logger=pi_logger)
        k = to_run.index("Replicator")
        pi_logger.save("Replicator")
        pi_logger.paint(axes[k // nCol, k % nCol])
        print(np.round(Replicator_pi, 2))
        print()
    
    if "FoReL" in to_run:
        print("start FoReL")
        pi_logger = EvalLogger(game)
        FoReL_pi = FoReL(np.copy(pi_0), game, n_iters=global_n_inter, lr=20, alpha=0, noise=0, logger=pi_logger)
        k = to_run.index("FoReL")
        pi_logger.save("FoReL")
        pi_logger.paint(axes[k // nCol, k % nCol])
        print(np.round(FoReL_pi, 2))
        print()
    
    if "MWU" in to_run:
        print("start MWU")
        pi_logger = EvalLogger(game)
        MWU_pi = MWU(np.copy(pi_0), game, n_iters=global_n_inter, lr=10, alpha=0, noise=0, logger=pi_logger)
        k = to_run.index("MWU")
        pi_logger.save("MWU")
        pi_logger.paint(axes[k // nCol, k % nCol])
        print(np.round(MWU_pi, 2))
        print()

    if "CFR" in to_run:
        print("start CFR")
        pi_logger = EvalLogger(game)
        CFR_pi = CFR(np.copy(pi_0), game, n_iters=global_n_inter, lr=10, alpha=0, noise=0, logger=pi_logger)
        k = to_run.index("CFR")
        pi_logger.save("CFR")
        pi_logger.paint(axes[k // nCol, k % nCol])
        print(np.round(CFR_pi, 2))
        print()

    if "PSRO_uniform" in to_run or "PSRO_uniform_meta" in to_run:
        print("start PSRO_uniform")
        pi_logger, meta_logger = EvalLogger(game), EvalLogger(game)
        PSRO_uniform_pi = PSRO_uniform(np.copy(pi_0), game, m=20, n_iters=global_n_inter, lr=global_lr, alpha=0, noise=0, logger=(pi_logger, meta_logger))
        if "PSRO_uniform" in to_run:
            k = to_run.index("PSRO_uniform")
            pi_logger.save("PSRO_uniform")
            pi_logger.paint(axes[k // nCol, k % nCol])
        if "PSRO_uniform_meta" in to_run:
            k = to_run.index("PSRO_uniform_meta")
            meta_logger.save("PSRO_uniform_meta")
            meta_logger.paint(axes[k // nCol, k % nCol])
        print(np.round(PSRO_uniform_pi, 2))
        print()
    
    if "PSRO_uniform_warm" in to_run or "PSRO_uniform_warm_meta" in to_run:
        print("start PSRO_uniform_warm")
        pi_logger, meta_logger = EvalLogger(game), EvalLogger(game)
        PSRO_uniform_pi = PSRO_uniform(np.copy(pi_0), game, m=20, n_iters=global_n_inter, lr=global_lr, alpha=0, noise=0, logger=(pi_logger, meta_logger), reset=False)
        if "PSRO_uniform_warm" in to_run:
            k = to_run.index("PSRO_uniform_warm")
            pi_logger.save("PSRO_uniform_warm")
            pi_logger.paint(axes[k // nCol, k % nCol])
        if "PSRO_uniform_warm_meta" in to_run:
            k = to_run.index("PSRO_uniform_warm_meta")
            meta_logger.save("PSRO_uniform_warm_meta")
            meta_logger.paint(axes[k // nCol, k % nCol])
        print(np.round(PSRO_uniform_pi, 2))
        print()

    if "PSRO_NE" in to_run or "PSRO_NE_meta" in to_run:
        print("start PSRO_NE")
        pi_logger, meta_logger = EvalLogger(game), EvalLogger(game)
        PSRO_NE_pi = PSRO_NE(np.copy(pi_0), game, m=12, n_iters=global_n_inter, lr=global_lr, alpha=0, noise=0, logger=(pi_logger, meta_logger))
        if "PSRO_NE" in to_run:
            k = to_run.index("PSRO_NE")
            pi_logger.paint(axes[k // nCol, k % nCol])
            pi_logger.save("PSRO_NE")
        if "PSRO_NE_meta" in to_run:
            k = to_run.index("PSRO_NE_meta")
            meta_logger.paint(axes[k // nCol, k % nCol])
            meta_logger.save("PSRO_NE_meta")
        print(np.round(PSRO_NE_pi, 2))
        print()

    if "PSRO_NE_warm" in to_run or "PSRO_NE_warm_meta" in to_run:
        print("start PSRO_NE_warm")
        pi_logger, meta_logger = EvalLogger(game), EvalLogger(game)
        PSRO_NE_pi = PSRO_NE(np.copy(pi_0), game, m=20, n_iters=global_n_inter, lr=global_lr, alpha=0, noise=0, logger=(pi_logger, meta_logger), reset=False)
        if "PSRO_NE_warm" in to_run:
            k = to_run.index("PSRO_NE_warm")
            pi_logger.paint(axes[k // nCol, k % nCol])
            pi_logger.save("PSRO_NE_warm")
        if "PSRO_NE_warm_meta" in to_run:
            k = to_run.index("PSRO_NE_warm_meta")
            meta_logger.paint(axes[k // nCol, k % nCol])
            meta_logger.save("PSRO_NE_warm_meta")
        print(np.round(PSRO_NE_pi, 2))
        print()
    
    if "ODO" in to_run or "ODO_meta" in to_run:
        print("start ODO")
        pi_logger, meta_logger = EvalLogger(game), EvalLogger(game)
        ODO_pi = ODO(np.copy(pi_0), game, m=12, mu=1, n_iters=global_n_inter, lr=global_lr, alpha=0, noise=0, logger=(pi_logger, meta_logger))
        if "ODO" in to_run:
            k = to_run.index("ODO")
            pi_logger.paint(axes[k // nCol, k % nCol])
            pi_logger.save("ODO")
        if "ODO_meta" in to_run:
            k = to_run.index("ODO_meta")
            meta_logger.paint(axes[k // nCol, k % nCol])
            meta_logger.save("ODO_meta")
        print(np.round(ODO_pi, 2))
        print()
    
    if "ODO_warm" in to_run or "ODO_warm_meta" in to_run:
        print("start ODO_warm")
        pi_logger, meta_logger = EvalLogger(game), EvalLogger(game)
        ODO_pi = ODO(np.copy(pi_0), game, m=20, mu=1, n_iters=global_n_inter, lr=global_lr, alpha=0, noise=0, logger=(pi_logger, meta_logger), reset=False)
        if "ODO_warm" in to_run:
            k = to_run.index("ODO_warm")
            pi_logger.paint(axes[k // nCol, k % nCol])
            pi_logger.save("ODO_warm")
        if "ODO_warm_meta" in to_run:
            k = to_run.index("ODO_warm_meta")
            meta_logger.paint(axes[k // nCol, k % nCol])
            meta_logger.save("ODO_warm_meta")
        print(np.round(ODO_pi, 2))
        print()
    
    if "FXP" in to_run or "FXP_c" in to_run or "FXP_meta" in to_run:
        print("start FXP")
        pi_logger, ct_logger, meta_logger = EvalLogger(game), EvalLogger(game), EvalLogger(game)
        FXP_pi, FXP_meta_pi = FXP(np.copy(pi_0), game, m=10, eta=0.3, n_iters=global_n_inter, lr=global_lr, alpha=0, noise=0, logger=(pi_logger, ct_logger, meta_logger))
        if "FXP" in to_run:
            k = to_run.index("FXP")
            pi_logger.paint(axes[k // nCol, k % nCol])
            pi_logger.save("FXP")
        if "FXP_c" in to_run:
            k = to_run.index("FXP_c")
            ct_logger.paint(axes[k // nCol, k % nCol])
            ct_logger.save("FXP_c")
        if "FXP_meta" in to_run:
            k = to_run.index("FXP_meta")
            meta_logger.paint(axes[k // nCol, k % nCol])
            meta_logger.save("FXP_meta")
        print(np.round(FXP_pi, 2))
        print()
        print(np.round(FXP_meta_pi, 2))
        print()

    if "FXP_reset" in to_run or "FXP_reset_c" in to_run or "FXP_reset_meta" in to_run:
        print("start FXP_reset")
        pi_logger, ct_logger, meta_logger = EvalLogger(game), EvalLogger(game), EvalLogger(game)
        FXP_pi, FXP_meta_pi = FXP(np.copy(pi_0), game, m=10, eta=0.3, n_iters=global_n_inter, lr=global_lr, alpha=0, noise=0, logger=(pi_logger, ct_logger, meta_logger), reset=True)
        if "FXP_reset" in to_run:
            k = to_run.index("FXP_reset")
            pi_logger.paint(axes[k // nCol, k % nCol])
            pi_logger.save("FXP_reset")
        if "FXP_reset_c" in to_run:
            k = to_run.index("FXP_reset_c")
            ct_logger.paint(axes[k // nCol, k % nCol])
            ct_logger.save("FXP_reset_c")
        if "FXP_reset_meta" in to_run:
            k = to_run.index("FXP_reset_meta")
            meta_logger.paint(axes[k // nCol, k % nCol])
            meta_logger.save("FXP_reset_meta")
        print(np.round(FXP_pi, 2))
        print()
        print(np.round(FXP_meta_pi, 2))
        print()


    red_patch = mpatches.Patch(color="red", label="p0 expl")
    green_patch = mpatches.Patch(color="green", label="p1 expl")
    fig.legend(handles=[red_patch, green_patch], fontsize=30)
    for i, name in enumerate(to_run):
        axes[i // nCol, i % nCol].set_title(name, fontsize=30)
    fig.savefig("figs/example_all.png")
