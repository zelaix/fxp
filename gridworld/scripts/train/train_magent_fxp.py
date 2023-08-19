from pathlib import Path
import numpy as np
import os
import setproctitle
import socket
import sys
import torch
import wandb

from gridworld.config import get_config
from gridworld.envs.magent.MAgent_Env import MAgentXPEnv
from gridworld.envs.env_wrappers import SubprocVecEnv, DummyVecEnv


def make_main_env(all_args):
    kwargs = dict(
        main_prob=all_args.main_sp_prob,
        counter_prob=all_args.main_xp_prob,
        main_pop_prob=all_args.main_fsp_prob,
        counter_pop_prob=all_args.main_fxp_prob,
    )
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "MAgentXP":
                env = MAgentXPEnv(all_args, **kwargs)
            else:
                print("Can not support the " + all_args.env_name + " environment.")
                raise NotImplementedError
            env.seed(all_args.seed + rank * 1000)
            return env
        return init_env
    if all_args.n_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])


def make_counter_env(all_args):
    kwargs = dict(
        main_prob=all_args.counter_xp_prob,
        counter_prob=all_args.counter_sp_prob,
        main_pop_prob=all_args.counter_fxp_prob,
        counter_pop_prob=all_args.counter_fsp_prob,
    )
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "MAgentXP":
                env = MAgentXPEnv(all_args, **kwargs)
            else:
                print("Can not support the " + all_args.env_name + " environment.")
                raise NotImplementedError
            env.seed(all_args.seed + rank * 1000)
            return env
        return init_env
    if all_args.n_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])


def make_eval_env(all_args):
    kwargs = dict(
        main_prob=0,
        counter_prob=0,
        main_pop_prob=0.5,
        counter_pop_prob=0.5,
        is_eval=True
    )
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "MAgentXP":
                env = MAgentXPEnv(all_args, **kwargs)
            else:
                print("Can not support the " + all_args.env_name + " environment.")
                raise NotImplementedError
            env.seed(all_args.seed * 50000 + rank * 10000)
            return env
        return init_env
    if all_args.n_eval_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(all_args.n_eval_rollout_threads)])


def parse_args(args, parser):
    parser.add_argument("--scenario_name", type=str, default="battle", help="which scenario to run on.")
    parser.add_argument("--map_size", type=int, default=13, help="map size of the env.")
    parser.add_argument("--max_episode_length", type=int, default=200, help="maximum episode length.")
    parser.add_argument("--num_agents", type=int, default=2, help="number of controlled players.")
    parser.add_argument("--fp_interval", type=int, default=2000000, help="ficititious-play interval, by default 2000000.")
    parser.add_argument("--main_sp_prob", type=float, default=0.3, help="main self play probability.")
    parser.add_argument("--main_fsp_prob", type=float, default=0.2, help="main self play probability.")
    parser.add_argument("--main_xp_prob", type=float, default=0.3, help="main cross play probability.")
    parser.add_argument("--main_fxp_prob", type=float, default=0.2, help="main cross play probability.")
    parser.add_argument("--counter_sp_prob", type=float, default=0, help="counter self play probability.")
    parser.add_argument("--counter_fsp_prob", type=float, default=0.3, help="counter self play probability.")
    parser.add_argument("--counter_xp_prob", type=float, default=0.4, help="counter cross play probability.")
    parser.add_argument("--counter_fxp_prob", type=float, default=0.3, help="counter cross play probability.")
    parser.add_argument("--main_pop_size", type=int, default=0, help="number of policies in main population.")
    parser.add_argument("--main_pop_sample_probs", type=str, default=None, help="main population sample probabilities separate by space ' ' ")
    parser.add_argument("--counter_pop_size", type=int, default=0, help="number of policies in counter population.")
    parser.add_argument("--counter_pop_sample_probs", type=str, default=None, help="counter population sample probabilities separated by space ' ' ")
    all_args = parser.parse_known_args(args)[0]
    if all_args.main_pop_size == 0:
        all_args.main_pop_sample_probs = []
    elif all_args.main_pop_sample_probs is None:
        all_args.main_pop_sample_probs = np.ones(all_args.main_pop_size) / all_args.main_pop_size
    else:
        probs = np.array([float(prob) for prob in all_args.main_pop_sample_probs.split(" ")])
        probs = probs / np.sum(probs)
        assert len(probs) == all_args.main_pop_size
        all_args.main_pop_sample_probs = probs

    if all_args.counter_pop_size == 0:
        all_args.counter_pop_sample_probs = []
    elif all_args.counter_pop_sample_probs is None:
        all_args.counter_pop_sample_probs = np.ones(all_args.counter_pop_size) / all_args.counter_pop_size
    else:
        probs = np.array([float(prob) for prob in all_args.counter_pop_sample_probs.split(" ")])
        probs = probs / np.sum(probs)
        assert len(probs) == all_args.counter_pop_size
        all_args.counter_pop_sample_probs = probs

    return all_args


def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)

    if all_args.algorithm_name == "rmappo" or all_args.algorithm_name == "rmappg":
        assert (all_args.use_recurrent_policy or all_args.use_naive_recurrent_policy), ("check recurrent policy!")
    elif all_args.algorithm_name == "mappo" or all_args.algorithm_name == "mappg":
        assert (all_args.use_recurrent_policy == False and all_args.use_naive_recurrent_policy == False), ("check recurrent policy!")
    else:
        raise NotImplementedError

    # cuda
    if all_args.cuda and torch.cuda.is_available():
        print("choose to use gpu...")
        device = torch.device("cuda:0")
        torch.set_num_threads(all_args.n_training_threads)
        if all_args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        print("choose to use cpu...")
        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_training_threads)

    # run dir
    run_dir = Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[0] + "/results") / all_args.env_name / all_args.scenario_name / all_args.algorithm_name / all_args.experiment_name
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    # wandb
    if all_args.use_wandb:
        run = wandb.init(config=all_args,
                         project="magent",
                         entity=all_args.user_name,
                         notes=socket.gethostname(),
                         name=all_args.wandb_name,
                         group=all_args.scenario_name,
                         dir=str(run_dir),
                         job_type="training",
                         reinit=True)
    else:
        if not run_dir.exists():
            curr_run = 'run1'
        else:
            exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in run_dir.iterdir() if str(folder.name).startswith('run')]
            if len(exst_run_nums) == 0:
                curr_run = 'run1'
            else:
                curr_run = 'run%i' % (max(exst_run_nums) + 1)
        run_dir = run_dir / curr_run
        if not run_dir.exists():
            os.makedirs(str(run_dir))
    all_args.run_dir = run_dir

    setproctitle.setproctitle("-".join([
        all_args.env_name, 
        all_args.scenario_name, 
        all_args.algorithm_name, 
        all_args.experiment_name
    ]) + "@" + all_args.user_name)
    
    # seed
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)

    # env init
    main_envs = make_main_env(all_args)
    counter_envs = make_counter_env(all_args)
    main_eval_envs = make_eval_env(all_args) if all_args.use_eval else None
    counter_eval_envs = make_eval_env(all_args) if all_args.use_eval else None
    num_agents = all_args.num_agents

    config = {
        "all_args": all_args,
        "main_envs": main_envs,
        "counter_envs": counter_envs,
        "main_eval_envs": main_eval_envs,
        "counter_eval_envs": counter_eval_envs,
        "num_agents": num_agents,
        "device": device,
        "run_dir": run_dir
    }

    # run experiments
    if all_args.share_policy:
        from gridworld.runner.cross_play.magent_fxp_runner import MAgentFXPRunner as Runner
    else:
        raise NotImplementedError

    runner = Runner(config)
    runner.run()
    
    # post process
    main_envs.close()
    counter_envs.close()
    if all_args.use_eval and main_eval_envs is not None:
        main_eval_envs.close()
    if all_args.use_eval and counter_eval_envs is not None:
        counter_eval_envs.close()

    if all_args.use_wandb:
        run.finish()
    else:
        runner.writter.export_scalars_to_json(str(runner.log_dir + '/summary.json'))
        runner.writter.close()


if __name__ == "__main__":
    main(sys.argv[1:])
