from pathlib import Path
import numpy as np
import os
import setproctitle
import socket
import sys
import torch
import wandb

from gridworld.config import get_config
from gridworld.envs.magent.MAgent_Env import MAgentEnv
from gridworld.envs.env_wrappers import SubprocVecEnv, DummyVecEnv


def make_train_env(all_args):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "MAgent":
                env = MAgentEnv(all_args)
            else:
                print("Can not support the " +
                      all_args.env_name + " environment.")
                raise NotImplementedError
            env.seed(all_args.seed + rank * 1000)
            return env
        return init_env
    if all_args.n_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])


def make_eval_env(all_args):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "MAgent":
                env = MAgentEnv(all_args, is_eval=True)
            else:
                print("Can not support the " +
                      all_args.env_name + " environment.")
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
    parser.add_argument("--num_agents", type=int, default=4, help="number of controlled players.")
    parser.add_argument("--one_side", action="store_true", default=False, 
                        help="by default False. If True, only one side of the env is controlled by the trained agents.")
    parser.add_argument("--use_population", action="store_true", default=False, 
                        help="by default False. If True, use opponent population.")
    parser.add_argument("--oppo_model_dir", type=str, default="random", help="by default random. Set to be the path to opponent model.")
    parser.add_argument("--oppo_sample_probs", type=str, default=None, help="sample probability of opponent policy, by default uniform.")
    parser.add_argument("--psro_eval_episodes", type=int, default=50, help="number of episode for each payoff entry.")
    parser.add_argument("--meta_solver", type=str, default="uniform", choices=["uniform", "nash"],
                        help="by default random. Set to be the path to opponent model.")
    parser.add_argument("--iter_min_steps", type=int, default=2000000, help="minimum steps for a PSRO iteration.")
    parser.add_argument("--iter_max_steps", type=int, default=50000000, help="maximum steps for a PSRO iteration.")
    parser.add_argument("--mean_threshold", type=float, default=0.99, help="minimum win rate mean for convergence check.")
    parser.add_argument("--std_threshold", type=float, default=1e-2, help="maximum win rate std for convergence check.")
    parser.add_argument("--eval_deterministic", action="store_true", default=False, 
                        help="by default False. If True, use action with highest probability")
    all_args = parser.parse_known_args(args)[0]
    all_args.use_sp = False
    all_args.sp_prob = 0.0
    if all_args.meta_solver == "uniform":
        all_args.use_eval = False
    else:
        all_args.use_eval = True
    all_args.oppo_model_dir = all_args.oppo_model_dir.split(" ")
    all_args.num_oppo_policy = len(all_args.oppo_model_dir)
    if all_args.oppo_sample_probs is None:
        all_args.oppo_sample_probs = [1 / all_args.num_oppo_policy] * all_args.num_oppo_policy
    else:
        raise NotImplementedError
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
    envs = make_train_env(all_args)
    eval_envs = make_eval_env(all_args) if all_args.use_eval else None
    num_agents = all_args.num_agents

    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": eval_envs,
        "num_agents": num_agents,
        "device": device,
        "run_dir": run_dir
    }

    # run experiments
    if all_args.share_policy:
        from gridworld.runner.shared.magent_psro_runner import MAgentPSRORunner as Runner
    else:
        raise NotImplementedError

    runner = Runner(config)
    runner.run()
    
    # post process
    envs.close()
    if all_args.use_eval and eval_envs is not envs:
        eval_envs.close()

    if all_args.use_wandb:
        run.finish()
    else:
        runner.writter.export_scalars_to_json(str(runner.log_dir + '/summary.json'))
        runner.writter.close()


if __name__ == "__main__":
    main(sys.argv[1:])
