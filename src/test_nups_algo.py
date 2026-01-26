import argparse
import copy
import glob
import json
import logging
import os
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import tqdm

from cox.store import Store
from policy_gradients.agent import Trainer, TrainerWithNUPS, calculate_max_kl_div
from run import add_common_parser_opts, override_json_params

logging.disable(logging.INFO)
np.set_printoptions(suppress=True, precision=2)
torch.set_printoptions(precision = 4, sci_mode = False)

# import sys
# import os
# current_dir = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(os.path.join(current_dir, '../'))
# from src.policy_gradients.models import *

import time
# from src.policy_gradients.convex_relaxation import get_kl_bound as get_state_kl_bound
from policy_gradients.convex_relaxation import get_kl_bound as get_state_kl_bound







def load_trainer(params):
    override_params = copy.deepcopy(params)
    excluded_params = [
        "config_path",
        "out_dir_prefix",
        "num_episodes",
        "row_id",
        "exp_id",
        "load_model",
        "seed",
        "deterministic",
        "noise_factor",
        "compute_kl_cert",
        "use_full_backward",
        "sqlite_path",
        "early_terminate",
    ]
    sarsa_params = [
        "sarsa_enable",
        "sarsa_steps",
        "sarsa_eps",
        "sarsa_reg",
        "sarsa_model_path",
    ]
    imit_params = ["imit_enable", "imit_epochs", "imit_model_path", "imit_lr"]

    # original_params contains all flags in config files that are overridden via command.
    for k in list(override_params.keys()):
        if k in excluded_params:
            del override_params[k]

    if params["config_path"]:
        # Load from a pretrained model using existing config.
        # First we need to create the model using the given config file.
        json_params = json.load(open(params["config_path"]))

        params = override_json_params(
            params, 
            json_params, 
            excluded_params + sarsa_params + imit_params + ['beta', 'save_root_dir', 'use_kl_bound', 'num_sampled_episodes'] # ["statewise_attack_eps", "statewise_attack_epsisodes"]
        )

    if params["sarsa_enable"]:
        assert params["attack_method"] == "none" or params["attack_method"] is None, (
            "--train-sarsa is only available when --attack-method=none, but got {}".format(
                params["attack_method"]
            )
        )

    if "load_model" in params and params["load_model"]:
        for k, v in zip(params.keys(), params.values()):
            assert v is not None, f"Value for {k} is None"

        # Create the agent from config file.
        # p = Trainer.agent_from_params(params, store=None)
        p = TrainerWithNUPS.agent_from_params(params, store = None)
        print("Loading pretrained model", params["load_model"])
        pretrained_model = torch.load(params["load_model"])
        if "policy_model" in pretrained_model:
            p.policy_model.load_state_dict(pretrained_model["policy_model"])
        if "val_model" in pretrained_model:
            p.val_model.load_state_dict(pretrained_model["val_model"])
        if "policy_opt" in pretrained_model:
            p.POLICY_ADAM.load_state_dict(pretrained_model["policy_opt"])
        if "val_opt" in pretrained_model:
            p.val_opt.load_state_dict(pretrained_model["val_opt"])
        # Restore environment parameters, like mean and std.
        if "envs" in pretrained_model:
            p.envs = pretrained_model["envs"]
        for e in p.envs:
            e.normalizer_read_only = True
            e.setup_visualization(
                params["show_env"], params["save_frames"], params["save_frames_path"]
            )
    else:
        # Load from experiment directory. No need to use a config.
        base_directory = params["out_dir"]
        store = Store(base_directory, params["exp_id"], mode="r")
        if params["row_id"] < 0:
            row = store["final_results"].df
        else:
            checkpoints = store["checkpoints"].df
            row_id = params["row_id"]
            row = checkpoints.iloc[row_id : row_id + 1]
        print("row to test: ", row)
        if params["cpu"] == None:
            cpu = False
        else:
            cpu = params["cpu"]
        # p, _ = Trainer.agent_from_data(
        #     store,
        #     row,
        #     cpu,
        #     extra_params=params,
        #     override_params=override_params,
        #     excluded_params=excluded_params,
        # )
        p, _ = TrainerWithNUPS.agent_from_data(
            store,
            row,
            cpu,
            extra_params=params,
            override_params=override_params,
            excluded_params=excluded_params,
        )
        store.close()

    # print('Gaussian noise in policy:')
    # print(torch.exp(p.policy_model.log_stdev))
    original_stdev = p.policy_model.log_stdev.clone().detach()
    if params["noise_factor"] != 1.0:
        p.policy_model.log_stdev.data[:] += np.log(params["noise_factor"])
    # if params["deterministic"]:
    #     # print('Policy runs in deterministic mode. Ignoring Gaussian noise.')
    #     p.policy_model.log_stdev.data[:] = -100
    # print('Gaussian noise in policy (after adjustment):')
    # print(torch.exp(p.policy_model.log_stdev))

    return p, params



def get_parser():
    parser = argparse.ArgumentParser(description="Generate experiments to be run.")
    parser.add_argument(
        "--config-path",
        type=str,
        default="",
        required=False,
        help="json for this config",
    )
    parser.add_argument(
        "--out-dir-prefix",
        type=str,
        default="",
        required=False,
        help="prefix for output log path",
    )
    parser.add_argument(
        "--exp-id", type=str, help="experiement id for testing", default=""
    )
    parser.add_argument(
        "--row-id", type=int, help="which row of the table to use", default=-1
    )
    parser.add_argument(
        "--num-episodes", type=int, help="number of episodes for testing", default=50
    )
    parser.add_argument(
        "--compute-kl-cert", action="store_true", help="compute KL certificate"
    )
    parser.add_argument(
        "--use-full-backward",
        action="store_true",
        help="Use full backward LiRPA bound for computing certificates",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="disable Gaussian noise in action for evaluation",
    )
    parser.add_argument(
        "--noise-factor",
        type=float,
        default=1.0,
        help="increase the noise (Gaussian std) by this factor.",
    )
    parser.add_argument(
        "--load-model", type=str, help="load a pretrained model file", default=""
    )
    parser.add_argument("--seed", type=int, help="random seed", default=1234)
    # Sarsa training related options.
    parser.add_argument(
        "--sarsa-enable", action="store_true", help="train a sarsa attack model."
    )
    parser.add_argument(
        "--sarsa-steps", type=int, help="Sarsa training steps.", default=30
    )
    parser.add_argument(
        "--sarsa-model-path",
        type=str,
        help="path to save the sarsa value network.",
        default="sarsa.model",
    )
    parser.add_argument(
        "--imit-enable", action="store_true", help="train a imit attack model."
    )
    parser.add_argument(
        "--imit-epochs", type=int, help="Imit training steps.", default=100
    )
    parser.add_argument(
        "--imit-model-path",
        type=str,
        help="path to save the imit policy network.",
        default="imit.model",
    )
    parser.add_argument(
        "--imit-lr", type=float, help="lr for imitation learning training", default=1e-3
    )
    parser.add_argument(
        "--sarsa-eps",
        type=float,
        help="eps for actions for sarsa training.",
        default=0.02,
    )
    parser.add_argument(
        "--sarsa-reg",
        type=float,
        help="regularization term for sarsa training.",
        default=0.1,
    )
    # Other configs
    parser.add_argument(
        "--sqlite-path", type=str, help="save results to a sqlite database.", default=""
    )
    parser.add_argument(
        "--early-terminate",
        action="store_true",
        help="terminate attack early if low attack reward detected in sqlite.",
    )


    parser.add_argument('--beta', type=float, default=50)
    parser.add_argument('--save_root_dir', type=str, default='./test_nups_robust_ppo_sgld_b1')
    parser.add_argument('--use_kl_bound', type=str, default='b1')
    parser.add_argument('--num_sampled_episodes', type=int, default=10)

    parser = add_common_parser_opts(parser)

    return parser


def test_tensor_eps_in_autolirpa():
    from auto_LiRPA.perturbations import PerturbationLpNorm
    from auto_LiRPA import BoundedModule
    from auto_LiRPA.eps_scheduler import LinearScheduler
    from auto_LiRPA.bounded_tensor import BoundedTensor

    net = nn.Linear(11, 3)

    eps = torch.randn(11) * 0.01
    ptb = PerturbationLpNorm(np.inf, eps = eps)
    print(ptb)
    inputs = torch.randn(11)
    bm_net = BoundedModule(net, inputs)
    x = BoundedTensor(inputs, ptb=PerturbationLpNorm(norm=np.inf, eps=eps))
    print(bm_net)
    print(x)



def main():
    parser = get_parser()

    args = parser.parse_args()
    if args.load_model:
        assert args.config_path, (
            "Need to specificy a config file when loading a pretrained model."
        )

    if args.early_terminate:
        assert args.sqlite_path != "", (
            "Need to specify --sqlite-path to terminate early."
        )

    if args.sarsa_enable:
        if args.sqlite_path != "":
            print(
                "When --sarsa-enable is specified, --sqlite-path and --early-terminate will be ignored."
            )

    params = vars(args)
    seed = params["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)

    p, params = load_trainer(params)

    final_results = {}

    beta = params['beta']
    gamma = p.params.GAMMA
    eps = p.params.ROBUST_PPO_EPS
    print(f'beta = {beta}, gamma = {gamma}, eps = {eps}')

    use_kl_bound = params['use_kl_bound']
    save_root_dir = params['save_root_dir']
    assert use_kl_bound in ['b1', 'b2']
    max_episode_length = 1024
    num_sampled_episodes = params['num_sampled_episodes']

    final_results['beta'] = beta
    final_results['gamma'] = gamma
    final_results['eps'] = eps
    final_results['use_kl_bound'] = use_kl_bound

    if not os.path.exists(save_root_dir):
        os.makedirs(save_root_dir)

    sampled_ep_rewards = []
    sampled_ep_discounted_returns = []
    abs_A_max = -1e5
    for _ in range(num_sampled_episodes):
        ep_length, ep_reward, actions, action_means, states, kl_upper_bound, advantages, ep_discounted_return = p.run_test(max_episode_length, attack_with_nups = False)
        sampled_ep_rewards.append(ep_reward)
        sampled_ep_discounted_returns.append(ep_discounted_return)
        abs_A_max = max(torch.max(torch.abs(advantages)).item(), abs_A_max)
    print('No NUPS!')
    print(f'avg ep_reward of {num_sampled_episodes} episodes = {np.mean(sampled_ep_rewards)} +- {np.std(sampled_ep_rewards)}')
    print(f'avg ep_discounted_return of {num_sampled_episodes} episodes = {np.mean(sampled_ep_discounted_returns)} +- {np.std(sampled_ep_discounted_returns)}')
    max_kl_div = calculate_max_kl_div(gamma, beta, abs_A_max, use_kl_bound)

    print(f'Use KL bound {use_kl_bound}')
    print(f'|A|_max = {abs_A_max}')
    print(f'D = {max_kl_div}')

    final_results['abs_A_max'] = abs_A_max
    final_results['max_kl_div'] = max_kl_div
    final_results['sampled_ep_rewards'] = {}
    final_results['sampled_ep_rewards']['none'] = copy.deepcopy(sampled_ep_rewards)
    final_results['sampled_ep_discounted_returns'] = {}
    final_results['sampled_ep_discounted_returns']['none'] = copy.deepcopy(sampled_ep_discounted_returns)

    # critic, random, sarsa, action
    print('Use NUPS!')
    for attack_method in ['critic', 'random', 'action']:
        p.params.ATTACK_METHOD = attack_method
        sampled_ep_rewards = []
        sampled_ep_discounted_returns = []
        for _ in range(num_sampled_episodes):
            ep_length, ep_reward, actions, action_means, states, kl_upper_bound, advantages, ep_discounted_return = p.run_test(max_episode_length, attack_with_nups = True, max_kl_div = max_kl_div)
            sampled_ep_rewards.append(ep_reward)
            sampled_ep_discounted_returns.append(ep_discounted_return)
        print(f'Attack method = {p.params.ATTACK_METHOD}')
        print(f'avg ep_reward of {num_sampled_episodes} episodes = {np.mean(sampled_ep_rewards)} +- {np.std(sampled_ep_rewards)}')
        final_results['sampled_ep_rewards'][attack_method] = copy.deepcopy(sampled_ep_rewards)

        print(f'avg ep_discounted_return of {num_sampled_episodes} episodes = {np.mean(sampled_ep_discounted_returns)} +- {np.std(sampled_ep_discounted_returns)}')
        final_results['sampled_ep_discounted_returns'][attack_method] = copy.deepcopy(sampled_ep_discounted_returns)

        expected_performance_drop = advantages.mean().item() / (1-gamma)
        print(advantages.mean().item())
        print(f'E[A(s_t,a_t)]/(1-\gamma) = {expected_performance_drop}')


    final_results_json_path = f'{p.params.GAME}_{params["exp_id"]}_klbound={use_kl_bound}_beta={beta}'
    with open(os.path.join(save_root_dir, f'{final_results_json_path}.json'), 'w') as json_file:
        json.dump(final_results, json_file, indent=4) # 'indent=4' makes the file human-readable



def print_nups_res(
    root_dir = './test_nups_b1/', 
    kl_bound = 'b1',
    plot_total_rewards = False,
    plot_discounted_returns = True,
):
    envs = ['Hopper-v5', 'HalfCheetah-v5', 'Walker2d-v5']
    betas = [10.0, 25.0, 50.0, 100.0, 200.0, 400.0, 800.0]
    attack_methods = ['none', 'random', 'critic', 'action']
    
    rewards_table = defaultdict(lambda: [])
    d_returns_table = defaultdict(lambda: [])

    for env in envs:
        for beta in betas:
            json_file_paths = glob.glob(os.path.join(root_dir, f'{env}*klbound={kl_bound}_beta={beta}.json'))
            if len(json_file_paths) == 0:
                continue
            all_rewards = defaultdict(lambda: [])
            all_d_returns = defaultdict(lambda: [])
            for p in json_file_paths:
                with open(p, 'r') as jf:
                    data = json.load(jf)
                    for atk in attack_methods:
                        all_rewards[atk].extend(data['sampled_ep_rewards'][atk])
                        all_d_returns[atk].extend(data['sampled_ep_discounted_returns'][atk])
            
            if plot_total_rewards:
                rewards_table['Env'].append(env)
                rewards_table['Beta'].append(f'{int(beta):d}')
                for atk in attack_methods:
                    rewards_mean, rewards_std = np.mean(all_rewards[atk]).astype(int), np.std(all_rewards[atk]).astype(int)

                    # no attack rewards
                    no_attack_rewards_mean = np.mean(all_rewards['none']).astype(int)
                    diff = rewards_mean - no_attack_rewards_mean

                    rewards_table[atk].append(f'{rewards_mean:d} $\pm$ {rewards_std:d} ({diff:d})')


            if plot_discounted_returns:
                d_returns_table['Env'].append(env)
                d_returns_table['Beta'].append(f'{int(beta):d}')
                for atk in attack_methods:
                    d_returns_mean, d_returns_std = np.mean(all_d_returns[atk]).astype(int), np.std(all_d_returns[atk]).astype(int)

                    # no attack returns
                    no_attack_d_returns_mean = np.mean(all_d_returns['none']).astype(int)
                    diff = d_returns_mean - no_attack_d_returns_mean

                    d_returns_table[atk].append(f'{d_returns_mean:d} $\pm$ {d_returns_std:d} ({diff:d})')

    if plot_total_rewards:
        rewards_table = pd.DataFrame(dict(rewards_table))
        print(rewards_table.to_latex(index = False, caption = 'Total rewards $\sum_t R(s_t,a_t)$'))
        print()

    if plot_discounted_returns:
        d_returns_table = pd.DataFrame(dict(d_returns_table))
        print(d_returns_table.to_latex(index = False, caption = 'Discounted returns $\sum_t \gamma^t R(s_t,a_t)$'))




def generate_nups_task_scripts(
    algos = ['robust_ppo_convex', 'robust_ppo_sgld', 'vanilla_ppo'],
    betas = [10, 25, 50, 100, 200, 400, 800],
    target_sh_name = 'test_nups_algo.sh',
    root_dir = 'exps',
    envs = ['Hopper-v5', 'HalfCheetah-v5', 'Walker2d-v5']
):
    envs_game_to_json = {
        'Hopper-v5' : 'hopper', 
        'HalfCheetah-v5' : 'halfcheetah', 
        'Walker2d-v5' : 'walker'
    }

    all_tasks = []

    for algo in algos:
        for game_env in envs:
            # print(json_env)
            json_env = envs_game_to_json[game_env]
            env_algo_root_path = os.path.join(root_dir, game_env, algo, 'agents')
            for exp_id in os.listdir(env_algo_root_path):
                if os.path.isdir(os.path.join(env_algo_root_path, exp_id)):
                    # print(name)
                    for beta in betas:
                        task_str = f'python -u src/test_nups_algo.py --config-path src/config_{json_env}_{algo}.json --exp-id {exp_id} --deterministic --beta {beta} --use_kl_bound b1 --save_root_dir ./test_nups_{algo}_b1 \n'
                        all_tasks.append(task_str)

    with open(target_sh_name, 'w') as f:
        f.writelines(all_tasks)
    os.chmod(target_sh_name, 0o777)


if __name__ == "__main__":
    # main(use_kl_bound = 'b1', save_root_dir = './test_nups_b1')
    # print_nups_res(./test_nups_b1/, 'b1')

    # main(use_kl_bound = 'b2', save_root_dir = './test_nups_b2')
    # print_nups_res('./test_nups_b2/', 'b2')


    # generate_nups_task_scripts(
    #     algos = ['vanilla_ppo'],
    #     betas = [10, 25, 50, 100, 200, 400, 800],
    #     target_sh_name = 'test_nups_vanilla_ppo.sh',
    #     root_dir = 'exps',
    #     envs = ['Hopper-v5', 'HalfCheetah-v5', 'Walker2d-v5'][1:]
    # )

    main()

    # root_dir = './test_nups_vanilla_ppo_b1_approx_tdadv/'
    # print_nups_res(root_dir, 'b1')