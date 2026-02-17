import argparse
import copy
import glob
import json
import logging
import os
import numpy as np
import pandas as pd
import torch
from collections import defaultdict
from cox.store import Store
from policy_gradients.agent import Trainer
from run import add_common_parser_opts, override_json_params

logging.disable(logging.INFO)
np.set_printoptions(suppress=True, precision=2)
torch.set_printoptions(precision = 4, sci_mode = False)


import numpy as np
import torch

def estimate_maxa_absA_cmaes(
    advantage_net, state, action_space, 
    num_samples=64, num_elites=16, max_iters=10, 
    sigma_init=0.5
):
    """
    Estimates max_a |A^\pi(s,a)| using a simplified Covariance Matrix Adaptation Evolution Strategy (CMA-ES).
    
    Args:
        advantage_net: The pretrained Advantage network A^\pi(s,a)
        state: The current state tensor (1, state_dim)
        action_dim: Dimension of the action space
        pop_size: Number of candidate actions per iteration
        max_iters: Number of generations to evolve
        sigma_init: Initial step size (standard deviation)
    """
    device = state.device
    action_dim = action_space.shape[0]
    action_low = torch.from_numpy(action_space.low).float().to(device)
    action_high = torch.from_numpy(action_space.high).float().to(device)

    # Initialize CMA-ES parameters
    mean = torch.zeros(action_dim, dtype=torch.float32, device=device)
    cov = torch.eye(action_dim, dtype=torch.float32, device=device)
    sigma = sigma_init
    
    # Selection parameters
    weights = np.log(num_elites + 0.5) - torch.log(torch.arange(1, num_elites + 1).to(device))
    weights /= torch.sum(weights)

    best = -float('inf')

    with torch.no_grad():
        for i in range(max_iters):
            dist = torch.distributions.MultivariateNormal(mean, cov * sigma**2)
            samples = dist.sample((num_samples,))
            actions_sampled = torch.clamp(samples, min=action_low, max=action_high)

            # 2. Evaluate advantages for all sampled actions
            state_batch = state.repeat(num_samples, 1)
            
            # Calculate |A(s, a)|
            advantages = advantage_net(torch.cat([state_batch, actions_sampled], -1)).squeeze(-1)
            abs_advantages = torch.abs(advantages)

            # 3. Sort and select elite candidates
            indices = torch.argsort(abs_advantages) # Aescending
            elites = actions_sampled[indices[-num_elites:]]
            
            current_best = abs_advantages[indices[-1]].item()
            if current_best > best:
                best = current_best

            # 4. Update Mean: Weighted average of elites
            old_mean = mean.clone()
            mean = weights @ elites
            
            # 5. Update Covariance: Estimate new shape from elites
            # This is a simplified rank-mu update
            centered_elites = elites - old_mean.unsqueeze(0)
            new_cov = torch.zeros_like(cov)
            for j in range(num_elites):
                new_cov += weights[j] * torch.outer(centered_elites[j], centered_elites[j])
            
            # Add a small regularization to keep cov positive definite
            cov = 0.8 * cov + 0.2 * (new_cov / (sigma**2) + 1e-6 * torch.eye(action_dim).to(device))
            
            # 6. Step size control (simple decay for this implementation)
            sigma *= 0.95 

            print(f'Iter {i}: max_a |A| = {best:.5f}')

    return best



nuus_params = [
    'nuus_max_d_kl',
    'nuus_beta',
    'nuus_num_iterations_cem',
    'nuus_num_sampled_actions_cem',
    'nuus_num_elite_actions_cem',
    'nuus_ub_type',
    'nuus_solver',
    'nuus_num_sampled_actions_fim'
]


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
            excluded_params + sarsa_params + imit_params + nuus_params
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
        p = Trainer.agent_from_params(params, store = None)
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
        p, _ = Trainer.agent_from_data(
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
    if params["deterministic"]:
        # print('Policy runs in deterministic mode. Ignoring Gaussian noise.')
        p.policy_model.log_stdev.data[:] = -100
    print('Gaussian noise in policy (after adjustment):')
    print(torch.exp(p.policy_model.log_stdev))

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

    # NUUS settings
    # Set ATTACK_EPS = 'nuus' to use the following parameters
    parser.add_argument('--nuus_max_d_kl', type=str, default='auto')
    parser.add_argument('--nuus_beta', type=float, default=100)
    parser.add_argument('--nuus_num_iterations_cem', type=int, default=1)
    parser.add_argument('--nuus_num_sampled_actions_cem', type=int, default=32)
    parser.add_argument('--nuus_num_elite_actions_cem', type=int, default=8)
    parser.add_argument('--nuus_ub_type', type=int, choices=[1], default=1)
    parser.add_argument('--nuus_num_sampled_actions_fim', type=int, default=1)
    parser.add_argument('--nuus_solver', type=str, choices=['approx', 'cvxpy'], default='approx')

    parser = add_common_parser_opts(parser)

    return parser



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

    # 
    # params['attack_eps'] = 'nuus'

    p, params = load_trainer(params)
    num_sampled_episodes = 10

    final_results = {}
    final_results['results'] = {}
    final_results['params'] = {key : params[key] for key in nuus_params}

    for attack_method in ['none', 'critic', 'random', 'action', 'sarsa'][:1]:
        p.params.ATTACK_METHOD = attack_method
        if attack_method == 'sarsa':
            p.params.ATTACK_SARSA_NETWORK = os.path.join(p.store.path, 'sarsa.model')

        sampled_ep_rewards = []
        sampled_ep_returns = []
        sampled_ep_lengths = []
        while len(sampled_ep_rewards) < num_sampled_episodes:
            ep_length, ep_reward, ep_return, ep_ori_return, actions, action_means, states, kl_upper_bound, _, ep_discounted_return = p.run_test(max_len = 1000)
            if ep_reward != np.nan:
                sampled_ep_rewards.append(ep_reward)
                sampled_ep_returns.append(ep_return)
                sampled_ep_lengths.append(ep_length)
        print(f'Attack method = {p.params.ATTACK_METHOD}')
        print(f'avg of {num_sampled_episodes} episodes: \
              ep_reward = {np.mean(sampled_ep_rewards):.4f} +- {np.std(sampled_ep_rewards):.4f} | \
              ep_return = {np.mean(sampled_ep_returns):.4f} +- {np.std(sampled_ep_returns):.4f} | \
              ep_length = {np.mean(sampled_ep_lengths):.4f} +- {np.std(sampled_ep_lengths):.4f}'
        )

        final_results['results'][attack_method] = {
            'ep_reward' : sampled_ep_rewards,
            'ep_return' : sampled_ep_returns,
            'ep_length' : sampled_ep_lengths,
        }

    with open(os.path.join(params['out_dir'], params['exp_id'], 'eval_results.json'), 'w') as f:
        json.dump(final_results, f, indent=4)



if __name__ == "__main__":
    # main()

    import gymnasium as gym
    env = gym.make('Hopper-v5')
    obs_space = env.observation_space
    action_space = env.action_space

    net = torch.nn.Linear(obs_space.shape[0] + action_space.shape[0], 1)
    state = torch.randn(1, obs_space.shape[0])

    best = estimate_maxa_absA_cmaes(net, state, action_space)
    print(best)