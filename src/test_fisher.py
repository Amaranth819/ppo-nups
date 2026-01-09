import argparse
import copy
import glob
import json
import logging
from collections import defaultdict

import numpy as np
import pandas as pd
import torch

from cox.store import Store
from policy_gradients.agent import Trainer
from run import add_common_parser_opts, override_json_params

logging.disable(logging.INFO)
np.set_printoptions(suppress=True, precision=2)


def test_fisher(p: Trainer):
    # Arrays to be updated with historic info
    envs = p.envs
    initial_states = p.reset_envs(envs)
    if hasattr(p, "imit_network"):
        p.imit_network.reset()
    p.policy_model.reset()
    p.val_model.reset()

    # Holds information (length and true reward) about completed episodes
    completed_episode_info = []

    max_len = 1024
    shape = (1, max_len)
    all_actions = torch.zeros(shape + (p.NUM_ACTIONS,))
    all_action_means = torch.zeros_like(all_actions)
    all_action_stds = torch.zeros_like(all_actions)
    all_states = torch.zeros((1, max_len + 1) + initial_states.shape[2:])
    all_rewards = torch.zeros(shape)

    all_states[:, 0, :] = initial_states
    last_states = all_states[:, 0, :]

    for t in range(max_len):
        # pause updating hidden state because the attack may inference the model.
        p.policy_model.pause_history()
        p.val_model.pause_history()
        if hasattr(p, "imit_network"):
            p.imit_network.pause_history()

        # Attack the state inputs
        maybe_attacked_last_states = last_states

        p.policy_model.continue_history()
        p.val_model.continue_history()
        if hasattr(p, "imit_network"):
            p.imit_network.continue_history()

        action_pds = p.policy_model(maybe_attacked_last_states)
        if hasattr(p, "imit_network"):
            _ = p.imit_network(maybe_attacked_last_states)

        next_action_means, next_action_stds = action_pds
        next_actions = p.policy_model.sample(action_pds)

        # if discrete, next_actions is (# actors, 1)
        # otw if continuous (# actors, 1, action dim)
        next_actions = next_actions.unsqueeze(1)

        ret = p.multi_actor_step(next_actions, envs)

        # done_info = List of (length, reward) pairs for each completed trajectory
        # (next_rewards, next_states, next_dones) act like multi-actor env.step()
        done_info, next_rewards, next_states, next_not_dones = ret
        # Reset the policy (if the policy has memory if we are done)
        if next_not_dones.item() == 0:
            p.policy_model.reset()
            p.val_model.reset()

        # Update histories
        # each shape: (nact, t, ...) -> (nact, t + 1, ...)
        pairs = [
            (all_rewards, next_rewards),
            (all_actions, next_actions),  # The sampled actions.
            (all_action_means, next_action_means),  # The sampled actions.
            (all_action_stds, next_action_stds),
            (all_states, next_states),
        ]

        last_states = next_states[:, 0, :]
        for total, v in pairs:
            if total is all_states:
                # Next states, stores in the next position.
                total[:, t + 1] = v
            else:
                # The current action taken, and reward received.
                total[:, t] = v

        # If some of the actors finished AND this is not the last step
        # OR some of the actors finished AND we have no episode information
        if len(done_info) > 0:
            completed_episode_info.extend(done_info)
            break

    all_actions = all_actions[0][: t + 1]
    all_action_means = all_action_means[0][: t + 1]
    all_action_stds = all_action_stds[0][: t + 1]
    all_states = all_states[0][: t + 1]
    all_rewards = all_rewards[0][: t + 1]

    to_ret = (all_states, all_actions, all_action_means, all_action_stds, all_rewards)
    return to_ret


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
            params, json_params, excluded_params + sarsa_params + imit_params + ["statewise_attack_eps", "statewise_attack_epsisodes"]
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
        p = Trainer.agent_from_params(params, store=None)
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
    # print('Gaussian noise in policy (after adjustment):')
    # print(torch.exp(p.policy_model.log_stdev))

    return p, params


def run_test(p: Trainer):
    import time

    to_ret = test_fisher(p)
    all_states, all_actions, all_action_means, all_action_stds, all_rewards = to_ret

    start = time.time()

    from collections import defaultdict
    ranks = 11
    topn_eig_indices = {}
    for r in range(ranks):
        topn_eig_indices[r+1] = defaultdict(lambda: 0)

    F = torch.zeros((all_states.size(-1), all_states.size(-1)), dtype = torch.float32, device = 'cuda')
    
    for i in range(all_states.size(0)):
        select_state = all_states[i]
        select_state.requires_grad_(True)
        mean, std = p.policy_model(select_state)
        pd = torch.distributions.Normal(mean, std)

        num_samples = 10
        for _ in range(num_samples):
            # sampled_action = pd.sample()
            # log_prob = pd.log_prob(sampled_action).sum()
            log_prob = pd.log_prob(all_actions[i]).sum()
            if select_state.grad is not None:
                select_state.grad.zero_()
            grads = torch.autograd.grad(log_prob, select_state, retain_graph = True)[0]
            # grads = grads.unsqueeze(-1)
            # F.add_((grads @ grads.T).detach())
            F.add_(torch.outer(grads, grads).detach())
    F.divide_(all_states.size(0) * num_samples)
    F = 0.5 * (F + F.T)
    eig_vals, eig_vecs = torch.linalg.eigh(F)
    min_eig = eig_vals.min().item()
    if min_eig < 1e-5:
        reg = max(1e-5, -min_eig + 1e-5)
        F = F + reg * torch.eye(select_state.size(-1))

    eigs, _ = torch.linalg.eig(F)
    eigs = eigs.real
    # eigs = torch.linalg.eigvals(F).real
    # eigs = torch.linalg.eigvalsh(F)
    # eigs, Q = torch.linalg.eigh(F)
    # eigs = torch.clamp(eigs, min = 0)
    # eigs = torch.linalg.svdvals(F)
    # print(i, eigs.min(), eigs.max())
    print(i, eigs)
    print(i, eigs.sort(descending = True))

    # # Print the sorted eigenvalues
    # sorted_eigs = eigs.sort(descending = True)
    # for rank in range(ranks):
    #     topn_eig_indices[rank+1][sorted_eigs[-1][rank].item()] += 1
    # print(i, sorted_eigs[0][:ranks].cpu().numpy().astype(float), sorted_eigs[1][:ranks].cpu().numpy())

    # # Calculate the number of non-zero eigenvalues
    # eigs = eigs.float()
    # eigs = torch.clamp(eigs, min = 0)
    # eigs = torch.where(eigs < 1e-4, torch.zeros_like(eigs), eigs)
    # nonzero_eigs = torch.nonzero(eigs).size(0)
    # print(i, nonzero_eigs)

    # # Omega = Cholesky decomposition
    # Omega = torch.linalg.cholesky(F, upper = False).T
    # print(i, Omega)

    # print(i, eigs.cpu().numpy().astype(float))

    end = time.time()
    # print(end - start, 'seconds')
    for rank, indices in topn_eig_indices.items():
        indices = dict(indices)
        indices = list(sorted(indices.items(), key=lambda item: item[1], reverse=True))
        print(f'Rank={rank}', indices)


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

    parser = add_common_parser_opts(parser)

    return parser


if __name__ == "__main__":
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
    run_test(p)