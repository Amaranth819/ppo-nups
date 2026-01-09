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


def run_statewise_attack(p: Trainer, statewise_eps: list, max_len=2048):
    # Arrays to be updated with historic info
    envs = p.envs
    initial_states = p.reset_envs(envs)
    if hasattr(p, "imit_network"):
        p.imit_network.reset()
    p.policy_model.reset()
    p.val_model.reset()

    # Holds information (length and true reward) about completed episodes
    completed_episode_info = []

    shape = (1, max_len)
    rewards = torch.zeros(shape)

    actions_shape = shape + (p.NUM_ACTIONS,)
    actions = torch.zeros(actions_shape)
    # Mean of the action distribution. Used for avoid unnecessary recomputation.
    action_means = torch.zeros(actions_shape)

    states_shape = (1, max_len + 1) + initial_states.shape[2:]
    states = torch.zeros(states_shape)

    iterator = range(max_len)

    states[:, 0, :] = initial_states
    last_states = states[:, 0, :]

    # Convert statewise_eps into tensor
    assert len(statewise_eps) == last_states.size(-1)
    statewise_eps_ts = torch.as_tensor(
        statewise_eps, dtype=torch.float32, device=last_states.device
    )
    statewise_eps_ts = statewise_eps_ts.unsqueeze(0)

    for t in iterator:
        # if (t+1) % 100 == 0:
        #     print('Step {} '.format(t+1))
        # assert shape_equal([self.NUM_ACTORS, self.NUM_FEATURES], last_states)
        # Retrieve probabilities
        # action_pds: (# actors, # actions), prob dists over actions
        # next_actions: (# actors, 1), indices of actions

        # pause updating hidden state because the attack may inference the model.
        p.policy_model.pause_history()
        p.val_model.pause_history()
        if hasattr(p, "imit_network"):
            p.imit_network.pause_history()

        # Attack the state inputs
        # signs = torch.where(torch.rand_like(statewise_eps_ts) >= 0.5, torch.ones_like(statewise_eps_ts), -torch.ones_like(statewise_eps_ts))
        signs = torch.ones_like(statewise_eps_ts)
        maybe_attacked_last_states = signs * statewise_eps_ts + last_states

        p.policy_model.continue_history()
        p.val_model.continue_history()
        if hasattr(p, "imit_network"):
            p.imit_network.continue_history()

        action_pds = p.policy_model(maybe_attacked_last_states)
        if hasattr(p, "imit_network"):
            _ = p.imit_network(maybe_attacked_last_states)

        next_action_means, next_action_stds = action_pds
        # # Double check if the attack is within eps range.
        # if p.params.ATTACK_METHOD != "none":
        #     max_eps = (maybe_attacked_last_states - last_states).abs().max()
        #     attack_eps = float(p.params.ROBUST_PPO_EPS) if p.params.ATTACK_EPS == "same" else float(p.params.ATTACK_EPS)
        #     if max_eps > attack_eps + 1e-5:
        #         raise RuntimeError(f"{max_eps} > {attack_eps}. Attack implementation has bug and eps is not correctly handled.")
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
            (rewards, next_rewards),
            (actions, next_actions),  # The sampled actions.
            (action_means, next_action_means),  # The sampled actions.
            (states, next_states),
        ]

        last_states = next_states[:, 0, :]
        for total, v in pairs:
            if total is states:
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

    if len(completed_episode_info) > 0:
        ep_length, ep_reward = completed_episode_info[0]
    else:
        ep_length = np.nan
        ep_reward = np.nan

    actions = actions[0][: t + 1]
    action_means = action_means[0][: t + 1]
    states = states[0][: t + 1]

    to_ret = (
        ep_length,
        ep_reward,
        actions.detach().cpu().numpy(),
        action_means.detach().cpu().numpy(),
        states.detach().cpu().numpy(),
    )
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

    # if params['sqlite_path']:
    #     print(f"Will save results in sqlite database in {params['sqlite_path']}")
    #     connection = sqlite3.connect(params['sqlite_path'])
    #     cur = connection.cursor()
    #     cur.execute('''create table if not exists attack_results
    #           (method varchar(20),
    #           mean_reward real,
    #           std_reward real,
    #           min_reward real,
    #           max_reward real,
    #           sarsa_eps real,
    #           sarsa_reg real,
    #           sarsa_steps integer,
    #           deterministic bool,
    #           early_terminate bool)''')
    #     connection.commit()
    #     # We will set this flag to True we break early.
    #     early_terminate = False

    # # Append a prefix for output path.
    # if params['out_dir_prefix']:
    #     params['out_dir'] = os.path.join(params['out_dir_prefix'], params['out_dir'])
    #     print(f"setting output dir to {params['out_dir']}")

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


def run_test(p: Trainer, statewise_attack_eps=0.2, num_episodes=50):
    envs = p.envs
    initial_states = p.reset_envs(envs)
    num_statedims = initial_states.shape[-1]

    save_result_dict = {}

    for state_idx in range(num_statedims):
        statewise_eps = [0 for _ in range(initial_states.shape[-1])]
        statewise_eps[state_idx] = statewise_attack_eps

        all_rewards = []
        all_lens = []

        for i in range(num_episodes):
            print("Episode %d / %d" % (i + 1, num_episodes))
            # ep_length, ep_reward, actions, action_means, states, kl_certificates = p.run_test(compute_bounds=params['compute_kl_cert'], use_full_backward=params['use_full_backward'], original_stdev=original_stdev)
            ep_length, ep_reward, actions, action_means, states = run_statewise_attack(
                p, statewise_eps, max_len=2048
            )
            if i == 0:
                all_actions = actions.copy()
                all_states = states.copy()
            else:
                all_actions = np.concatenate((all_actions, actions), axis=0)
                all_states = np.concatenate((all_states, states), axis=0)

            all_rewards.append(ep_reward)
            all_lens.append(ep_length)
            # Current step mean, std, min and max
            mean_reward, std_reward, min_reward, max_reward = (
                np.mean(all_rewards),
                np.std(all_rewards),
                np.min(all_rewards),
                np.max(all_rewards),
            )

        mean_reward, std_reward, min_reward, max_reward = (
            np.mean(all_rewards),
            np.std(all_rewards),
            np.min(all_rewards),
            np.max(all_rewards),
        )

        print("\n")
        print("all rewards:", all_rewards)
        print(
            "rewards stats:\nmean: {}, std:{}, min:{}, max:{}".format(
                mean_reward, std_reward, min_reward, max_reward
            )
        )

        save_result_dict[state_idx] = {
            "all_rewards": all_rewards,
            "all_lens": all_lens,
        }

    return save_result_dict


def result_to_latex(root_path = './'):
    import os
    result_dict_paths = glob.glob(os.path.join(root_path, "statewise_attack_hopper*.json"))

    all_results = defaultdict(lambda: [])
    num_state_dims = -1

    for path in result_dict_paths:
        with open(path, "r") as f:
            data_dict = json.load(f)
        path = path.split('/')[-1]
        eps_val = float(eval(path.split("_")[3].split("eps")[1]))
        num_state_dims = len(data_dict.items())
        for state_dim_idx, single_result in data_dict.items():
            all_results[eps_val, int(eval(state_dim_idx))].extend(single_result["all_rewards"])

    all_eps_vals = list(sorted(list(set([k[0] for k in all_results.keys()]))))

    latex_data = {}
    latex_data["$\epsilon$"] = all_eps_vals
    for state_idx in range(num_state_dims):
        latex_data[state_idx] = []
        for eps in all_eps_vals:
            result = all_results[eps, state_idx]
            # result = list(filter(lambda x: x > 1000, result))
            single_result_mean = np.mean(result).astype(int)
            single_result_std = np.std(result).astype(int)
            latex_data[state_idx].append(
                f"{single_result_mean:d} $\pm$ {single_result_std:d}"
            )

    df = pd.DataFrame(latex_data)
    # df.drop(columns = df.columns[0])
    latex_str = df.to_latex(index = False)
    print(latex_str)


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


    parser.add_argument('--statewise-attack-eps', type = float, default = 0.1)
    parser.add_argument('--statewise-attack-epsisodes', type = int, default = 50)

    parser = add_common_parser_opts(parser)

    return parser


if __name__ == "__main__":
    # parser = get_parser()
    # args = parser.parse_args()
    # if args.load_model:
    #     assert args.config_path, (
    #         "Need to specificy a config file when loading a pretrained model."
    #     )

    # if args.early_terminate:
    #     assert args.sqlite_path != "", (
    #         "Need to specify --sqlite-path to terminate early."
    #     )

    # if args.sarsa_enable:
    #     if args.sqlite_path != "":
    #         print(
    #             "When --sarsa-enable is specified, --sqlite-path and --early-terminate will be ignored."
    #         )

    # params = vars(args)
    # seed = params["seed"]

    # torch.manual_seed(seed)
    # np.random.seed(seed)

    # statewise_attack_eps = args.statewise_attack_eps
    # num_test_episodes = args.statewise_attack_epsisodes
    # p, params = load_trainer(params)
    # save_result_dict = run_test(p, statewise_attack_eps, num_episodes=num_test_episodes)

    # save_result_dict_path = f"statewise_attack_{params['game'][:-3].lower()}_eps{statewise_attack_eps}_{params['exp_id']}_results.json"
    # with open(save_result_dict_path, "w") as f:
    #     json.dump(save_result_dict, f, indent=4)
    
    # Save the results as a latex table after finishing all experiments.
    result_to_latex('./statewise_attack')