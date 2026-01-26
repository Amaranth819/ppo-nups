import mujoco
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import argparse
import copy
import os
import glob
import json
from cox.store import Store
from typing import Union

from run import add_common_parser_opts, override_json_params
from policy_gradients.agent import Trainer, TrainerWithNUPS


# Unwrap the environment to access mjData and mjModel
import mujoco



def snapshot_mjdata(
    uenv # the unwrapped environment
):
    data = uenv.data
    return {
        "qpos": data.qpos.copy(),
        "qvel": data.qvel.copy(),
        "qacc": data.qacc.copy(),
        "act": data.act.copy(),
        "ctrl": data.ctrl.copy(),
        "time": float(data.time),
        "mocap_pos": data.mocap_pos.copy(),
        "mocap_quat": data.mocap_quat.copy(),
    }


def restore_mjdata(
    uenv, # the unwrapped environment
    snap # the stored mjdata using snapshot_mjdata
):
    uenv.data.qpos[:] = snap["qpos"]
    uenv.data.qvel[:] = snap["qvel"]
    uenv.data.qacc[:] = snap["qacc"]
    uenv.data.act[:]  = snap["act"]
    uenv.data.ctrl[:] = snap["ctrl"]
    uenv.data.time = snap["time"]
    uenv.data.mocap_pos[:] = snap["mocap_pos"]
    uenv.data.mocap_quat[:] = snap["mocap_quat"]
    mujoco.mj_forward(uenv.model, uenv.data)






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

    parser.add_argument('--num_iterations', type = int, default = 3)
    parser.add_argument('--num_sampled_actions', type = int, default = 32)
    parser.add_argument('--num_elite_actions', type = int, default = 8)
    parser.add_argument('--num_sampled_episodes', type = int, default = 5)
    parser.add_argument('--save_result_root_dir', type = str, default = "estimated_abs_A_max")

    parser = add_common_parser_opts(parser)

    return parser




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
            excluded_params + sarsa_params + imit_params + ['num_iterations', 'num_sampled_actions', 'num_elite_actions', 'num_sampled_episodes', 'save_result_root_dir']
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
        # p = TrainerWithNUPS.agent_from_params(params, store = None)
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
        # p, _ = TrainerWithNUPS.agent_from_data(
        #     store,
        #     row,
        #     cpu,
        #     extra_params=params,
        #     override_params=override_params,
        #     excluded_params=excluded_params,
        # )
        store.close()

    # print('Gaussian noise in policy:')
    # print(torch.exp(p.policy_model.log_stdev))
    original_stdev = p.policy_model.log_stdev.clone().detach()
    if params["noise_factor"] != 1.0:
        p.policy_model.log_stdev.data[:] += np.log(params["noise_factor"])
    if params["deterministic"]:
        print('Policy runs in deterministic mode. Ignoring Gaussian noise.')
        p.policy_model.log_stdev.data[:] = -100
    # print('Gaussian noise in policy (after adjustment):')
    # print(torch.exp(p.policy_model.log_stdev))

    return p, params



def estimate_abs_A_max(
    p : Trainer, # Union[Trainer, TrainerWithNUPS], # Trainer
    num_iterations : int, # The number of CEM iterations for a given state.
    num_sampled_actions : int, # The number of sampled actions within an iteration.
    num_elite_actions : int, # The number of elite actions for updating Gaussian.
    num_sampled_episodes : int,
    verbose = True,
):
    # Arrays to be updated with historic info
    if hasattr(p, "imit_network"):
        p.imit_network.reset()
    p.policy_model.reset()
    p.val_model.reset()

    device = None
    if p.params['cpu']:
        device = 'cpu'
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    gamma = p.params['gamma']

    # Unwrap the environment to access the underlying MuJoCo objects
    uenv = p.envs[0]
    uenv.env = uenv.env.unwrapped
    
    max_episode_length = 1000

    action_space_size = uenv.env.action_space.shape[0]
    action_low, action_high = uenv.env.action_space.low, uenv.env.action_space.high

    abs_A_max = -1e5

    for e in range(num_sampled_episodes):
        if hasattr(p, "imit_network"):
            p.imit_network.reset()
        p.policy_model.reset()
        p.val_model.reset()
        
        obs = uenv.reset()
        ep_total_rewards = []

        for t in range(max_episode_length):
            snap = snapshot_mjdata(uenv.env)

            with torch.no_grad():
                obs_ts = torch.from_numpy(obs).float().to(device)
                V_s = p.val_model(obs_ts).flatten().item()

            # Initialize Gaussian
            mu = np.zeros(shape = (1, action_space_size))
            sigma = np.ones(shape = (1, action_space_size)) * 1

            for iter in range(num_iterations):
                sampled_actions = np.random.randn(num_sampled_actions, action_space_size) * sigma + mu
                sampled_actions = np.clip(sampled_actions, action_low, action_high)

                collected_advs = []
                for action in sampled_actions:
                    next_obs, rew, done, info = uenv.step(action)
                    next_obs_ts = torch.from_numpy(next_obs).float().to(device)

                    # Calculate the advantage
                    with torch.no_grad():
                        V_s_next = p.val_model(next_obs_ts).flatten().item()
                        
                    adv = rew + gamma * V_s_next - V_s
                    # adv = info['step_reward'] + gamma * V_s_next - V_s
                    collected_advs.append(adv)
                    
                    restore_mjdata(uenv.env, snap)

                # Update abs_A_max
                abs_advs = np.abs(np.array(collected_advs))
                abs_A_max = max(abs_A_max, np.max(abs_advs))

                # Select elite actions and update Gaussian
                elite_indices = abs_advs.argsort()[-num_elite_actions:]
                elite_actions = sampled_actions[elite_indices]
                mu = np.mean(elite_actions, axis = 0, keepdims = True)
                sigma = np.std(elite_actions, axis = 0, keepdims = True) + 1e-6


            restore_mjdata(uenv.env, snap)

            action_pds = p.policy_model(obs_ts)
            next_actions = p.policy_model.sample(action_pds).detach().cpu().numpy()
            obs, rew, done, info = uenv.step(next_actions)
            ep_total_rewards.append(info['step_reward'])

            if done:
                break

        ep_length = t + 1
        ep_total_rewards = sum(ep_total_rewards)

        if verbose:
            print(f'Episode {e}: Rewards {ep_total_rewards:.4f} | Length: {int(ep_length):d} | |A|_max: {abs_A_max:.4f}')

    return abs_A_max



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
    p.params.ATTACK_METHOD = "none"

    # for _ in range(10):
    #     ep_length, ep_reward, _, _, _, = p.run_test_trajectories(max_len = 1024)
    #     print(ep_length, ep_reward)
    # return

    abs_A_max = estimate_abs_A_max(
        p, 
        params['num_iterations'],
        params['num_sampled_actions'],
        params['num_elite_actions'],
        params['num_sampled_episodes']    
    )

    if params['save_result_root_dir'] != "":
        if not os.path.exists(params['save_result_root_dir']):
            os.makedirs(params['save_result_root_dir'])

        result_file_name = f"{params['game']}_{params['exp_id']}.json"
        result_content = {
            'game' : params['game'],
            'exp_id' : params['exp_id'],
            'seed' : params['seed'],
            'num_iterations' : params['num_iterations'],
            'num_sampled_actions' : params['num_sampled_actions'],
            'num_elite_actions' : params['num_elite_actions'],
            'num_sampled_episodes' : params['num_sampled_episodes'],
            'abs_A_max' : abs_A_max
        }

        with open(os.path.join(params['save_result_root_dir'], result_file_name), 'w') as json_file:
            json.dump(result_content, json_file, indent=4) # 'indent=4' makes the file human-readable



def generate_batch_tasks(
    algos = ['vanilla_ppo'],
    target_sh_name = 'advantage_est.sh',
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
                    task_str = f'python -u src/advantage_est.py --config-path src/config_{json_env}_{algo}.json --exp-id {exp_id} --deterministic \n'
                    all_tasks.append(task_str)

    with open(target_sh_name, 'w') as f:
        f.writelines(all_tasks)
    os.chmod(target_sh_name, 0o777)


if __name__ == '__main__':
    main()
    # generate_batch_tasks()