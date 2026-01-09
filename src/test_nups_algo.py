import argparse
import copy
import glob
import json
import logging
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import tqdm

from cox.store import Store
from policy_gradients.agent import Trainer
from run import add_common_parser_opts, override_json_params

logging.disable(logging.INFO)
np.set_printoptions(suppress=True, precision=2)
torch.set_printoptions(precision = 4, sci_mode = False)

import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, '../'))
from src.policy_gradients.models import *

import time
from src.policy_gradients.convex_relaxation import get_kl_bound as get_state_kl_bound


def calculate_total_discounted_return(ep_reward_list, gamma):
    polys = np.arange(len(ep_reward_list))
    return np.sum(gamma**polys * np.array(ep_reward_list))


class TrainerWithNUPS(Trainer):
    def __init__(self, policy_net_class, value_net_class, params, store, advanced_logging=True, log_every=5):
        super().__init__(policy_net_class, value_net_class, params, store, advanced_logging, log_every)


    @staticmethod
    def agent_from_params(params, store=None):
        if params['history_length'] > 0:
            agent_policy = CtsLSTMPolicy
            if params['use_lstm_val']:
                agent_value = ValueLSTMNet
            else:
                agent_value = value_net_with_name(params['value_net_type'])
        else:
            agent_policy = policy_net_with_name(params['policy_net_type'])
            agent_value = value_net_with_name(params['value_net_type'])

        advanced_logging = params['advanced_logging'] and store is not None
        log_every = params['log_every'] if store is not None else 0

        if params['cpu']:
            torch.set_num_threads(1)
        p = TrainerWithNUPS(agent_policy, agent_value, params, store, log_every=log_every,
                    advanced_logging=advanced_logging)

        return p
    

    @staticmethod
    def agent_from_data(store, row, cpu, extra_params=None, override_params=None, excluded_params=None):
        '''
        Initializes an agent from serialized data (via cox)
        Inputs:
        - store, the name of the store where everything is logged
        - row, the exact row containing the desired data for this agent
        - cpu, True/False whether to use the CPU (otherwise sends to GPU)
        - extra_params, a dictionary of extra agent parameters. Only used
          when a key does not exist from the loaded cox store.
        - override_params, a dictionary of agent parameters that will override
          current agent parameters.
        - excluded_params, a dictionary of parameters that we do not copy or
          override.
        Outputs:
        - agent, a constructed agent with the desired initialization and
              parameters
        - agent_params, the parameters that the agent was constructed with
        '''

        ckpts = store['final_results']

        get_item = lambda x: list(row[x])[0]

        items = ['val_model', 'policy_model', 'val_opt', 'policy_opt']
        names = {i: get_item(i) for i in items}

        param_keys = list(store['metadata'].df.columns)
        param_values = list(store['metadata'].df.iloc[0,:])

        def process_item(v):
            try:
                return v.item()
            except:
                return v

        param_values = [process_item(v) for v in param_values]
        agent_params = {k:v for k, v in zip(param_keys, param_values)}

        if 'adam_eps' not in agent_params: 
            agent_params['adam_eps'] = 1e-5
        if 'cpu' not in agent_params:
            agent_params['cpu'] = cpu

        # Update extra params if they do not exist in current parameters.
        if extra_params is not None:
            for k in extra_params.keys():
                if k not in agent_params and k not in excluded_params:
                    print(f'adding key {k}={extra_params[k]}')
                    agent_params[k] = extra_params[k]
        if override_params is not None:
            for k in override_params.keys():
                if k not in excluded_params and override_params[k] is not None and override_params[k] != agent_params[k]:
                    print(f'overwriting key {k}: old={agent_params[k]}, new={override_params[k]}')
                    agent_params[k] = override_params[k]

        agent = TrainerWithNUPS.agent_from_params(agent_params)

        def load_state_dict(model, ckpt_name):
            mapper = torch.device('cuda:0') if not cpu else torch.device('cpu')
            state_dict = ckpts.get_state_dict(ckpt_name, map_location=mapper)
            model.load_state_dict(state_dict)

        load_state_dict(agent.policy_model, names['policy_model'])
        load_state_dict(agent.val_model, names['val_model'])
        if agent.ANNEAL_LR:
            agent.POLICY_SCHEDULER.last_epoch = get_item('iteration')
            agent.VALUE_SCHEDULER.last_epoch = get_item('iteration')
        load_state_dict(agent.POLICY_ADAM, names['policy_opt'])
        load_state_dict(agent.val_opt, names['val_opt'])
        agent.envs = ckpts.get_pickle(get_item('envs'))

        return agent, agent_params
    

    def apply_attack(self, last_states, nups_eps : torch.Tensor):
        # if self.params.ATTACK_RATIO < random.random():
        #     # Only attack a portion of steps.
        #     return last_states
        # eps = self.params.ATTACK_EPS
        # if eps == "same":
        #     eps = self.params.ROBUST_PPO_EPS
        # else:
        #     eps = float(eps)

        if len(nups_eps.size()) == 1:
            nups_eps = nups_eps.unsqueeze(0)
        elif len(nups_eps.size()) == 2:
            assert nups_eps.size(0) == 1
        else:
            raise ValueError
        assert nups_eps.min() > 0

        steps = self.params.ATTACK_STEPS
        if self.params.ATTACK_METHOD == "critic":
            # Find a state that is close the last_states and decreases value most.
            if steps > 0:
                if self.params.ATTACK_STEP_EPS == "auto":
                    # step_eps = eps / steps
                    step_eps = nups_eps / steps
                else:
                    step_eps = float(self.params.ATTACK_STEP_EPS)
                clamp_min = last_states - nups_eps
                clamp_max = last_states + nups_eps
                # Random start.
                # noise = torch.empty_like(last_states).uniform_(-step_eps, step_eps)
                noise = torch.empty_like(last_states).uniform_(-1, 1) * step_eps
                states = last_states + noise
                with torch.enable_grad():
                    for i in range(steps):
                        states = states.clone().detach().requires_grad_()
                        value = self.val_model(states).mean(dim=1)
                        value.backward()
                        update = states.grad.sign() * step_eps
                        # Clamp to +/- eps.
                        states.data = torch.min(torch.max(states.data - update, clamp_min), clamp_max)
                    self.val_model.zero_grad()
                return states.detach()
            else:
                return last_states
        elif self.params.ATTACK_METHOD == "random":
            # Apply an uniform random noise.
            # noise = torch.empty_like(last_states).uniform_(-eps, eps)
            noise = torch.empty_like(last_states).uniform_(-1, 1) * nups_eps
            return (last_states + noise).detach()
        elif self.params.ATTACK_METHOD == "action" or self.params.ATTACK_METHOD == "action+imit":
            if steps > 0:
                if self.params.ATTACK_STEP_EPS == "auto":
                    # step_eps = eps / steps
                    step_eps = nups_eps / steps
                else:
                    step_eps = float(self.params.ATTACK_STEP_EPS)
                # clamp_min = last_states - eps
                # clamp_max = last_states + eps
                clamp_min = last_states - nups_eps
                clamp_max = last_states + nups_eps
                # SGLD noise factor. We simply set beta=1.
                # noise_factor = np.sqrt(2 * step_eps)
                noise_factor = torch.sqrt(2 * step_eps)
                noise = torch.randn_like(last_states) * noise_factor
                # The first step has gradient zero, so add the noise and projection directly.
                states = last_states + noise.sign() * step_eps
                # Current action at this state.
                if self.params.ATTACK_METHOD == "action+imit":
                    if not hasattr(self, "imit_network") or self.imit_network == None:
                        assert self.params.imit_model_path != None
                        print('\nLoading imitation network for attack: ', self.params.imit_model_path)
                        # Setup imitation network
                        self.setup_imit(train=False)
                        imit_ckpt = torch.load(self.params.imit_model_path)
                        self.imit_network.load_state_dict(imit_ckpt['state_dict'])
                        self.imit_network.reset()
                        self.imit_network.pause_history()
                    old_action, old_stdev = self.imit_network(last_states)
                else:
                    old_action, old_stdev = self.policy_model(last_states)
                # Normalize stdev, avoid numerical issue
                old_stdev /= (old_stdev.mean())
                old_stdev = old_stdev.detach() # To prevent inplace operation!
                old_action = old_action.detach()
                with torch.enable_grad():
                # with torch.autograd.set_detect_anomaly(True):
                    for i in range(steps):
                        states = states.clone().detach().requires_grad_()
                        if self.params.ATTACK_METHOD == "action+imit":
                            action_change = (self.imit_network(states)[0] - old_action) / old_stdev
                        else:
                            action_change = (self.policy_model(states)[0] - old_action) / old_stdev
                        # action_change = (action_change * action_change).sum(dim=1)
                        action_change = (action_change * action_change).sum(dim=1).mean()
                        action_change.backward()

                        # Reduce noise at every step.
                        # noise_factor = np.sqrt(2 * step_eps) / (i+2)
                        noise_factor = torch.sqrt(2 * step_eps) / (i+2)
                        # Project noisy gradient to step boundary.
                        update = (states.grad + noise_factor * torch.randn_like(last_states)).sign() * step_eps
                        # Clamp to +/- eps.
                        states.data = torch.min(torch.max(states.data + update, clamp_min), clamp_max)
                    if self.params.ATTACK_METHOD == "action+imit": 
                        self.imit_network.zero_grad() 
                    self.policy_model.zero_grad()
                return states.detach()
            else:
                return last_states
        elif self.params.ATTACK_METHOD == "sarsa" or self.params.ATTACK_METHOD == "sarsa+action":
            # Attack using a learned value network.
            assert self.params.ATTACK_SARSA_NETWORK is not None
            use_action = self.params.ATTACK_SARSA_ACTION_RATIO > 0 and self.params.ATTACK_METHOD == "sarsa+action"
            action_ratio = self.params.ATTACK_SARSA_ACTION_RATIO
            assert action_ratio >= 0 and action_ratio <= 1
            if not hasattr(self, "sarsa_network"):
                self.sarsa_network = ValueDenseNet(state_dim=self.NUM_FEATURES+self.NUM_ACTIONS, init="normal")
                print("Loading sarsa network", self.params.ATTACK_SARSA_NETWORK)
                sarsa_ckpt = torch.load(self.params.ATTACK_SARSA_NETWORK)
                sarsa_meta = sarsa_ckpt['metadata']
                sarsa_eps = sarsa_meta['sarsa_eps'] if 'sarsa_eps' in sarsa_meta else "unknown"
                sarsa_reg = sarsa_meta['sarsa_reg'] if 'sarsa_reg' in sarsa_meta else "unknown"
                sarsa_steps = sarsa_meta['sarsa_steps'] if 'sarsa_steps' in sarsa_meta else "unknown"
                print(f"Sarsa network was trained with eps={sarsa_eps}, reg={sarsa_reg}, steps={sarsa_steps}")
                if use_action:
                    print(f"objective: {1.0 - action_ratio} * sarsa + {action_ratio} * action_change")
                else:
                    print("Not adding action change objective.")
                self.sarsa_network.load_state_dict(sarsa_ckpt['state_dict'])
            if steps > 0:
                if self.params.ATTACK_STEP_EPS == "auto":
                    step_eps = nups_eps / steps
                else:
                    step_eps = float(self.params.ATTACK_STEP_EPS)
                clamp_min = last_states - nups_eps
                clamp_max = last_states + nups_eps
                # Random start.
                # noise = torch.empty_like(last_states).uniform_(-step_eps, step_eps)
                noise = torch.empty_like(last_states).uniform_(-1, 1) * step_eps
                states = last_states + noise
                if use_action:
                    # Current action at this state.
                    old_action, old_stdev = self.policy_model(last_states)
                    old_stdev /= (old_stdev.mean())
                    old_action = old_action.detach()
                with torch.enable_grad():
                    for i in range(steps):
                        states = states.clone().detach().requires_grad_()
                        # This is the mean action...
                        actions = self.policy_model(states)[0]
                        value = self.sarsa_network(torch.cat((last_states, actions), dim=1)).mean(dim=1)
                        if use_action:
                            action_change = (actions - old_action) / old_stdev
                            # We want to maximize the action change, thus the minus sign.
                            action_change = -(action_change * action_change).mean(dim=1)
                            loss = action_ratio * action_change + (1.0 - action_ratio) * value
                        else:
                            action_change = 0.0
                            loss = value
                        loss.backward()
                        update = states.grad.sign() * step_eps
                        # Clamp to +/- eps.
                        states.data = torch.min(torch.max(states.data - update, clamp_min), clamp_max)
                    self.val_model.zero_grad()
                return states.detach()
            else:
                return last_states
        elif self.params.ATTACK_METHOD == "advpolicy":
            # # Attack using a learned policy network.
            # assert self.params.ATTACK_ADVPOLICY_NETWORK is not None
            # if not hasattr(self, "attack_policy_network"):
            #     self.attack_policy_network = self.policy_net_class(self.NUM_FEATURES, self.NUM_FEATURES,
            #                                      self.INITIALIZATION,
            #                                      time_in_state=self.VALUE_CALC == "time",
            #                                      activation=self.policy_activation)
            #     print("Loading adversary policy network", self.params.ATTACK_ADVPOLICY_NETWORK)
            #     advpolicy_ckpt = torch.load(self.params.ATTACK_ADVPOLICY_NETWORK)
            #     self.attack_policy_network.load_state_dict(advpolicy_ckpt['adversary_policy_model'])
            # # Unlike other attacks we don't need step or eps here.
            # # We don't sample and use deterministic adversary policy here.
            # perturbations_mean, _ = self.attack_policy_network(last_states)
            # # Clamp using tanh.
            # perturbed_states = last_states + ch.nn.functional.hardtanh(perturbations_mean) * eps
            # """
            # adv_perturbation_pds = self.attack_policy_network(last_states)
            # next_adv_perturbations = self.attack_policy_network.sample(adv_perturbation_pds)
            # perturbed_states = last_states + ch.tanh(next_adv_perturbations) * eps
            # """
            # return perturbed_states.detach()
            raise NotImplementedError
        elif self.params.ATTACK_METHOD == "none":
            return last_states
        else:
            raise ValueError(f'Unknown attack method {self.params.ATTACK_METHOD}')
        


    def run_test_trajectories(self, max_len, attack_with_nups = False, max_kl_div = 0, should_tqdm=False):
        # Arrays to be updated with historic info
        envs = self.envs
        initial_states = self.reset_envs(envs)
        if hasattr(self, "imit_network"):
            self.imit_network.reset()
        self.policy_model.reset()
        self.val_model.reset()

        # Holds information (length and true reward) about completed episodes
        completed_episode_info = []

        shape = (1, max_len)
        rewards = torch.zeros(shape)

        actions_shape = shape + (self.NUM_ACTIONS,)
        actions = torch.zeros(actions_shape)
        # Mean of the action distribution. Used for avoid unnecessary recomputation.
        action_means = torch.zeros(actions_shape)
        action_stds = torch.zeros_like(actions)
        rewards = torch.zeros(shape)
        not_dones = torch.zeros(shape, dtype = torch.bool)

        states_shape = (1, max_len+1) + initial_states.shape[2:]
        states =  torch.zeros(states_shape)

        iterator = range(max_len) if not should_tqdm else tqdm.trange(max_len)


        states[:, 0, :] = initial_states
        last_states = states[:, 0, :]
        
        for t in iterator:
            # if (t+1) % 100 == 0:
            #     print('Step {} '.format(t+1))
            # assert shape_equal([self.NUM_ACTORS, self.NUM_FEATURES], last_states)
            # Retrieve probabilities 
            # action_pds: (# actors, # actions), prob dists over actions
            # next_actions: (# actors, 1), indices of actions
            
            # pause updating hidden state because the attack may inference the model.
            self.policy_model.pause_history()
            self.val_model.pause_history()
            if hasattr(self, "imit_network"):
                self.imit_network.pause_history()
            
            if attack_with_nups:
                assert max_kl_div > 0
                nups_eps = calculate_nups(self, last_states, max_kl_div, num_sampled_actions = 1)
                maybe_attacked_last_states = self.apply_attack(last_states, nups_eps)
            else:
                maybe_attacked_last_states = last_states
            
            self.policy_model.continue_history()
            self.val_model.continue_history()
            if hasattr(self, "imit_network"):
                self.imit_network.continue_history()

            action_pds = self.policy_model(maybe_attacked_last_states)
            if hasattr(self, "imit_network"):
                _ = self.imit_network(maybe_attacked_last_states) 
            
            next_action_means, next_action_stds = action_pds
            # Double check if the attack is within eps range.
            if self.params.ATTACK_METHOD != "none":
                # max_eps = (maybe_attacked_last_states - last_states).abs().max()
                # attack_eps = float(self.params.ROBUST_PPO_EPS) if self.params.ATTACK_EPS == "same" else float(self.params.ATTACK_EPS)
                # if max_eps > attack_eps + 1e-5:
                #     raise RuntimeError(f"{max_eps} > {attack_eps}. Attack implementation has bug and eps is not correctly handled.")

                state_diff = (maybe_attacked_last_states - last_states).abs()
                if torch.any(state_diff > nups_eps + 1e-5):
                    print('state_diff = ', state_diff)
                    print('nups_eps = ', nups_eps)
                    raise RuntimeError('Attack implementation has bug and eps is not correctly handled.')

            
            if self.params.DETERMINISTIC:
                next_actions = next_action_means
            else:
                next_actions = self.policy_model.sample(action_pds)


            # if discrete, next_actions is (# actors, 1) 
            # otw if continuous (# actors, 1, action dim)
            next_actions = next_actions.unsqueeze(1).detach()

            ret = self.multi_actor_step(next_actions, envs)

            # done_info = List of (length, reward) pairs for each completed trajectory
            # (next_rewards, next_states, next_dones) act like multi-actor env.step()
            done_info, next_rewards, next_states, next_not_dones = ret
            # Reset the policy (if the policy has memory if we are done)
            if next_not_dones.item() == 0:
                self.policy_model.reset()
                self.val_model.reset()

            # Update histories
            # each shape: (nact, t, ...) -> (nact, t + 1, ...)

            pairs = [
                (rewards, next_rewards),
                (actions, next_actions), # The sampled actions.
                (action_means, next_action_means), # The sampled actions.
                (states, next_states),
                (action_stds, next_action_stds),
                (not_dones, next_not_dones)
            ]

            last_states = next_states[:, 0, :]
            for total, v in pairs:
                if total is states:
                    # Next states, stores in the next position.
                    total[:, t+1] = v
                else:
                    # The current action taken, and reward received.
                    total[:, t] = v
            
            # If some of the actors finished AND this is not the last step
            # OR some of the actors finished AND we have no episode information
            if len(done_info) > 0:
                completed_episode_info.extend(done_info)
                break

        if len(completed_episode_info) > 0:
            # ep_length, ep_reward = completed_episode_info[0]

            ep_length, ep_reward, ep_step_rewards = completed_episode_info[0]
        else:
            ep_length = np.nan
            ep_reward = np.nan

        actions = actions[0][:t+1]
        action_means = action_means[0][:t+1]
        states = states[0][:t+1]

        action_stds = action_stds[0][:t+1]
        rewards = rewards[0][:t+1]
        not_dones = not_dones[0][:t+1]

        values = self.val_model(states).squeeze(-1)
        advantages, returns = p.advantage_and_return(
            rewards = rewards.unsqueeze(0), 
            values = values.unsqueeze(0), 
            not_dones = not_dones.unsqueeze(0)
        )
        advantages = advantages.squeeze(0)

        ep_discounted_return = calculate_total_discounted_return(ep_step_rewards, self.params.GAMMA)

        to_ret = (ep_length, ep_reward, actions, action_means, states, advantages, ep_discounted_return)
        
        return to_ret
            

    def run_test(
        self, 
        max_len=2048, 
        attack_with_nups = False,
        max_kl_div = 0,
        compute_bounds=False, 
        use_full_backward=False, 
        original_stdev=None
    ):
        # print("-" * 80)
        start_time = time.time()
        if compute_bounds and not hasattr(self, "relaxed_policy_model"):
            self.create_relaxed_model()
        #saps, avg_ep_reward, avg_ep_length = self.collect_saps(num_saps=None, should_log=True, test=True, num_episodes=num_episodes)
        # with torch.no_grad():
        output = self.run_test_trajectories(max_len=max_len, attack_with_nups = attack_with_nups, max_kl_div = max_kl_div)
        ep_length, ep_reward, actions, action_means, states, advantages, ep_discounted_return = output
        # msg = "Episode reward: %f | episode length: %f"
        # print(msg % (ep_reward, ep_length))
        # if compute_bounds:
        #     if original_stdev is None:
        #         kl_stdev = torch.exp(self.policy_model.log_stdev)
        #     else:
        #         kl_stdev = torch.exp(original_stdev)
        #     eps = float(self.params.ROBUST_PPO_EPS) if self.params.ATTACK_EPS == "same" else float(self.params.ATTACK_EPS)
        #     kl_upper_bound = get_state_kl_bound(self.relaxed_policy_model, states, action_means,
        #             eps=eps, beta=0.0,
        #             stdev=kl_stdev, use_full_backward=use_full_backward).mean()
        #     kl_upper_bound = kl_upper_bound.item()
        # else:
        #     kl_upper_bound = float("nan")
        kl_upper_bound = float("nan")
        # Unroll the trajectories (actors, T, ...) -> (actors*T, ...)
        return ep_length, ep_reward, actions.detach().cpu().numpy(), action_means.detach().cpu().numpy(), states.detach().cpu().numpy(), kl_upper_bound, advantages.detach().cpu(), ep_discounted_return


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




# def collect(p: TrainerWithNUPS, max_len = 1024):
#     # Arrays to be updated with historic info
#     envs = p.envs
#     initial_states = p.reset_envs(envs)
#     if hasattr(p, "imit_network"):
#         p.imit_network.reset()
#     p.policy_model.reset()
#     p.val_model.reset()

#     # Holds information (length and true reward) about completed episodes
#     completed_episode_info = []

#     shape = (1, max_len)
#     all_actions = torch.zeros(shape + (p.NUM_ACTIONS,))
#     all_action_means = torch.zeros_like(all_actions)
#     all_action_stds = torch.zeros_like(all_actions)
#     all_states = torch.zeros((1, max_len + 1) + initial_states.shape[2:])
#     all_rewards = torch.zeros(shape)
#     all_not_dones = torch.zeros(shape, dtype = torch.bool)

#     all_states[:, 0, :] = initial_states
#     last_states = all_states[:, 0, :]

#     for t in range(max_len):
#         # pause updating hidden state because the attack may inference the model.
#         p.policy_model.pause_history()
#         p.val_model.pause_history()
#         if hasattr(p, "imit_network"):
#             p.imit_network.pause_history()

#         # Attack the state inputs
#         maybe_attacked_last_states = last_states

#         p.policy_model.continue_history()
#         p.val_model.continue_history()
#         if hasattr(p, "imit_network"):
#             p.imit_network.continue_history()

#         action_pds = p.policy_model(maybe_attacked_last_states)
#         if hasattr(p, "imit_network"):
#             _ = p.imit_network(maybe_attacked_last_states)

#         next_action_means, next_action_stds = action_pds

#         if p.params.DETERMINISTIC:
#             next_actions = next_action_means
#         else:
#             next_actions = p.policy_model.sample(action_pds)

#         # if discrete, next_actions is (# actors, 1)
#         # otw if continuous (# actors, 1, action dim)
#         next_actions = next_actions.unsqueeze(1)

#         ret = p.multi_actor_step(next_actions, envs)

#         # done_info = List of (length, reward) pairs for each completed trajectory
#         # (next_rewards, next_states, next_dones) act like multi-actor env.step()
#         done_info, next_rewards, next_states, next_not_dones = ret
#         # Reset the policy (if the policy has memory if we are done)
#         if next_not_dones.item() == 0:
#             p.policy_model.reset()
#             p.val_model.reset()

#         # Update histories
#         # each shape: (nact, t, ...) -> (nact, t + 1, ...)
#         pairs = [
#             (all_rewards, next_rewards),
#             (all_actions, next_actions),  # The sampled actions.
#             (all_action_means, next_action_means),  # The sampled actions.
#             (all_action_stds, next_action_stds),
#             (all_states, next_states),
#             (all_not_dones, next_not_dones)
#         ]

#         last_states = next_states[:, 0, :]
#         for total, v in pairs:
#             if total is all_states:
#                 # Next states, stores in the next position.
#                 total[:, t + 1] = v
#             else:
#                 # The current action taken, and reward received.
#                 total[:, t] = v

#         # If some of the actors finished AND this is not the last step
#         # OR some of the actors finished AND we have no episode information
#         if len(done_info) > 0:
#             completed_episode_info.extend(done_info)
#             break

#     all_actions = all_actions[0][: t + 1]
#     all_action_means = all_action_means[0][: t + 1]
#     all_action_stds = all_action_stds[0][: t + 1]
#     all_states = all_states[0][: t + 1]
#     all_rewards = all_rewards[0][: t + 1]
#     all_not_dones = all_not_dones[0][: t + 1]

#     all_values = p.val_model(all_states).squeeze(-1)
#     all_advantages, all_returns = p.advantage_and_return(
#         rewards = all_rewards.unsqueeze(0), 
#         values = all_values.unsqueeze(0), 
#         not_dones = all_not_dones.unsqueeze(0)
#     )
#     all_advantages = all_advantages.squeeze(0)

#     to_ret = (all_states, all_actions, all_action_means, all_action_stds, all_rewards, all_advantages)
#     return to_ret



def calculate_max_kl_div(
    gamma,
    beta,
    abs_A_max, 
    use_which_bound = 'b1'
):
    if use_which_bound == 'b1':
        return ((1 - gamma) * beta / abs_A_max)**2 / 2
    elif use_which_bound == 'b2':
        return ((1 - gamma) * (np.sqrt(1 + 4*gamma*beta/abs_A_max) - 1) / gamma)**2 / 8
    else:
        raise NotImplementedError
    


def method1(u, n, eps, D):
    omegas_vals = torch.zeros_like(u)
    for idx in range(n):
        omegas_vals[..., idx] = n * eps / np.sqrt(2 * D) * u[..., idx]
    return omegas_vals


def method2(F, n, eps, D, tol = 1e-4):
    F = torch.abs(F)
    # eigenvalues, eigenvectors = torch.linalg.eigh(F)
    # mask = eigenvalues > 1e-4
    # valid_eigenvalues = eigenvalues[mask]
    # valid_eigenvectors = eigenvectors[:, mask]
    # Q = valid_eigenvectors @ torch.diag(torch.sqrt(valid_eigenvalues))

    Q = torch.cholesky_solve(F)
    print(Q)

    C = 2 * D / (n * eps**2)
    S = torch.ones(size = (Q.size(-1),), dtype = torch.float32, device = Q.device)
    eye = torch.eye(Q.size(-1), dtype = torch.float32, device = Q.device)
    
    # S.requires_grad_(True)
    # optim = torch.optim.Adam([S], lr = 1e-3)
    
    # diff = 1e6
    # counter = 0
    # while diff > tol:
    #     # pred_S = C * Q.T @ (1 / (Q @ S))
    #     # loss = torch.mean(torch.abs(S - pred_S))
    #     # optim.zero_grad()
    #     # loss.backward()
    #     # optim.step()
    #     # diff = loss.item()
    #     # counter += 1

    #     print(counter, diff)
    #     print(S)
    #     print(pred_S)
    #     if counter >= 1:
    #         break

    max_iter = 100
    for it in range(max_iter):
        QS = Q @ S                       # (n,)
        if torch.any(QS <= 0):
            raise RuntimeError("Encountered non-positive QS")

        inv_QS = 1.0 / QS                # (n,)
        F = S - C * (Q.T @ inv_QS)       # (m,)

        # Convergence check
        if torch.norm(F) / torch.norm(S) < tol:
            break

        # Jacobian: J = I + c Q^T diag(1/(QS)^2) Q
        W = inv_QS ** 2                  # (n,)
        J = eye + C * (Q.T * W) @ Q      # (m, m)

        # Newton step
        delta = torch.linalg.solve(J, F)
        damping = 1.0
        S = S - damping * delta

        # Enforce positivity (numerical safety)
        S = torch.clamp(S, min=1e-12)

    # Recover omega
    omega_vals = (n * eps ** 2 / (2 * D)) * (Q @ S)
    return omega_vals


def calculate_nups(
    p : Trainer,
    state : torch.Tensor, 
    D : float, # D = calculate_max_kl_div(gamma, beta, abs_A_max, which_kl_bound)
    num_sampled_actions = 1
):
    # Constants
    eps = p.params.ROBUST_PPO_EPS
    n = p.params.NUM_FEATURES
    
    state.requires_grad_(True)
    mean, std = p.policy_model(state)
    pd = torch.distributions.Normal(mean, std)

    if num_sampled_actions == 1:
        selected_action = pd.sample()
        log_prob = pd.log_prob(selected_action).sum(-1).mean()

        if state.grad is not None:
            state.grad.zero_()
        # print(state)
        # print(log_prob)
        grads = torch.autograd.grad(log_prob, state, retain_graph = False)[0]
        u = torch.abs(grads)
        omega_vals = method1(u, n, eps, D)
        new_eps = eps / omega_vals
    elif num_sampled_actions > 1:
        # F = torch.zeros(size = (n, n), dtype = torch.float32, device = select_state.device)

        # num_sampled_actions = 10
        # for _ in range(num_sampled_actions):
        #     selected_action = pd.sample()
        #     log_prob = pd.log_prob(selected_action).sum()
        #     if select_state.grad is not None:
        #         select_state.grad.zero_()
        #     grads = torch.autograd.grad(log_prob, select_state, retain_graph = True)[0]
        #     F.add_(torch.outer(grads, grads))
        # F.divide_(num_sampled_actions)
        # # print(F)

        # print('Sample multiple actions')
        # print('original_eps=', eps)
        # print('new_eps_1=', method2(F, n, eps, D1))
        # print('new_eps_2=', method2(F, n, eps, D2))
        raise NotImplementedError
    else:
        raise NotImplementedError
    
    # print('num_sampled_actions=', num_sampled_actions)
    # print('original_eps=', eps)
    # print('new_eps=', new_eps)

    state.requires_grad_(False)

    return new_eps





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

    beta = 1000
    gamma = p.params.GAMMA
    eps = p.params.ROBUST_PPO_EPS
    print(f'beta = {beta}, gamma = {gamma}, eps = {eps}')

    use_kl_bound = 'b1'
    max_episode_length = 1024
    num_sampled_episodes = 50

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
        print(f'Attack method = {p.params.ATTACK_METHOD}| avg ep_reward of {num_sampled_episodes} episodes = {np.mean(sampled_ep_rewards)} +- {np.std(sampled_ep_rewards)}')
        print(f'avg ep_discounted_return of {num_sampled_episodes} episodes = {np.mean(sampled_ep_discounted_returns)} +- {np.std(sampled_ep_discounted_returns)}')