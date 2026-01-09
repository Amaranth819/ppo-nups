# If I switch the order of the observation space, then will the sorted order of eigenvalues change?

from policy_gradients.agent import Trainer
import git
import pickle
import random
import numpy as np
import os
import argparse
import traceback
from policy_gradients import models
import sys
import json
import torch
from cox.store import Store, schema_from_dict


# Trainer with customized state space
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, '../'))

import torch.optim as optim
from src.policy_gradients.torch_utils import *
from src.policy_gradients.models import *
from src.policy_gradients.custom_env import Env
from src.policy_gradients.steps import value_step, step_with_mode, pack_history
from src.policy_gradients.logging import *
from PIL import Image


class CustomStateSpaceEnv(Env):
    def __init__(self, game, norm_states, norm_rewards, params, add_t_with_horizon=None, clip_obs=None, clip_rew=None, show_env=False, save_frames=False, save_frames_path=""):
        super().__init__(game, norm_states, norm_rewards, params, add_t_with_horizon, clip_obs, clip_rew, show_env, save_frames, save_frames_path)


    def _change_state_func(self, s):
        # return s
        s = np.concatenate([s[..., 1:], s[..., :1]], axis = -1)
        return s
    

    def reset(self):
        start_state = self.env.reset()[0]
        self.total_true_reward = 0.0
        self.counter = 0.0
        self.episode_counter += 1
        if self.save_frames:
            os.makedirs(os.path.join(self.save_frames_path, f"{self.episode_counter:03d}"), exist_ok=True)
            self.frame_counter = 0
        self.state_filter.reset()
        self.reward_filter.reset()
        
        start_state = self._change_state_func(start_state)

        return self.state_filter(start_state, reset=True)
    

    def step(self, action):
        # state, reward, is_done, info = self.env.step(action)
        state, reward, terminated, truncated, info = self.env.step(action)
        is_done = np.logical_or(terminated, truncated)
        # if self.show_env:
        #     self.env.render()
        # Frameskip (every 6 frames, will be rendered at 25 fps)
        if self.save_frames and int(self.counter) % 6 == 0:
            image = self.env.render(mode='rgb_array')
            path = os.path.join(self.save_frames_path, f"{self.episode_counter:03d}", f"{self.frame_counter+1:04d}.bmp")
            image = Image.fromarray(image)
            image.save(path)
            self.frame_counter += 1

        state = self._change_state_func(state)

        state = self.state_filter(state)
        self.total_true_reward += reward
        self.counter += 1
        _reward = self.reward_filter(reward)
        if is_done:
            info['done'] = (self.counter, self.total_true_reward)
        return state, _reward, is_done, info
    




class TrainerWithCustomStateSpaceEnv(Trainer):
    def __init__(self, policy_net_class, value_net_class, params, store, advanced_logging=True, log_every=5):
        '''
        Initializes a new Trainer class.
        Inputs;
        - policy, the class of policy network to use (inheriting from nn.Module)
        - val, the class of value network to use (inheriting from nn.Module)
        - step, a reference to a function to use for the policy step (see steps.py)
        - params, an dictionary with all of the required hyperparameters
        '''
        # Parameter Loading
        self.params = Parameters(params)

        # Whether or not the value network uses the current timestep
        time_in_state = self.VALUE_CALC == "time"

        # Whether to use GPU (as opposed to CPU)
        if not self.CPU:
            torch.set_default_tensor_type("torch.cuda.FloatTensor")

        # Environment Loading
        def env_constructor():
            # Whether or not we should add the time to the state
            horizon_to_feed = self.T if time_in_state else None
            return CustomStateSpaceEnv(self.GAME, 
                norm_states=self.NORM_STATES,
                norm_rewards=self.NORM_REWARDS,
                params=self.params,
                add_t_with_horizon=horizon_to_feed,
                clip_obs=self.CLIP_OBSERVATIONS,
                clip_rew=self.CLIP_REWARDS,
                show_env=self.SHOW_ENV,
                save_frames=self.SAVE_FRAMES,
                save_frames_path=self.SAVE_FRAMES_PATH
            )

        self.envs = [env_constructor() for _ in range(self.NUM_ACTORS)]
        self.params.AGENT_TYPE = "discrete" if self.envs[0].is_discrete else "continuous"
        self.params.NUM_ACTIONS = self.envs[0].num_actions
        self.params.NUM_FEATURES = self.envs[0].num_features
        self.policy_step = step_with_mode(self.MODE, adversary=False)
        self.adversary_policy_step = step_with_mode(self.MODE, adversary=True)
        self.params.MAX_KL_INCREMENT = (self.params.MAX_KL_FINAL - self.params.MAX_KL) / self.params.TRAIN_STEPS
        self.advanced_logging = advanced_logging
        self.n_steps = 0
        self.log_every = log_every
        self.policy_net_class = policy_net_class

        # Instantiation
        self.policy_model = policy_net_class(self.NUM_FEATURES, self.NUM_ACTIONS,
                                             self.INITIALIZATION,
                                             time_in_state=time_in_state,
                                             activation=self.policy_activation)

        # Instantiate convex relaxation model when mode is 'robust_ppo'
        if self.MODE == 'robust_ppo' or self.MODE == 'adv_sa_ppo':
            self.create_relaxed_model(time_in_state)

        # Minimax training
        if self.MODE == 'adv_ppo' or self.MODE == 'adv_trpo' or self.MODE == 'adv_sa_ppo':
            # Copy parameters if they are set to "same".
            if self.params.ADV_PPO_LR_ADAM == "same":
                self.params.ADV_PPO_LR_ADAM = self.params.PPO_LR_ADAM
            if self.params.ADV_VAL_LR == "same":
                self.params.ADV_VAL_LR = self.params.VAL_LR
            if self.params.ADV_CLIP_EPS == "same":
                self.params.ADV_CLIP_EPS = self.params.CLIP_EPS
            if self.params.ADV_EPS == "same":
                self.params.ADV_EPS = self.params.ROBUST_PPO_EPS
            if self.params.ADV_ENTROPY_COEFF == "same":
                self.params.ADV_ENTROPY_COEFF = self.params.ENTROPY_COEFF
            # The adversary policy has features as input, features as output.
            self.adversary_policy_model = policy_net_class(self.NUM_FEATURES, self.NUM_FEATURES,
                                                 self.INITIALIZATION,
                                                 time_in_state=time_in_state,
                                                 activation=self.policy_activation)
            # Optimizer for adversary
            self.params.ADV_POLICY_ADAM = optim.Adam(self.adversary_policy_model.parameters(), lr=self.ADV_PPO_LR_ADAM, eps=1e-5)

            # Adversary value function.
            self.adversary_val_model = value_net_class(self.NUM_FEATURES, self.INITIALIZATION)
            self.adversary_val_opt = optim.Adam(self.adversary_val_model.parameters(), lr=self.ADV_VAL_LR, eps=1e-5)
            assert self.adversary_policy_model.discrete == (self.AGENT_TYPE == "discrete")

            # Learning rate annealling for adversary.
            if self.ANNEAL_LR:
                adv_lam = lambda f: 1-f/self.TRAIN_STEPS
                adv_ps = optim.lr_scheduler.LambdaLR(self.ADV_POLICY_ADAM, 
                                                        lr_lambda=adv_lam)
                adv_vs = optim.lr_scheduler.LambdaLR(self.adversary_val_opt, lr_lambda=adv_lam)
                self.params.ADV_POLICY_SCHEDULER = adv_ps
                self.params.ADV_VALUE_SCHEDULER = adv_vs

        opts_ok = (self.PPO_LR == -1 or self.PPO_LR_ADAM == -1)
        assert opts_ok, "One of ppo_lr and ppo_lr_adam must be -1 (off)."
        # Whether we should use Adam or simple GD to optimize the policy parameters
        if self.PPO_LR_ADAM != -1:
            kwargs = {
                'lr':self.PPO_LR_ADAM,
            }

            if self.params.ADAM_EPS > 0:
                kwargs['eps'] = self.ADAM_EPS

            self.params.POLICY_ADAM = optim.Adam(self.policy_model.parameters(),
                                                 **kwargs)
        else:
            self.params.POLICY_ADAM = optim.SGD(self.policy_model.parameters(), lr=self.PPO_LR)

        # If using a time dependent value function, add one extra feature
        # for the time ratio t/T
        if time_in_state:
            self.params.NUM_FEATURES = self.NUM_FEATURES + 1

        # Value function optimization
        self.val_model = value_net_class(self.NUM_FEATURES, self.INITIALIZATION)
        self.val_opt = optim.Adam(self.val_model.parameters(), lr=self.VAL_LR, eps=1e-5) 
        assert self.policy_model.discrete == (self.AGENT_TYPE == "discrete")

        # Learning rate annealing
        # From OpenAI hyperparametrs:
        # Set adam learning rate to 3e-4 * alpha, where alpha decays from 1 to 0 over training
        if self.ANNEAL_LR:
            lam = lambda f: 1-f/self.TRAIN_STEPS
            ps = optim.lr_scheduler.LambdaLR(self.POLICY_ADAM, 
                                                    lr_lambda=lam)
            vs = optim.lr_scheduler.LambdaLR(self.val_opt, lr_lambda=lam)
            self.params.POLICY_SCHEDULER = ps
            self.params.VALUE_SCHEDULER = vs

        if store is not None:
            self.setup_stores(store)
        else:
            print("Not saving results to cox store.")


    @staticmethod
    def agent_from_params(params, store=None):
        '''
        Construct a trainer object given a dictionary of hyperparameters.
        Trainer is in charge of sampling trajectories, updating policy network,
        updating value network, and logging.
        Inputs:
        - params, dictionary of required hyperparameters
        - store, a cox.Store object if logging is enabled
        Outputs:
        - A Trainer object for training a PPO/TRPO agent
        '''
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
        p = TrainerWithCustomStateSpaceEnv(agent_policy, agent_value, params, store, log_every=log_every,
                    advanced_logging=advanced_logging)

        return p






# Tee object allows for logging to both stdout and to file
class Tee(object):
    def __init__(self, file_path, stream_type, mode='a'):
        assert stream_type in ['stdout', 'stderr']

        self.file = open(file_path, mode)
        self.stream_type = stream_type
        self.errors = 'chill'

        if stream_type == 'stdout':
            self.stream = sys.stdout
            sys.stdout = self
        else:
            self.stream = sys.stderr
            sys.stderr = self

    def write(self, data):
        self.file.write(data)
        self.stream.write(data)

    def flush(self):
        self.file.flush()
        self.stream.flush()

def main(params):
    for k, v in zip(params.keys(), params.values()):
        assert v is not None, f"Value for {k} is None"

    # #
    # Setup logging
    # #
    metadata_schema = schema_from_dict(params)
    base_directory = params['out_dir']

    from datetime import datetime
    now = datetime.now()
    formatted_string = now.strftime("%Y-%m-%d_%H-%M-%S")

    exp_id = 'change_statespace_' + formatted_string
    store = Store(base_directory, exp_id = exp_id)

    # redirect stderr, stdout to file
    """
    def make_err_redirector(stream_name):
        tee = Tee(os.path.join(store.path, stream_name + '.txt'), stream_name)
        return tee

    stderr_tee = make_err_redirector('stderr')
    stdout_tee = make_err_redirector('stdout')
    """

    # Store the experiment path and the git commit for this experiment
    metadata_schema.update({
        'store_path': str,
        # 'git_commit': str
    })

    # repo = git.Repo(path=os.path.dirname(os.path.realpath(__file__)),
    #                 search_parent_directories=True)

    metadata_table = store.add_table('metadata', metadata_schema)
    metadata_table.update_row(params)
    metadata_table.update_row({
        'store_path': store.path,
        # 'git_commit': repo.head.object.hexsha
    })
    metadata_table.flush_row()

    # Extra items in table when minimax training is enabled.
    if params['mode'] == "adv_ppo" or params['mode'] == 'adv_trpo' or params['mode'] == 'adv_sa_ppo':
        adversary_table_dict = {
            'adversary_policy_model': store.PYTORCH_STATE,
            'adversary_policy_opt': store.PYTORCH_STATE,
            'adversary_val_model': store.PYTORCH_STATE,
            'adversary_val_opt': store.PYTORCH_STATE,
        }
    else:
        adversary_table_dict = {}

    # Table for checkpointing models and envs
    if params['save_iters'] > 0:
        checkpoint_dict = {
            'val_model': store.PYTORCH_STATE,
            'policy_model': store.PYTORCH_STATE,
            'envs': store.PICKLE,
            'policy_opt': store.PYTORCH_STATE,
            'val_opt': store.PYTORCH_STATE,
            'iteration': int,
            '5_rewards': float,
        }
        checkpoint_dict.update(adversary_table_dict)
        store.add_table('checkpoints', checkpoint_dict)

    # The trainer object is in charge of sampling trajectories and
    # taking PPO/TRPO optimization steps

    p = TrainerWithCustomStateSpaceEnv.agent_from_params(params, store=store)
    # p = Trainer.agent_from_params(params, store=store)
    if params['initial_std'] != 1.0:
        p.policy_model.log_stdev.data[:] = np.log(params['initial_std'])
    if 'load_model' in params and params['load_model']:
        print('Loading pretrained model', params['load_model'])
        pretrained_model = torch.load(params['load_model'])
        if 'policy_model' in pretrained_model:
            p.policy_model.load_state_dict(pretrained_model['policy_model'])
        if params['deterministic']:
            print('Policy runs in deterministic mode. Ignoring Gaussian noise.')
            p.policy_model.log_stdev.data[:] = -100
        else:
            print('Policy runs in non deterministic mode with Gaussian noise.')
        if 'val_model' in pretrained_model:
            p.val_model.load_state_dict(pretrained_model['val_model'])
        if 'policy_opt' in pretrained_model:
            p.POLICY_ADAM.load_state_dict(pretrained_model['policy_opt'])
        if 'val_opt' in pretrained_model:
            p.val_opt.load_state_dict(pretrained_model['val_opt'])
        # Load adversary models.
        if 'no_load_adv_policy' in params and params['no_load_adv_policy']:
            print('Skipping loading adversary models.')
        else:
            if 'adversary_policy_model' in pretrained_model and hasattr(p, 'adversary_policy_model'):
                p.adversary_policy_model.load_state_dict(pretrained_model['adversary_policy_model'])
            if 'adversary_val_model' in pretrained_model and hasattr(p, 'adversary_val_model'):
                p.adversary_val_model.load_state_dict(pretrained_model['adversary_val_model'])
            if 'adversary_policy_opt' in pretrained_model and hasattr(p, 'adversary_policy_opt'):
                p.adversary_policy_opt.load_state_dict(pretrained_model['adversary_policy_opt'])
            if 'adversary_val_opt' in pretrained_model and hasattr(p, 'adversary_val_opt'):
                p.adversary_val_opt.load_state_dict(pretrained_model['adversary_val_opt'])
        # Load optimizer states.
        # p.POLICY_ADAM.load_state_dict(pretrained_models['policy_opt'])
        # p.val_opt.load_state_dict(pretrained_models['val_opt'])
        # Restore environment parameters, like mean and std.
        if 'envs' in pretrained_model:
            p.envs = pretrained_model['envs']
        for e in p.envs:
            e.setup_visualization(params['show_env'], params['save_frames'], params['save_frames_path'])
    rewards = []

    # # Test 
    # p.envs[0].reset()
    # for _ in range(15):
    #     ns, _, _, _ = p.envs[0].step(p.envs[0].env.action_space.sample())
    #     print(ns)
    # return

    # Table for final results
    final_dict = {
        'iteration': int,
        '5_rewards': float,
        'terminated_early': bool,
        'val_model': store.PYTORCH_STATE,
        'policy_model': store.PYTORCH_STATE,
        'envs': store.PICKLE,
        'policy_opt': store.PYTORCH_STATE,
        'val_opt': store.PYTORCH_STATE,
    }
    final_dict.update(adversary_table_dict)
    final_table = store.add_table('final_results', final_dict)

    def add_adversary_to_table(p, table_dict):
        if params['mode'] == "adv_ppo" or params['mode'] == 'adv_trpo' or params['mode'] == 'adv_sa_ppo':
            table_dict["adversary_policy_model"] = p.adversary_policy_model.state_dict()
            table_dict["adversary_policy_opt"] = p.ADV_POLICY_ADAM.state_dict()
            table_dict["adversary_val_model"] = p.adversary_val_model.state_dict()
            table_dict["adversary_val_opt"] = p.adversary_val_opt.state_dict()
        return table_dict

    def finalize_table(iteration, terminated_early, rewards):
        final_5_rewards = np.array(rewards)[-5:].mean()
        final_dict = {
            'iteration': iteration,
            '5_rewards': final_5_rewards,
            'terminated_early': terminated_early,
            'val_model': p.val_model.state_dict(),
            'policy_model': p.policy_model.state_dict(),
            'policy_opt': p.POLICY_ADAM.state_dict(),
            'val_opt': p.val_opt.state_dict(),
            'envs': p.envs
        }
        final_dict = add_adversary_to_table(p, final_dict)
        final_table.append_row(final_dict)

    ret = 0
    # Try-except so that we save if the user interrupts the process
    try:
        for i in range(params['train_steps']):
            print('Step %d' % (i,))
            if params['save_iters'] > 0 and i % params['save_iters'] == 0 and i != 0:
                final_5_rewards = np.array(rewards)[-5:].mean()
                print(f'Saving checkpoints to {store.path} with reward {final_5_rewards:.5g}')
                checkpoint_dict = {
                    'iteration': i,
                    'val_model': p.val_model.state_dict(),
                    'policy_model': p.policy_model.state_dict(),
                    'policy_opt': p.POLICY_ADAM.state_dict(),
                    'val_opt': p.val_opt.state_dict(),
                    'envs': p.envs,
                    '5_rewards': final_5_rewards,
                }
                checkpoint_dict = add_adversary_to_table(p, checkpoint_dict)
                store['checkpoints'].append_row(checkpoint_dict)

            mean_reward = p.train_step()
            rewards.append(mean_reward)

            # For debugging and tuning, we can break in the middle.
            if i == params['force_stop_step']:
                print('Terminating early because --force-stop-step is set.')
                raise KeyboardInterrupt

        finalize_table(i, False, rewards)
    except KeyboardInterrupt:
        finalize_table(i, True, rewards)
        ret = 1
    except:
        print("An error occurred during training:")
        traceback.print_exc()
        # Other errors, make sure to finalize the cox store before exiting.
        finalize_table(i, True, rewards)
        ret = -1
    print(f'Models saved to {store.path}')
    store.close()
    return ret

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def add_common_parser_opts(parser):
    # Basic setup
    parser.add_argument('--game', type=str, help='gym game')
    parser.add_argument('--mode', type=str, choices=['ppo', 'trpo', 'robust_ppo', 'adv_ppo', 'adv_trpo', 'adv_sa_ppo'],
                        help='pg alg')
    parser.add_argument('--out-dir', type=str,
                        help='out dir for store + logging')
    parser.add_argument('--advanced-logging', type=str2bool, const=True, nargs='?')
    parser.add_argument('--kl-approximation-iters', type=int,
                        help='how often to do kl approx exps')
    parser.add_argument('--log-every', type=int)
    parser.add_argument('--policy-net-type', type=str,
                        choices=models.POLICY_NETS.keys())
    parser.add_argument('--value-net-type', type=str,
                        choices=models.VALUE_NETS.keys())
    parser.add_argument('--train-steps', type=int,
                        help='num agent training steps')
    parser.add_argument('--cpu', type=str2bool, const=True, nargs='?')

    # Which value loss to use
    parser.add_argument('--value-calc', type=str,
                        help='which value calculation to use')
    parser.add_argument('--initialization', type=str)

    # General Policy Gradient parameters
    parser.add_argument('--num-actors', type=int, help='num actors (serial)',
                        choices=[1])
    parser.add_argument('--t', type=int,
                        help='num timesteps to run each actor for')
    parser.add_argument('--gamma', type=float, help='discount on reward')
    parser.add_argument('--lambda', type=float, help='GAE hyperparameter')
    parser.add_argument('--val-lr', type=float, help='value fn learning rate')
    parser.add_argument('--val-epochs', type=int, help='value fn epochs')
    parser.add_argument('--initial-std', type=float, help='initial value of std for Gaussian policy. Default is 1.')

    # PPO parameters
    parser.add_argument('--adam-eps', type=float, choices=[0, 1e-5], help='adam eps parameter')

    parser.add_argument('--num-minibatches', type=int,
                        help='num minibatches in ppo per epoch')
    parser.add_argument('--ppo-epochs', type=int)
    parser.add_argument('--ppo-lr', type=float,
                        help='if nonzero, use gradient descent w this lr')
    parser.add_argument('--ppo-lr-adam', type=float,
                        help='if nonzero, use adam with this lr')
    parser.add_argument('--anneal-lr', type=str2bool,
                        help='if we should anneal lr linearly from start to finish')
    parser.add_argument('--clip-eps', type=float, help='ppo clipping')
    parser.add_argument('--clip-val-eps', type=float, help='ppo clipping value')
    parser.add_argument('--entropy-coeff', type=float,
                        help='entropy weight hyperparam')
    parser.add_argument('--value-clipping', type=str2bool,
                        help='should clip values (w/ ppo eps)')
    parser.add_argument('--value-multiplier', type=float,
                        help='coeff for value loss in combined step ppo loss')
    parser.add_argument('--share-weights', type=str2bool,
                        help='share weights in valnet and polnet')
    parser.add_argument('--clip-grad-norm', type=float,
                        help='gradient norm clipping (-1 for no clipping)')
    parser.add_argument('--policy-activation', type=str,
                        help='activation function for countinous policy network')
    
    # TRPO parameters
    parser.add_argument('--max-kl', type=float, help='trpo max kl hparam')
    parser.add_argument('--max-kl-final', type=float, help='trpo max kl final')
    parser.add_argument('--fisher-frac-samples', type=float,
                        help='frac samples to use in fisher vp estimate')
    parser.add_argument('--cg-steps', type=int,
                        help='num cg steps in fisher vp estimate')
    parser.add_argument('--damping', type=float, help='damping to use in cg')
    parser.add_argument('--max-backtrack', type=int, help='max bt steps in fvp')
    parser.add_argument('--trpo-kl-reduce-func', type=str, help='reduce function for KL divergence used in line search. mean or max.')

    # Robust PPO parameters.
    parser.add_argument('--robust-ppo-eps', type=float, help='max eps for robust PPO training')
    parser.add_argument('--robust-ppo-method', type=str, choices=['convex-relax', 'sgld', 'pgd'], help='robustness regularization methods')
    parser.add_argument('--robust-ppo-pgd-steps', type=int, help='number of PGD optimization steps')
    parser.add_argument('--robust-ppo-detach-stdev', type=str2bool, help='detach gradient of standard deviation term')
    parser.add_argument('--robust-ppo-reg', type=float, help='robust PPO regularization')
    parser.add_argument('--robust-ppo-eps-scheduler-opts', type=str, help='options for epsilon scheduler for robust PPO training')
    parser.add_argument('--robust-ppo-beta', type=float, help='max beta (IBP mixing factor) for robust PPO training')
    parser.add_argument('--robust-ppo-beta-scheduler-opts', type=str, help='options for beta scheduler for robust PPO training')

    # Adversarial PPO parameters.
    parser.add_argument('--adv-ppo-lr-adam', type=float,
                        help='if nonzero, use adam for adversary policy with this lr')
    parser.add_argument('--adv-entropy-coeff', type=float,
                        help='entropy weight hyperparam for adversary policy')
    parser.add_argument('--adv-eps', type=float, help='adversary perturbation eps')
    parser.add_argument('--adv-clip-eps', type=float, help='ppo clipping for adversary policy')
    parser.add_argument('--adv-val-lr', type=float, help='value fn learning rate for adversary policy')
    parser.add_argument('--adv-policy-steps', type=float, help='number of policy steps before adversary steps')
    parser.add_argument('--adv-adversary-steps', type=float, help='number of adversary steps before adversary steps')
    parser.add_argument('--adv-adversary-ratio', type=float, help='percentage of frames to attack for the adversary')

    # Adversarial attack parameters.
    parser.add_argument('--attack-method', type=str, choices=["none", "critic", "random", "action", "sarsa", "sarsa+action", "advpolicy", "action+imit"], help='adversarial attack methods.')
    parser.add_argument('--attack-ratio', type=float, help='attack only a ratio of steps.')
    parser.add_argument('--attack-steps', type=int, help='number of PGD optimization steps.')
    parser.add_argument('--attack-eps', type=str, help='epsilon for attack. If set to "same", we will use value of robust-ppo-eps.')
    parser.add_argument('--attack-step-eps', type=str, help='step size for each iteration. If set to "auto", we will use attack-eps / attack-steps')
    parser.add_argument('--attack-sarsa-network', type=str, help='sarsa network to load for attack.')
    parser.add_argument('--attack-sarsa-action-ratio', type=float, help='When set to non-zero, enable sarsa-action attack.')
    parser.add_argument('--attack-advpolicy-network', type=str, help='adversarial policy network to load for attack.')
    parser.add_argument('--collect-perturbed-states', type=str2bool, help='collect perturbed states during training')

    # Normalization parameters
    parser.add_argument('--norm-rewards', type=str, help='type of rewards normalization', 
                        choices=['rewards', 'returns', 'none'])
    parser.add_argument('--norm-states', type=str2bool, help='should norm states')
    parser.add_argument('--clip-rewards', type=float, help='clip rews eps')
    parser.add_argument('--clip-observations', type=float, help='clips obs eps')

    # Sequence training parameters
    parser.add_argument('--history-length', type=int, help='length of history to use for LSTM. If <= 1, we do not use LSTM.')
    parser.add_argument('--use-lstm-val', type=str2bool, help='use a lstm for value function')

    # Saving
    parser.add_argument('--save-iters', type=int, help='how often to save model (0 = no saving)')
    parser.add_argument('--force-stop-step', type=int, help='forcibly terminate after a given number of steps. Useful for debugging and tuning.')

    # Visualization
    parser.add_argument('--show-env', type=str2bool, help='Show environment visualization')
    parser.add_argument('--save-frames', type=str2bool, help='Save environment frames')
    parser.add_argument('--save-frames-path', type=str, help='Path to save environment frames')

    # For grid searches only
    # parser.add_argument('--cox-experiment-path', type=str, default='')
    return parser


def override_json_params(params, json_params, excluding_params):
    # Override the JSON config with the argparse config
    missing_keys = []
    for key in json_params:
        if key not in params:
            missing_keys.append(key)
    assert not missing_keys, "Following keys not in args: " + str(missing_keys)

    missing_keys = []
    for key in params:
        if key not in json_params and key not in excluding_params:
            missing_keys.append(key)
    assert not missing_keys, "Following keys not in JSON: " + str(missing_keys)

    json_params.update({k: params[k] for k in params if params[k] is not None})
    return json_params


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate experiments to be run.')
    parser.add_argument('--config-path', type=str, required=True,
                        help='json for this config')
    parser.add_argument('--out-dir-prefix', type=str, default="", required=False,
                        help='prefix for output log path')
    parser.add_argument('--load-model', type=str, default=None, required=False, help='load pretrained model and optimizer states before training')
    parser.add_argument('--no-load-adv-policy', action='store_true', required=False, help='Do not load adversary policy and value network from pretrained model.')
    parser.add_argument('--adv-policy-only', action='store_true', required=False, help='Run adversary only, by setting main agent learning rate to 0')
    parser.add_argument('--deterministic', action='store_true', help='disable Gaussian noise in action for --adv-policy-only mode')
    parser.add_argument('--seed', type=int, help='random seed', default=-1)
    parser = add_common_parser_opts(parser)
    
    args = parser.parse_args()

    params = vars(args)
    seed = params['seed']
    json_params = json.load(open(args.config_path))

    extra_params = ['config_path', 'out_dir_prefix', 'load_model', 'no_load_adv_policy', 'adv_policy_only', 'deterministic', 'seed']
    params = override_json_params(params, json_params, extra_params)

    if params['adv_policy_only']:
        if params['adv_ppo_lr_adam'] == 'same':
            params['adv_ppo_lr_adam'] = params['ppo_lr_adam']
            print(f"automatically setting adv_ppo_lr_adam to {params['adv_ppo_lr_adam']}")
        print('disabling policy training (train adversary only)')
        params['ppo_lr_adam'] = 0.0 * params['ppo_lr_adam']
    else:
        # deterministic mode only valid when --adv-policy-only is set
        assert not params['deterministic']

    if seed != -1:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        random.seed(seed)
        np.random.seed(seed)
    torch.set_printoptions(threshold=5000, linewidth=120)

    # Append a prefix for output path.
    if args.out_dir_prefix:
        params['out_dir'] = os.path.join(args.out_dir_prefix, params['out_dir'])
        print(f"setting output dir to {params['out_dir']}")
    main(params)