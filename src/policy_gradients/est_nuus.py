'''
    Functions for calculating the Non-Uniform Uncertainty Set (NUUS) 
'''

import torch
import numpy as np
import cvxpy as cp
import mujoco 
import gymnasium as gym


'''
    Method 1: quick approximation with 1 sampled action and log probability gradient.
'''
def quick_approx_omegas(
    logprob_grad : torch.Tensor, # \nabla_s \log \pi(a|s)
    eps : float, # epsilon of L-inf norm uncertainty set 
    max_D_KL : float, # maximum KL divergence
):
    n = logprob_grad.size(-1)
    omegas = n * eps / np.sqrt(2 * max_D_KL) * torch.abs(logprob_grad)
    return omegas


'''
    Method 2: Formulate the problem as a geometric programming problem and solve it using cvxpy.
'''
class NUUS_GPSolverNumpy(object):
    def __init__(self, n, eps):
        self.eps = eps

        # Set up the problem, and we should guarantee that the problem is DGP. Here y = 1/omega !
        self.y = cp.Variable(n, pos = True)
        self.F = cp.Parameter(shape = (n, n), pos = True)
        self.F.value = np.eye(n) + 1e-4
        self.D = cp.Parameter(pos = True)
        obj = cp.prod(self.y ** (-1 / n))
        obj = cp.Minimize(obj)
        constraint = eps**2 / (2 * self.D) * cp.sum(cp.multiply(self.y, self.F @ self.y)) <= 1
        self.prob = cp.Problem(obj, [constraint])

        assert self.prob.is_dgp()


    def solve(self, F_val : np.ndarray, D_val : float):
        self.F.value = F_val
        self.D.value = D_val
        self.prob.solve(gp = True, solver = cp.SCS, eps_abs = 1e-5, eps_rel = 1e-5)
        y_vals = self.y.value
        omegas = 1 / y_vals
        return omegas


# Estimate the state Fisher Information Matrix F(s,\pi) = \mathbb{E}_{a\sim \pi}[ \nabla_s\log\pi(a|s) \nabla^T_s\log\pi(a|s)]
def calculate_state_FIM(
    state : torch.Tensor, # [1, n]
    action_dist : torch.distributions.Normal,
    num_sampled_actions : int = 1    
):
    # Current numpy solver does not support batch operation. The torch solver does support but is slow.
    # assert len(state.size()) == 2 and state.size(0) == 1
    state.requires_grad_(True)

    FIM = torch.zeros(shape = (state.size(-1), state.size(-1)), device = state.device)
    for a_idx in range(num_sampled_actions):
        selected_action = action_dist.sample()
        log_prob = action_dist.log_prob(selected_action).sum(-1).mean()

        if state.grad is not None:
            state.grad.zero_()
        grads = torch.autograd.grad(log_prob, state, retain_graph = False if a_idx == num_sampled_actions - 1 else True)[0].detach().squeeze(0)
        FIM += torch.outer(grads, grads)
    
    FIM /= num_sampled_actions

    state.requires_grad_(False)

    return FIM.detach().cpu().numpy()
    



def calculate_nuus(
    policy_model : torch.nn.Module,
    state : torch.Tensor, 
    max_D_KL : float, # maximum KL divergence
    eps : float,
    num_sampled_actions : int = 1,
    solver : str = 'approx',
    nuus_solver : NUUS_GPSolverNumpy = None,
):
    # Current numpy solver does not support batch operation. The torch solver does support but is slow.
    assert len(state.size()) == 2 and state.size(0) == 1
    
    state.requires_grad_(True)

    mean, std = policy_model(state)
    pd = torch.distributions.Normal(mean, std)

    if solver == 'approx':
        assert num_sampled_actions == 1
        selected_action = pd.sample()
        log_prob = pd.log_prob(selected_action).sum(-1).mean()

        if state.grad is not None:
            state.grad.zero_()
        grads = torch.autograd.grad(log_prob, state, retain_graph = False)[0]
        omega_vals = quick_approx_omegas(grads, eps, max_D_KL)

    elif solver == 'cvxpy':
        FIM = calculate_state_FIM(state, pd, num_sampled_actions)
        FIM = torch.abs(FIM)
        assert nuus_solver is not None
        omega_vals = nuus_solver.solve(FIM, max_D_KL)
        omega_vals = torch.from_numpy(omega_vals).to(state.device).float().unsqueeze(0)
    
    else:
        raise NotImplementedError(solver)

    state.requires_grad_(False)
    new_eps = eps / omega_vals

    return new_eps







'''
    Cross-Entropy Method (CEM) for estimating \max_a A^\pi(s,a).
    Sample a list of actions, do 1-step simulation, then recover the state.
'''
def snapshot_mjdata(
    env : gym.Env # Environment created by gym.make()
):
    data = env.get_wrapper_attr('data')
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
    env : gym.Env, # Environment created by gym.make()
    snap # the stored mjdata created by snapshot_mjdata()
):
    data = env.get_wrapper_attr('data')
    data.qpos[:] = snap["qpos"]
    data.qvel[:] = snap["qvel"]
    data.qacc[:] = snap["qacc"]
    data.act[:]  = snap["act"]
    data.ctrl[:] = snap["ctrl"]
    data.time = snap["time"]
    data.mocap_pos[:] = snap["mocap_pos"]
    data.mocap_quat[:] = snap["mocap_quat"]

    model = env.get_wrapper_attr('model')
    mujoco.mj_forward(model, data)



'''
    Estimate the maximum absolute value of A^\pi(s,a) at state s, using CEM 
'''
def estimate_maxa_absA_with_CEM(
    env : gym.Env, # Environment created by gym.make()
    val_model : torch.nn.Module,
    state : torch.Tensor,
    gamma : float,
    num_iterations : int,
    num_sampled_actions : int,
    num_elite_actions : int,
    cpu : bool = False,
):
    assert len(state.size()) == 2 and state.size(0) == 1

    snap = snapshot_mjdata(env)
    device = 'cpu' if cpu else 'cuda:0'
    action_space_size = env.action_space.shape[0]
    action_low, action_high = env.action_space.low, env.action_space.high

    with torch.no_grad():
        V_s = val_model(state).flatten().item()

    # Initialize Gaussian
    mu = np.zeros(shape = (1, action_space_size))
    sigma = np.ones(shape = (1, action_space_size)) * 1

    abs_A_max = -1e5
    collected_advs = []

    for _ in range(num_iterations):
        sampled_actions = np.random.randn(num_sampled_actions, action_space_size) * sigma + mu
        sampled_actions = np.clip(sampled_actions, action_low, action_high)

        s_next_lists = []
        reward_lists = []

        for action in sampled_actions:
            next_obs, rew, _, _, info = env.step(action)
            s_next_lists.append(next_obs)
            reward_lists.append(rew)
            restore_mjdata(env, snap)

        s_next_ts = torch.from_numpy(np.stack(s_next_lists)).float().to(device)
        with torch.no_grad():
            V_s_next = val_model(s_next_ts).squeeze(-1).cpu().numpy()
        curr_iter_advs = np.array(reward_lists) + gamma * V_s_next - V_s
        collected_advs.append(curr_iter_advs)

        # Update abs_A_max
        abs_advs = np.abs(curr_iter_advs)
        abs_A_max = max(abs_A_max, np.max(abs_advs))

        # Select elite actions and update Gaussian
        elite_indices = abs_advs.argsort()[-num_elite_actions:]
        elite_actions = sampled_actions[elite_indices]
        mu = np.mean(elite_actions, axis = 0, keepdims = True)
        sigma = np.std(elite_actions, axis = 0, keepdims = True) + 1e-6

    restore_mjdata(env, snap)

    collected_advs = np.concatenate(collected_advs).flatten()

    return abs_A_max, collected_advs




'''
    Upper bounds for KL divergence
'''
def D_KL_ub_expression_1(gamma, beta, maxa_absA):
    return 0.5 * ( (1-gamma) * beta / maxa_absA )**2



def calculate_max_D_KL(ub_type = 1, **args):
    if ub_type == 1:
        return D_KL_ub_expression_1(**args)
    else:
        raise NotImplementedError