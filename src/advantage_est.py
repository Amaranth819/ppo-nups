import mujoco
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn


env = gym.make("Walker2d-v5")
env.reset()

# Access the underlying MuJoCo objects
uenv = env.unwrapped # unwrapped is required to access mjModel and mjData
model = uenv.model # mjModel
data = uenv.data # mjData


def snapshot_mjdata(data):
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

def restore_mjdata(model, data, snap):
    data.qpos[:] = snap["qpos"]
    data.qvel[:] = snap["qvel"]
    data.qacc[:] = snap["qacc"]
    data.act[:]  = snap["act"]
    data.ctrl[:] = snap["ctrl"]
    data.time = snap["time"]
    data.mocap_pos[:] = snap["mocap_pos"]
    data.mocap_quat[:] = snap["mocap_quat"]
    mujoco.mj_forward(model, data)

Vnet = nn.Linear(uenv.observation_space.shape[0], 1)

iterations = 5
num_sampled_actions = 64
num_elite_actions = 10
max_episode_length = 1024

action_space_size = uenv.action_space.shape[0]
action_low, action_high = uenv.action_space.low, uenv.action_space.high

gamma = 0.99

abs_A_max = -1e5
best_action = None

obs, _ = uenv.reset()
model = uenv.model
data = uenv.data

for t in range(max_episode_length):
    snap = snapshot_mjdata(data)

    # Cross-Entropy Method
    mu = np.zeros(shape = (1, action_space_size))
    sigma = np.ones(shape = (1, action_space_size)) * 0.5

    for iter in range(iterations):
        sampled_actions = np.random.randn(num_sampled_actions, action_space_size) * sigma + mu
        sampled_actions = np.clip(sampled_actions, action_low, action_high)
        
        collected_advs = []
        for action in sampled_actions:
            next_obs, rewards, terminated, truncated, info = uenv.step(action)

            obs_ts = torch.from_numpy(obs).float()
            next_obs_ts = torch.from_numpy(next_obs).float()

            adv = rewards + gamma * Vnet(next_obs_ts).item() - Vnet(obs_ts).item()
            collected_advs.append(adv)
            
            restore_mjdata(model, data, snap)

            # # Verify the restored state
            # obs_restored = uenv._get_obs()
            # print(np.abs(obs_restored - obs))

        abs_advs = np.abs(np.array(collected_advs))

        # Update abs_A_max
        idx = np.argmax(abs_advs)
        if abs_advs[idx] > abs_A_max:
            abs_A_max = abs_advs[idx]
            best_action = sampled_actions[idx]

        # Select elite actions and update Gaussian
        elite_indices = abs_advs.argsort()[-num_elite_actions:]
        elite_actions = sampled_actions[elite_indices]

        mu = np.mean(elite_actions, axis = 0)
        sigma = np.std(elite_actions, axis = 0) + 1e-6

    # Take a step with the pretrained policy
    obs, rewards, terminated, truncated, info = uenv.step(action)
    if terminated or truncated:
        break

print(f'|A|_max = {abs_A_max}')