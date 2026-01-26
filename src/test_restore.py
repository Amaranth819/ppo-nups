import numpy as np
import gymnasium as gym
from policy_gradients.custom_env import snapshot_mjdata, restore_mjdata


def main():
    env = gym.make('Hopper-v5').unwrapped
    obs = env.reset()[0]
    snap = snapshot_mjdata(env)

    actions = [env.action_space.sample() for _ in range(5)]

    old_obs = []
    for a in actions:
        obs, _, _, _, _ = env.step(a)
        old_obs.append(np.copy(obs))
    old_obs = np.array(old_obs)

    restore_mjdata(env, snap)
    new_obs = []
    for a in actions:
        obs, _, _, _, _ = env.step(a)
        new_obs.append(np.copy(obs))
    new_obs = np.array(new_obs)

    print(np.mean(new_obs - old_obs, axis = 1))


if __name__ == '__main__':
    main()