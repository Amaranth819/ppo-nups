import numpy as np
from collections import defaultdict
import pandas as pd
import glob
import json
import os


def read_nuus_res(
    res_root_dir = 'nuus_test/',
    algo = 'ppo',
    envs = ['hopper', 'halfcheetah', 'walker2d'],
    attack_methods = ['none', 'random', 'critic', 'action'],
    betas = [2.0, 5.0, 10.0, 20.0, 40.0, 80.0, 160.0],
):
    
    rewards_table = defaultdict(lambda: [])
    d_returns_table = defaultdict(lambda: [])

    for env in envs:
        for attack in attack_methods:
            for beta in betas:
                json_file_paths = glob.glob(os.path.join(res_root_dir, env, algo, '*', attack, f'beta={beta}.json'))
                if len(json_file_paths) == 0:
                    assert ValueError

                for p in json_file_paths:
                    with open(p, 'r') as jf:
                        data = json.load(jf)
                        rewards_table[env, attack, beta].extend(data['results'][attack]['ep_reward'])
                        d_returns_table[env, attack, beta].extend(data['results'][attack]['ep_return'])

    return dict(rewards_table), dict(d_returns_table)



def export_res_latex_table(
    table : dict, 
    envs = ['hopper', 'halfcheetah', 'walker2d'],
    attack_methods = ['none', 'random', 'critic', 'action'],
    betas = [2.0, 5.0, 10.0, 20.0, 40.0, 80.0, 160.0],
    caption = None
):
    latex_table_dict = defaultdict(lambda: [])
    
    for env in envs:
        for idx, beta in enumerate(betas):
            latex_table_dict['Env'].append(env if idx == 0 else "")
            latex_table_dict['Beta'].append(f'{int(beta):d}')
            for attack in attack_methods:
                all_samples = table[env, attack, beta]
                mean, std = np.mean(all_samples).astype(int), np.std(all_samples).astype(int)

                # no attack rewards
                no_attack_mean = np.mean(table[env, 'none', beta]).astype(int)
                diff = mean - no_attack_mean

                latex_table_dict[attack].append(f'{mean:d} $\pm$ {std:d} ({diff:d})')

    latex_table = pd.DataFrame(dict(latex_table_dict))
    print(latex_table.to_latex(index = False, caption = caption if caption is not None else ""))



if __name__ == '__main__':
    algo = 'ppo'
    envs = ['hopper']
    attack_methods = ['none', 'random', 'critic', 'action']
    betas = [2.0]

    reward_table, d_return_table = read_nuus_res(res_root_dir = 'nuus_test', algo = algo, envs = envs, attack_methods = attack_methods, betas = betas)
    export_res_latex_table(d_return_table, envs = envs, attack_methods = attack_methods, betas = betas, caption = 'Discounted total return + PPO')