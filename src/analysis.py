def print_nups_res(
    root_dir = './test_nups_b1/', 
    kl_bound = 'b1',
    plot_total_rewards = False,
    plot_discounted_returns = True,
):
    envs = ['Hopper-v5', 'HalfCheetah-v5', 'Walker2d-v5']
    betas = [10.0, 25.0, 50.0, 100.0, 200.0, 400.0, 800.0, 1600.0]
    attack_methods = ['none', 'random', 'critic', 'action']
    
    rewards_table = defaultdict(lambda: [])
    d_returns_table = defaultdict(lambda: [])

    for env in envs:
        for beta in betas:
            json_file_paths = glob.glob(os.path.join(root_dir, f'{env}*klbound={kl_bound}_beta={beta}.json'))
            if len(json_file_paths) == 0:
                continue
            all_rewards = defaultdict(lambda: [])
            all_d_returns = defaultdict(lambda: [])
            for p in json_file_paths:
                with open(p, 'r') as jf:
                    data = json.load(jf)
                    for atk in attack_methods:
                        all_rewards[atk].extend(data['sampled_ep_rewards'][atk])
                        all_d_returns[atk].extend(data['sampled_ep_discounted_returns'][atk])
            
            if plot_total_rewards:
                rewards_table['Env'].append(env)
                rewards_table['Beta'].append(f'{int(beta):d}')
                for atk in attack_methods:
                    rewards_mean, rewards_std = np.mean(all_rewards[atk]).astype(int), np.std(all_rewards[atk]).astype(int)

                    # no attack rewards
                    no_attack_rewards_mean = np.mean(all_rewards['none']).astype(int)
                    diff = rewards_mean - no_attack_rewards_mean

                    rewards_table[atk].append(f'{rewards_mean:d} $\pm$ {rewards_std:d} ({diff:d})')


            if plot_discounted_returns:
                d_returns_table['Env'].append(env)
                d_returns_table['Beta'].append(f'{int(beta):d}')
                for atk in attack_methods:
                    d_returns_mean, d_returns_std = np.mean(all_d_returns[atk]).astype(int), np.std(all_d_returns[atk]).astype(int)

                    # no attack returns
                    no_attack_d_returns_mean = np.mean(all_d_returns['none']).astype(int)
                    diff = d_returns_mean - no_attack_d_returns_mean

                    d_returns_table[atk].append(f'{d_returns_mean:d} $\pm$ {d_returns_std:d} ({diff:d})')

    if plot_total_rewards:
        rewards_table = pd.DataFrame(dict(rewards_table))
        print(rewards_table.to_latex(index = False, caption = 'Total rewards $\sum_t R(s_t,a_t)$'))
        print()

    if plot_discounted_returns:
        d_returns_table = pd.DataFrame(dict(d_returns_table))
        print(d_returns_table.to_latex(index = False, caption = 'Discounted returns $\sum_t \gamma^t R(s_t,a_t)$'))