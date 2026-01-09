import os
import glob

# Training tasks
def condor_train_tasks(root_dir = 'run_exp_inputs/'):
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)
    
    for seed in [1, 23, 42, 233, 234, 236, 667, 888][4:]:
        for env in ['hopper', 'halfcheetah', 'walker', 'humanoid'][:]:
            for algo in ['robust_ppo_convex', 'robust_ppo_sgld', 'vanilla_ppo'][:]:
                content = []
                content.append(f'src/config_{env}_{algo}.json\n')
                content.append(f'{seed}\n')
                fn = os.path.join(root_dir, f'{env}_{algo}_seed{seed}.input')
                with open(fn, 'w') as f:
                    f.writelines(content)


# Testing tasks
def condor_test_tasks(exp_root_dir = 'exps/', algo = 'robust_ppo_sgld', test_task_root_dir = 'test_exp_inputs/'):
    eps = {
        'Walker2d-v5' : 0.05,
        'HalfCheetah-v5' : 0.075,
        'Hopper-v5' : 0.075,
        'Humanoid-v5' : 0.075
    }
    
    test_task_root_path = os.path.join(test_task_root_dir, algo)
    if not os.path.exists(test_task_root_path):
        os.makedirs(test_task_root_path)

    for path in glob.glob(os.path.join(exp_root_dir, '*', algo, 'agents', '*')):
        split_strs = path.split('/')
        env_id, exp_id = split_strs[1], split_strs[4]
        if env_id not in eps.keys():
            continue
        # previous_test_folders = glob.glob(os.path.join(path, 'attack*'))
        # if len(previous_test_folders) > 0:
        #     continue
        # print(path, split_strs)
        with open(os.path.join(test_task_root_path, f'{env_id}_{exp_id}.input'), 'w') as f:
            content = []
            content.append(f'src/config_{env_id[:-3].lower() if env_id!="Walker2d-v5" else "walker"}_{algo}.json' + '\n')
            content.append(exp_id + '\n')
            content.append(f'{eps[env_id]}' + '\n')
            content.append(path + '/sarsa.model' + '\n')
            f.writelines(content)


if __name__ == '__main__':
    # condor_train_tasks()
    condor_test_tasks(algo = 'robust_ppo_convex')
    condor_test_tasks(algo = 'robust_ppo_sgld')
    condor_test_tasks(algo = 'vanilla_ppo')