import os
import glob

# Training tasks
def condor_train_tasks(root_dir = 'run_exp_inputs/'):
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)
    
    idx = 0
    for seed in [1, 23, 42, 233, 234, 236, 667, 888][:4]:
        for env in ['hopper', 'halfcheetah', 'walker', 'humanoid', 'ant'][:3]:
            for algo in ['vanilla_ppo', 'robust_ppo_sgld', 'robust_q_ppo_sgld'][:1]:
                content = []
                content.append(f'config_{env}_{algo}.json\n')
                content.append(f'{seed}\n')
                fn = os.path.join(root_dir, f'{env}_{algo}_seed{seed}.input')
                with open(fn, 'w') as f:
                    f.writelines(content)
                idx += 1
    print(f'Created {idx} jobs!')


# Testing tasks
def condor_test_tasks(exp_root_dir = 'exps/', algo = 'robust_ppo_sgld', test_task_root_dir = 'test_exp_inputs/'):
    eps = {
        'Walker2d-v5' : 0.05,
        'HalfCheetah-v5' : 0.075,
        'Hopper-v5' : 0.075,
        'Humanoid-v5' : 0.075,
        'Ant-v5' : 0.15
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



def condor_nuus_tasks(
    exp_root_dir = 'exps', 
    envs = ['hopper', 'halfcheetah', 'walker2d'][:],
    algo = 'vanilla_ppo', 
    betas = [2.0, 5.0, 10.0, 20.0, 40.0, 80.0, 160.0][:1],
    test_task_root_dir = 'test_nuus_inputs/'
):
    test_task_root_path = os.path.join(test_task_root_dir, algo)
    if not os.path.exists(test_task_root_path):
        os.makedirs(test_task_root_path)

    for env in envs:
        for cfg_path in glob.glob(os.path.join(exp_root_dir, env, algo, 'agents', '*', 'train.json')):
            exp_id = cfg_path.split('/')[-2]

            for beta in betas:
                with open(os.path.join(test_task_root_path, f'{env}_{algo}_{exp_id}_beta={beta}.input'), 'w') as f:
                    content = []
                    content.append(cfg_path + '\n')
                    content.append(exp_id + '\n')
                    content.append(str(beta) + '\n')
                    f.writelines(content)


if __name__ == '__main__':
    # condor_train_tasks()
    condor_nuus_tasks()