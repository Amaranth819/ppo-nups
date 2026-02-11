import os

def generate_nuus_task_scripts(
    algo = 'vanilla_ppo',
    env = 'Hopper-v5',
    betas = [10, 25, 50, 100, 200, 400, 800],
    exp_root_dir = 'exps',
    target_sh_name = 'test_nuus.sh',
    append = True
):

    all_tasks = []

    env_algo_root_path = os.path.join(exp_root_dir, env[:-3].lower(), algo, 'agents')
    for exp_id in os.listdir(env_algo_root_path):
        exp_root_dir = os.path.join(env_algo_root_path, exp_id)
        if os.path.isdir(exp_root_dir):
            cfg_json_path = os.path.join(exp_root_dir, 'train.json')
            for beta in betas:
                task_str = f'python -u test_nuus.py --config-path {cfg_json_path} --exp-id {exp_id} --deterministic --nuus_beta {beta} \n'
                all_tasks.append(task_str)

    if append:
        with open(target_sh_name, 'a') as f:
            f.writelines(all_tasks)
    else:
        with open(target_sh_name, 'w') as f:
            f.writelines(all_tasks)    
    os.chmod(target_sh_name, 0o777)



def generate_sarsa_model_training(
    algo = 'vanilla_ppo',
    env = 'Hopper-v5',
    exp_root_dir = 'exps',
    target_sh_name = 'train_sarsa.sh',
    append = True
):
    all_tasks = []

    env_algo_root_path = os.path.join(exp_root_dir, env[:-3].lower(), algo, 'agents')
    for exp_id in os.listdir(env_algo_root_path):
        exp_root_dir = os.path.join(env_algo_root_path, exp_id)
        if os.path.isdir(exp_root_dir):
            cfg_json_path = os.path.join(exp_root_dir, 'train.json')
            task_str = f'python -u test.py --config-path {cfg_json_path} --exp-id {exp_id} --sarsa-enable --sarsa-model-path {os.path.join(exp_root_dir, "sarsa.model")} \n'
            all_tasks.append(task_str)

    if append:
        with open(target_sh_name, 'a') as f:
            f.writelines(all_tasks)
    else:
        with open(target_sh_name, 'w') as f:
            f.writelines(all_tasks)    
    os.chmod(target_sh_name, 0o777)




if __name__ == '__main__':
    # for algo in ['vanilla_ppo']:
    #     for env in ['Hopper-v5', 'HalfCheetah-v5', 'Walker2d-v5']:
    #         generate_nuus_task_scripts(
    #             algo, 
    #             env, 
    #             betas = [2, 5, 10, 20, 40, 80, 160],
    #             exp_root_dir = 'exps/'
    #         )


    generate_sarsa_model_training()