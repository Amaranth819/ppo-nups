#!/bin/bash
file="$1"
arr=()
while IFS= read -r line
do
  arr+=("$line")
done < "$file"

IDS=""
for id in $( echo $CUDA_VISIBLE_DEVICES | sed 's/,/ /g' )
do
	IDS=${IDS}$( nvidia-smi -L | grep $id | cut -f 2 -d' ' | sed 's/:/,/')
done

export C_INCLUDE_PATH=$C_LIBRARY_PATH:$HOME/miniconda3/envs/sa-ppo/include
export CUDA_VISIBLE_DEVICES=$( echo $IDS | sed 's/,$//' )
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/xchen168/.mujoco/mujoco210/bin

eval "$(/home/$(whoami)/miniconda3/bin/conda shell.bash hook)"

# if you're using a conda virtual environment uncomment the following
# line and replace your-env-name with the environment name
# conda activate your-env-name
conda activate sa-ppo

export C_INCLUDE_PATH=$C_LIBRARY_PATH:$HOME/miniconda3/envs/sa-ppo/include


# # Robust sarsa attack
python src/test_v2.py --config-path ${arr[0]} --exp-id ${arr[1]} --sarsa-enable --sarsa-model-path ${arr[3]}
python src/test_v2.py --config-path ${arr[0]} --exp-id ${arr[1]} --attack-eps ${arr[2]} --attack-method sarsa --attack-sarsa-network sarsa.model --deterministic


# # Maximal Action Difference (MAD) Attack
python src/test_v2.py --config-path ${arr[0]} --exp-id ${arr[1]} --attack-eps ${arr[2]} --attack-method action --deterministic

# # Critic based attack
python src/test_v2.py --config-path ${arr[0]} --exp-id ${arr[1]} --attack-eps ${arr[2]} --attack-method critic --deterministic

# # Random attack
python src/test_v2.py --config-path ${arr[0]} --exp-id ${arr[1]} --attack-eps ${arr[2]} --attack-method random --deterministic


# No attack
python src/test_v2.py --config-path ${arr[0]} --exp-id ${arr[1]} --attack-eps ${arr[2]} --attack-method none --deterministic