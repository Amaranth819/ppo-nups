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

export C_INCLUDE_PATH=$C_LIBRARY_PATH:$HOME/miniconda3/envs/radial-ppo/include
export CUDA_VISIBLE_DEVICES=$( echo $IDS | sed 's/,$//' )
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/xchen168/.mujoco/mujoco210/bin

eval "$(/home/$(whoami)/miniconda3/bin/conda shell.bash hook)"

# if you're using a conda virtual environment uncomment the following
# line and replace your-env-name with the environment name
# conda activate your-env-name
conda activate radial-ppo

export C_INCLUDE_PATH=$C_LIBRARY_PATH:$HOME/miniconda3/envs/radial-ppo/include
python -u src/run.py --config-path ${arr[0]} --seed ${arr[1]}