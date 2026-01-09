#!/bin/bash

python -u src/run_changestates.py --config-path src/config_hopper_vanilla_ppo.json --seed 1
python -u src/run_changestates.py --config-path src/config_hopper_vanilla_ppo.json --seed 42


python -u src/run_changestates.py --config-path src/config_walker_vanilla_ppo.json --seed 1
python -u src/run_changestates.py --config-path src/config_walker_vanilla_ppo.json --seed 42