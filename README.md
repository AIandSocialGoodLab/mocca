# mocca

This repo contains the experiments for defender, attacker modeling.

game/run_ddpg_single.py - driver of ddpg algorithm (only defender is trained)

game/run_sac_single.py - driver of sac algorithm (only defender is trained)

game/run_sac_multi.py - driver of multiagent sac algorithm (both attacker and defender are trained)

game/algo_ddpg - ddpg implementation for defender

game/algo_mameta  - multiagent sac implementation of defender and attacker

    game/algo_mameta/def_sac_alternative

game/algo_sac - sac implementation for defender

game/data - experiments results

envs - different environments used in the experiments

plots - plots generated when running experiments

To run experiments, 

<code>
    python3 run_ddpg_single.py
</code>

<code>
    python3 run_sac_single.py
</code>

<code>
    python3 run_sac_multi.py
</code>

referneces:
The sac code is modified from this implementation: https://github.com/openai/spinningup/tree/master/spinup/algos/tf1/sac