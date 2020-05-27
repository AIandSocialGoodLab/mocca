# mocca

#Files:

game/run_ddpg_single.py - driver of ddpg algorithm (only defender is trained)

game/run_sac_single.py - driver of sac algorithm (only defender is trained)

game/run_sac_multi.py - driver of multiagent sac algorithm (both attacker and defender are trained) (this is deprecated now)

game/run_sac_multi_stealth.py - key driver of multiagent sac algorithm used in experiments

game/algo_ddpg - ddpg implementation for defender

game/algo_mameta  - multiagent sac implementation of defender and attacker

game/algo_mameta/def_sac_alternative - same as def_sac but with standard deviation same as ongoing trained policy

game/algo_sac - sac implementation for defender

game/data - experiments results

game/model - trained model

game/plots - plots generated when running experiments

game/envs - different environments used in the experiments

game/envs/2d_ma_catcher_v7 - training environment used in experiments

game/envs/2d_ma_catcher_v7_test - testing environment used in experiments

game/envs/2d_ma_catcher_v8 - contains reward shaping for CycleMeta

game/envs/2d_ma_catcher_v8_test - testing environment for CycleMeta

#Key variables:



To run experiments, 
    python3 run_sac_multi_stealth.py


referneces:
The sac code is modified from this implementation: https://github.com/openai/spinningup/tree/master/spinup/algos/tf1/sac