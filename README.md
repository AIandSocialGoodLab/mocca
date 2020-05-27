# mocca

# files:

game/run_ddpg_single.py - driver of ddpg algorithm (only defender is trained)

game/run_sac_single.py - driver of sac algorithm (only defender is trained)

game/run_sac_multi.py - driver of multiagent sac algorithm (both attacker and defender are trained) (this is deprecated now)

game/run_sac_multi_stealth.py - key driver of multiagent sac algorithm used in experiments

game/algo_ddpg - ddpg implementation for defender

game/algo_mameta  - multiagent sac implementation of defender and attacker

game/algo_mameta/def_sac_alternative - same as def_sac but with standard deviation same as ongoing trained policy

game/algo_sac - sac implementation for defender

game/abstractGameLP/createGraph_v3 - LP meta strategy

game/abstractGameLP/createGraph_v2 - deprecated

game/abstractGameLP/createGraph - deprecated

game/data - experiments results

game/model - trained model

game/plots - plots generated when running experiments

game/envs - different environments used in the experiments

game/envs/2d_ma_catcher_v7 - training environment used in experiments

game/envs/2d_ma_catcher_v7_test - testing environment used in experiments

game/envs/2d_ma_catcher_v8 - contains reward shaping for CycleMeta

game/envs/2d_ma_catcher_v8_test - testing environment for CycleMeta

# key variables:

run_sac_multi_stealth.py
>centralizedQ: whether using centralized Q function

def_sac_alternative.py
>alpha: weight of regulator (if set to 0 becomes ddpg)<br/>
>self.m: flag of whether using meta strategy<br/>
>gamma: discount factor<br/>
>polyak: soft update weight<br/>
>lr: learning rate<br/>
>batch_size: size of batch<br/>
>start_steps: stop random actions after start_steps

To run experiments, 
>python3 run_sac_multi_stealth.py


referneces:
The sac code is modified from this implementation: https://github.com/openai/spinningup/tree/master/spinup/algos/tf1/sac