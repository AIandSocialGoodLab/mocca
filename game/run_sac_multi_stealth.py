import envs
import gym

from algo_mameta.def_sac import *
from algo_mameta.att_sac import *
from algo_mameta.def_heur import *
from algo_mameta.def_hard import *
from algo_mameta.def_sac_alternative import *

import matplotlib.pyplot as plt

# note: only need this line when residual training
# from abstractGameLP.createGraph_v2 import *

# from abstractGameLP.createGraph_v3 import *

class AttRand:

    def __init__(self, env_fn):
        self.env = env_fn()

    def act(self, o, t, deterministic=False):
        return self.env.att_action_space.sample()

    def set_session(self, sess):
        pass

class AttFix:

    def __init__(self, env_fn):
        self.env = env_fn()

    def act(self, o, t, deterministic=True):
        return np.array([20.0, 20.0])

    def set_session(self,sess):
        pass

    def train(*args):
        pass



"""
single version:
Number of parameters:    pi: 69896,      q1: 69121,      q2: 69121,      total: 416276


Number of parameters:    pi: 67588,      q1: 67329,      q2: 67329,      total: 404492
"""

"""
centralized version:
Number of parameters:    pi: 69896,      q1: 69633,      q2: 69633,      total: 418324


Number of parameters:    pi: 67588,      q1: 68353,      q2: 68353,      total: 408588
"""

def sacMeta(args):


    # initialize env
    # initialize defender
    # initialize attacker

    # initialize tf session

    

    seed = 0
    tf.set_random_seed(0)
    np.random.seed(0)

    # flag for centralized training
    centralizeQ = True

    env = gym.make('MultiAgent-Catcher2D-v7')

    test_env = gym.make('MultiAgent-Catcher2DTest-v7')

    # ty: now this should only be used by the cycle shaped defender
    # defender = DefSacMeta(lambda : gym.make('MultiAgent-Catcher2D-v8'),
    #                       ac_kwargs=dict(hidden_sizes=[args.hid]*args.l),
    #                       gamma=args.gamma, centralizeQ=centralizeQ)

    
    # defender = DefHard(lambda : gym.make('MultiAgent-Catcher2D-v7'))

    # ty: now this should be used by every other defender
    defender = DefSacAlternativeMeta(lambda : gym.make('MultiAgent-Catcher2D-v7'),
                          ac_kwargs=dict(hidden_sizes=[args.hid]*args.l),
                          gamma=args.gamma, centralizeQ=centralizeQ)


    attacker = AttSacMeta(lambda : gym.make('MultiAgent-Catcher2D-v7'),
                          ac_kwargs=dict(hidden_sizes=[args.hid]*args.l),
                          gamma=args.gamma, centralizeQ=centralizeQ)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
     # ty: IMPORTANT need this line to train agent
    sess.run(defender.target_init)
    sess.run(attacker.target_init)

    defender.set_session(sess)
    attacker.set_session(sess)
    
    saver = tf.train.Saver()

    steps_per_epoch=4000
    epochs=101
    total_steps = steps_per_epoch * epochs

    o_all, ep_ret, ep_len = env.reset(), 0, 0

   
    def a_att_fix(a_att, env):
        for i in range(env.num_target):
            tarX, tarY = env.tarPoss[i]
            attX, attY = env.state[-2:]
            if env._compDist(attX, attY, tarX, tarY) < 15.0:
                return np.array([0.0,0.0])
        return a_att

    def a_def_residual(a_def, env, o_all):

        return a_def + obs2mu(env.getDefObs(o_all)) 

    defRewardRecord = []

    for t in range(total_steps):

        # get def action
        a_def = defender.act(env.getDefObs(o_all), t)
        # get att action
        a_att = attacker.act(env.getAttObs(o_all), t)

        # ty: hardcoding a_att to ensure attacker get enough rewards
        a_att = a_att_fix(a_att, env)

        # a_def = a_def_residual(a_def, env, o_all)

        # todo: change attacker reward for out of boundary
        o_all2, (def_r, att_r), d, info = env.step(np.append(a_def, a_att))

        # def_r,att_r = get_reward(info, r)
        # a_def *= 2
        defender.train(env.getDefObs(o_all), a_def, def_r, env.getDefObs(o_all2), d, t, a_att)

        attacker.train(env.getAttObs(o_all), a_att, att_r, env.getAttObs(o_all2), d, t, a_def)

        ep_len += 1

        o_all = o_all2

        # end of episode
        if (d):
            o_all, ep_ret, ep_len = env.reset(), 0, 0

        # end of epoch -> do evaluation
        if (t+1) % steps_per_epoch == 0:

            epoch = (t+1) // steps_per_epoch

            # test after each epoch
            print("epoch: ", epoch)

            test_rewards = []
            success_vec = []
            plt.figure(figsize=(12, 12))

            for i in range(10):
                s_vec = []
                att_vec = []
                state = test_env.reset()

                s_t = np.array(state)

                total_reward = 0.0
                d = False
                step = 0
                success = False
                while not d:
                    s_vec.append(s_t)

                    a_def_test = defender.act(test_env.getDefObs(state), t, deterministic=True)

                    a_att_test = attacker.act(test_env.getAttObs(state), t, deterministic=True)

                    a_att_test = a_att_fix(a_att_test, test_env)

                    # a_def_test = a_def_residual(a_def_test, env, state)

                    # a_def_test *= 2
                    
                    o_all2_test, (r_def_test,r_att_test), d, info = test_env.step(np.append(a_def_test, a_att_test))

                    total_reward += r_def_test
                    state = o_all2_test

                    s_t = o_all2_test

                    if d and ("attacker caught" in info["done"]):
                        color = 'g'
                    elif d and ("target attacked" in info["done"]):
                        color = 'r'
                    elif d:
                        color = 'w'

                success_vec.append(success)
                test_rewards.append(total_reward)

                if i < 9:
                    plt.subplot(3, 3, i+1)
                    s_vec = np.array(s_vec)
                    def_vec = s_vec[:, :2]
                    uav_vec = s_vec[:, 2:4]
                    att_vec = s_vec[:, -2:]
                    plt.plot(def_vec[:, 0], def_vec[:, 1], '-o', label='def')
                    plt.plot(uav_vec[:, 0], uav_vec[:, 1], '-o', label='uav')
                    plt.plot(att_vec[:, 0], att_vec[:, 1], '-*', label='att', markersize=10)
                    plt.plot(env.tarPossNdarray[:,0], env.tarPossNdarray[:,1], 'o', label="targets")
                    plt.plot([0, 5, 5, 0, 0], [0, 0, 5, 5, 0 ], 'k-', linewidth=3)
                    plt.fill_between([-1, 501], [-1, -1], [501, 501], alpha=0.1,
                                     color=color)
                    plt.xlim([-1, 501])
                    plt.ylim([-1, 501])
                    if i == 0:
                        plt.legend(loc='lower left', fontsize=28, ncol=3, bbox_to_anchor=(0.1, 1.0))
                    if i == 8:
                        # Comment out the line below to disable plotting.
                        # plt.show()
                        plt.savefig('plots/myplot_' + str(epoch)+'.png')

            defRewardRecord.append(test_rewards)
            if epoch % 10 == 1:
                # save def reward to csv file
                np.savetxt("data/rewards_"+str(epoch)+".csv", defRewardRecord, delimiter=",")

            if (epoch == 101 or epoch == 100 or epoch == 99 or epoch == 98 or epoch == 97):
                saver.save(sess, save_path='./model/LPMeta1_' + str(epoch))
                print("weights saved")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    # parser.add_argument('--env', type=str, default='BipedalWalker-v3')
    parser.add_argument('--hid', type=int, default=256)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--exp_name', type=str, default='sac')
    args = parser.parse_args()

    # from spinup.utils.run_utils import setup_logger_kwargs
    # logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    
    sacMeta(args)






"""
expectation:

milp guiding training

multi type attacker
"""







