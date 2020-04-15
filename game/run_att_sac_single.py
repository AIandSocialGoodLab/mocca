"""
this driver is used only for reward shaping for the attacker

other code is deprecated
"""

import envs
import gym
from algo_sac.sac import *

from algo_mameta.def_sac import *
from algo_mameta.att_sac import *

import matplotlib.pyplot as plt


class AttRand:

    def __init__(self, env_fn):
        self.env = env_fn()

    def act(self, o, t):
        return self.env.att_action_space.sample()

    def set_session(self, sess):
        pass



def getDefObs(o_all):

    return o_all[:7]

def getAttObs(o_all):

    return o_all[7:]



class DefRand:

    def __init__(self, env_fn):
        self.env = env_fn()

    def act(self, o, t, deterministic=False):
        return self.env.def_action_space.sample()

    def set_session(self, sess):
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

    load_pretrained = False

    seed = 0
    tf.set_random_seed(0)
    np.random.seed(0)

    # flag for centralized training
    centralizeQ = True

    env = gym.make('MultiAgent-Catcher2D-v1')

    defender = DefSacMeta(lambda : gym.make('MultiAgent-Catcher2D-v1'),
                          ac_kwargs=dict(hidden_sizes=[args.hid]*args.l),
                          gamma=args.gamma, centralizeQ=centralizeQ)

    # attacker = AttRand(lambda : gym.make('MultiAgent-Catcher2D-v0'))

    attacker = AttSacMeta(lambda : gym.make('MultiAgent-Catcher2D-v1'),
                          ac_kwargs=dict(hidden_sizes=[args.hid]*args.l),
                          gamma=args.gamma, centralizeQ=centralizeQ)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    sess.run(defender.target_init)
    sess.run(attacker.target_init)

    defender.set_session(sess)
    attacker.set_session(sess)

    saver = tf.train.Saver()

    # ty: add restore code here

    if load_pretrained:
        attacker.saver.restore(sess, './model/attacker_pretrain')

    steps_per_epoch=4000
    epochs=100
    total_steps = steps_per_epoch * epochs

    o_all, ep_ret, ep_len = env.reset(), 0, 0

    for t in range(total_steps):

        # get def action
        a_def = defender.act(getDefObs(o_all), 0)
        # get att action
        a_att = attacker.act(getAttObs(o_all), t)

        o_all2, r, d, _ = env.step(np.append(a_def, a_att))

        # do not update weights for defender
        # defender.train(getDefObs(o_all), a_def, r, getDefObs(o_all2), d, t, a_att)

        attacker.train(getAttObs(o_all), a_att, -r, getAttObs(o_all2), d, t, a_def)

        ep_ret += r
        ep_len += 1

        o_all = o_all2

        # end of episode
        if (d):
            o_all, ep_ret, ep_len = env.reset(), 0, 0

        # end of epoch
        if (t+1) % steps_per_epoch == 0:

            epoch = (t+1) // steps_per_epoch

            # test after each epoch
            print("epoch: ", epoch)

            if (epoch == 10):
                saver.save(sess, './model/attacker_pretrain')
                print("all weights saved!")

            test_rewards = []
            success_vec = []
            plt.figure(figsize=(12, 12))

            for i in range(10):
                s_vec = []
                att_vec = []
                state = env.reset()

                s_t = np.array(state)
                total_reward = 0.0
                d = False
                step = 0
                success = False
                while not d:
                    s_vec.append(s_t)

                    a_def_test = defender.act(getDefObs(state), 0, deterministic=True)

                    a_att_test = attacker.act(getAttObs(state), t, deterministic=True)

                    o_all2_test, r_test, d, info = env.step(np.append(a_def_test, a_att_test))

                    if d and "attacker caught" in info["done"]:
                        success = True
                    # new_s = np.array(o_all2_test)
                    total_reward += r_test
                    state = o_all2_test

                    s_t = o_all2_test

                success_vec.append(success)
                test_rewards.append(total_reward)

                if i < 9:
                    plt.subplot(3, 3, i+1)
                    s_vec = np.array(s_vec)
                    def_vec = s_vec[:, :2]
                    uav_vec = s_vec[:, 2:4]
                    att_vec = s_vec[:, 7:9]
                    plt.plot(def_vec[:, 0], def_vec[:, 1], '-o', label='def')
                    plt.plot(uav_vec[:, 0], uav_vec[:, 1], '-o', label='uav')
                    plt.plot(att_vec[:, 0], att_vec[:, 1], '*', label='att', markersize=10)
                    plt.plot([0, 5, 5, 0, 0], [0, 0, 5, 5, 0 ], 'k-', linewidth=3)
                    plt.fill_between([-1, 501], [-1, -1], [501, 501], alpha=0.1,
                                     color='g' if success else 'r')
                    plt.xlim([-1, 501])
                    plt.ylim([-1, 501])
                    if i == 0:
                        plt.legend(loc='lower left', fontsize=28, ncol=3, bbox_to_anchor=(0.1, 1.0))
                    if i == 8:
                        # Comment out the line below to disable plotting.
                        # plt.show()
                        plt.savefig('plots/myplot_' + str(epoch)+'.png')



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







