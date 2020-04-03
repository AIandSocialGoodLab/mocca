import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import tensorflow.keras.backend as K

from .ReplayBuffer import ReplayBuffer
from .ActorNetwork import ActorNetwork
from .CriticNetwork import CriticNetwork

BUFFER_SIZE = 1000000
BATCH_SIZE = 256
GAMMA = 0.99                    # Discount for rewards.
TAU = 0.05                      # Target network update rate.
LEARNING_RATE_ACTOR =  0.0001 
LEARNING_RATE_CRITIC = 0.0005 


class EpsilonNormalActionNoise(object):
    """A class for adding noise to the actions for exploration."""

    # TODO: the input shapes are hard coded, fix this   
    # def __init__(self, mu=np.zeros(4), sigma=0.01 * np.ones(4), epsilon=0.1): 
    def __init__(self, mu=np.zeros(4), sigma=0.05 * np.ones(4), epsilon=0.3):
        """Initialize the class.

        Args:
            mu: (float) mean of the noise (probably 0).
            sigma: (float) std dev of the noise.
            epsilon: (float) probability in range [0, 1] with
            which to add noise.
        """
        self.mu = mu
        self.sigma = sigma
        self.epsilon = epsilon

    def __call__(self, action):
        """With probability epsilon, adds random noise to the action.
        Args:
            action: a batched tensor storing the action.
        Returns:
            noisy_action: a batched tensor storing the action.
        """
        if np.random.uniform() > self.epsilon:
            return action + np.random.normal(self.mu, self.sigma)
        else:
            return np.random.uniform(-1.0, 1.0, size=action.shape)


class DDPG(object):
    """A class for running the DDPG algorithm."""

    def __init__(self, env, outfile_name):
        """Initialize the DDPG object.

        Args:
            env: an instance of gym.Env on which we aim to learn a policy.
            outfile_name: (str) name of the output filename.
        """
        action_dim = len(env.action_space.low)
        state_dim = len(env.observation_space.low)
        np.random.seed(15457)
        self.outfile_name = outfile_name
        self.env = env



        
        tf.set_random_seed(15457)

        self.sess = tf.Session()
        tf.keras.backend.set_session(self.sess)

        self.replaybuffer = ReplayBuffer(BUFFER_SIZE)

        K.set_session(self.sess)

        # sess, state_size, action_size, batch_size, tau, learning_rate
        self.actor = ActorNetwork(self.sess, state_dim, action_dim, BATCH_SIZE, TAU, LEARNING_RATE_ACTOR)

        # sess, state_size, action_size, batch_size, tau, learning_rate
        # note: batch_size is not used?
        self.critic = CriticNetwork(self.sess, state_dim, action_dim, BATCH_SIZE, TAU, LEARNING_RATE_CRITIC)

        self.noise = EpsilonNormalActionNoise()

        self.tempCounter = 0

        self.rewardRecord = []


    def evaluate(self, num_episodes):
        """Evaluate the policy. Noise is not added during evaluation.

        Args:
            num_episodes: (int) number of evaluation episodes.
        Returns:
            success_rate: (float) fraction of episodes that were successful.
            average_return: (float) Average cumulative return.
        """
        self.tempCounter += 1
        test_rewards = []
        success_vec = []
        plt.figure(figsize=(12, 12))
        for i in range(num_episodes):
            s_vec = []
            state = self.env.reset()
            s_t = np.array(state)
            total_reward = 0.0
            done = False
            step = 0
            success = False
            while not done:
                s_vec.append(s_t)
                a_t = self.actor.actor.predict(s_t[None])[0]
                new_s, r_t, done, info = self.env.step(a_t)
                if done and "goal" in info["done"]:
                    success = True
                new_s = np.array(new_s)
                total_reward += r_t
                s_t = new_s
                step += 1
            success_vec.append(success)
            test_rewards.append(total_reward)

            if i < 9:
                plt.subplot(3, 3, i+1)
                s_vec = np.array(s_vec)
                pusher_vec = s_vec[:, :2]
                puck_vec = s_vec[:, 2:4]
                goal_vec = s_vec[:, 4:]
                plt.plot(pusher_vec[:, 0], pusher_vec[:, 1], '-o', label='pusher')
                plt.plot(puck_vec[:, 0], puck_vec[:, 1], '-o', label='puck')
                plt.plot(goal_vec[:, 0], goal_vec[:, 1], '*', label='goal', markersize=10)
                # plt.plot([0, 5, 5, 0, 0], [0, 0, 5, 5, 0], 'k-', linewidth=3)
                plt.fill_between([-1, 6], [-1, -1], [6, 6], alpha=0.1,
                                 color='g' if success else 'r')
                plt.xlim([-1, 6])
                plt.ylim([-1, 6])
                if i == 0:
                    plt.legend(loc='lower left', fontsize=28, ncol=3, bbox_to_anchor=(0.1, 1.0))
                if i == 8:
                    # Comment out the line below to disable plotting.
                    # plt.show()
                    plt.savefig('plots/myplot_' + str(self.tempCounter)+'.png')
                    # pass
        print('evaluating, r_t is', test_rewards)

        self.rewardRecord.append(test_rewards)
        if self.tempCounter % 5 == 0:
            np.savetxt("data/rewards_"+str(self.tempCounter)+".csv", self.rewardRecord, delimiter=",")

        return np.mean(success_vec), np.mean(test_rewards)


    def evaluate_catcher(self, num_episodes):
        """Evaluate the policy. Noise is not added during evaluation.

        Args:
            num_episodes: (int) number of evaluation episodes.
        Returns:
            success_rate: (float) fraction of episodes that were successful.
            average_return: (float) Average cumulative return.
        """
        self.tempCounter += 1
        test_rewards = []
        success_vec = []
        plt.figure(figsize=(12, 12))
        for i in range(num_episodes):
            s_vec = []
            att_vec = []
            state = self.env.reset()
            s_t = np.array(state)
            total_reward = 0.0
            done = False
            step = 0
            success = False
            while not done:
                s_vec.append(s_t)
                att_vec.append(self.env.getAttState())
                a_t = self.actor.actor.predict(s_t[None])[0]
                new_s, r_t, done, info = self.env.step(a_t)
                if done and "attacker caught" in info["done"]:
                    success = True
                new_s = np.array(new_s)
                total_reward += r_t
                s_t = new_s
                step += 1
            success_vec.append(success)
            test_rewards.append(total_reward)

            if i < 9:
                plt.subplot(3, 3, i+1)
                s_vec = np.array(s_vec)
                def_vec = s_vec[:, :2]
                uav_vec = s_vec[:, 2:4]
                att_vec = np.array(att_vec)
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
                    plt.savefig('plots/myplot_' + str(self.tempCounter)+'.png')
                    # pass
        print('evaluating, r_t is', test_rewards)

        self.rewardRecord.append(test_rewards)
        if self.tempCounter % 5 == 0:
            np.savetxt("data/rewards_"+str(self.tempCounter)+".csv", self.rewardRecord, delimiter=",")

        return np.mean(success_vec), np.mean(test_rewards)

    def train(self, num_episodes, hindsight=False):
        """Runs the DDPG algorithm.

        Args:
            num_episodes: (int) Number of training episodes.
            hindsight: (bool) Whether to use HER.
        """
        tdLossAry = []
        sumQLossAry = []

        for i in range(num_episodes):
            state = self.env.reset()
            s_t = np.array(state)
            total_reward = 0.0
            done = False
            step = 0
            loss = 0
            sumTargetQ = 0
            store_states = []
            store_actions = []
            while not done:
                step += 1
                # select action and add noise
                a_t = self.noise(self.actor.actor.predict(s_t[None])[0])
                # Collect one episode of experience, saving the states and actions
                # to store_states and store_actions, respectively.
                store_states.append(s_t.copy()) # ! has to copy, or apply_hindsight() will unintentionally change states in replaybuffer
                store_actions.append(a_t.copy())

                # perform training here
                # execute action a_t and get reward, new state s_{t+1}
                s_t1, r_t, done, info = self.env.step(a_t)

               
                total_reward += r_t
                # store transition in replay buffer
                # add(self, state, action, reward, new_state, done)
                self.replaybuffer.add(s_t, a_t, r_t, s_t1, done)

                # sample a random batch of N and compute y_i
                randomBatch = self.replaybuffer.get_batch(BATCH_SIZE)

                # -------------- old iterative predict ---------------------
                # yis = []
                # sis = []
                # ais = []

                # for j in range(len(randomBatch)):
                #     (s_i, a_i, r_i, s_i1, done_i) = randomBatch[j]
                #     mu_a = self.actor.actor_target.predict(s_i1[None])
                #     if not done_i:
                #         yi = r_i
                #     else:
                #         yi = r_i + GAMMA * self.critic.critic_target.predict([s_i1[None], mu_a])[0]
                    
                #     yis.append(yi)
                #     sis.append(s_i)
                #     ais.append(a_i)

                # yis = np.array(yis)
                # sis = np.array(sis)
                # ais = np.array(ais)

                # -------------- new matrix predict ---------------------
                # yis = []
                # sis = []
                # ais = []
                # ris = []
                # si1s = []
                # dis = []
                # for j in range(len(randomBatch)):

                #     (s_i, a_i, r_i, s_i1, done_i) = randomBatch[j]

                #     yis.append(0)
                #     sis.append(s_i)
                #     ais.append(a_i)
                #     ris.append(r_i)
                #     si1s.append(s_i1)
                #     dis.append(done_i)

                # yis = np.array(yis)
                # sis = np.array(sis)
                # ais = np.array(ais)
                # ris = np.array(ris)
                # si1s = np.array(si1s)
                # dis = np.array(dis)

                yis = np.array([0 for b in randomBatch])
                sis = np.array([b[0] for b in randomBatch])
                ais = np.array([b[1] for b in randomBatch])
                ris = np.array([b[2] for b in randomBatch])
                si1s = np.array([b[3] for b in randomBatch])
                dis = np.array([b[4] for b in randomBatch])

                targetQ = self.critic.critic_target.predict([si1s, self.actor.actor_target.predict(si1s)])

                sumTargetQ += np.mean(targetQ)

                for k in range(len(randomBatch)):
                    if dis[k]:
                        yis[k] = ris[k]
                    else:
                        yis[k] = ris[k] + GAMMA * targetQ[k,0]

                # train critic
                # self.critic.critic.fit([sis, ais], yis)
                loss += self.critic.critic.train_on_batch([sis,ais],yis)

                # update actor policy using gradients; train actor
                grads = self.critic.gradients(sis, self.actor.actor.predict(sis))

                self.actor.train(sis, grads)

                # update target network
                self.actor.update_target()
                self.critic.update_target()

                # important: update s_t
                s_t = s_t1

            if hindsight:
                # For HER, we also want to save the final next_state.
                store_states.append(s_t1)       # append final state
                self.add_hindsight_replay_experience(store_states, store_actions)

            del store_states, store_actions
            store_states, store_actions = [], []

            # Logging
            print("Episode %d: Total reward = %d" % (i, total_reward))
            print("\tTD loss = %.2f" % (loss / step,))
            print("\tAvg q = %.2f" % (sumTargetQ / step, ))
            print("\tSteps = %d; Info = %s" % (step, info['done']))
            tdLossAry.append(loss/step)
            sumQLossAry.append(sumTargetQ/step)
            if i % 1000 == 0:
            	np.savetxt("data/td_"+str(i)+".csv", tdLossAry, delimiter=",")
            	np.savetxt("data/sumQ_"+str(i)+".csv", sumQLossAry, delimiter=",")

            if i % 100 == 0:
                successes, mean_rewards = self.evaluate_catcher(10)
                print('Evaluation: success = %.2f; return = %.2f' % (successes, mean_rewards))
                with open(self.outfile_name, "a") as f:
                    f.write("%.2f, %.2f,\n" % (successes, mean_rewards))

            
    def add_hindsight_replay_experience(self, states, actions):
        """Relabels a trajectory using HER.

        Args:
            states: a list of states.
            actions: a list of states.
        """
        her_states, her_rewards = self.env.apply_hindsight(states)
        for i in range(len(her_rewards) - 1):
            s_t, s_t1 = her_states[i], her_states[i+1]
            r_t = her_rewards[i]
            a_t = actions[i]
            self.replaybuffer.add(s_t, a_t, r_t, s_t1, False)
        self.replaybuffer.add(her_states[-2], actions[-1], her_rewards[-1], her_states[-1], True)
