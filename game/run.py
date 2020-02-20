import gym
import envs
from algo.ddpg import DDPG


def main():
    # env = gym.make('Pushing2D-v0')
    # env = gym.make('Point2D-v0')
    # env = gym.make('Catcher2D-v0')
    env = gym.make('Catcher2D-v1')
    hindsight = False
    if hindsight:
        algo = DDPG(env, 'ddpg_log_her.txt')
    else:
        algo = DDPG(env, 'ddpg_log.txt')
    algo.train(50000, hindsight=hindsight)


if __name__ == '__main__':
    main()
