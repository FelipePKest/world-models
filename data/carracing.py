"""
Generating data from the CarRacing gym environment.
!!! DOES NOT WORK ON TITANIC, DO IT AT HOME, THEN SCP !!!
"""
import argparse
from os.path import join, exists
import gymnasium as gym
import numpy as np
from utils.misc import sample_continuous_policy
import os
# from datakey_utils import getkey
from data.key_utils import getkey

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def generate_data(rollouts, data_dir, noise_type): # pylint: disable=R0914
    """ Generates data """
    assert exists(data_dir), "The data directory does not exist..."

    env = gym.make("CarRacing-v2")
    seq_len = 1000

    for i in range(rollouts):
        env.reset()
        # env.env.viewer.window.dispatch_events()
        # env.unwrapped.viewer.window.dispatch_events()
        # env.env.get_wrapper_attr('viewer').window.dispatch_events()
        
        if noise_type == 'white':
            a_rollout = [env.action_space.sample() for _ in range(seq_len)]
        elif noise_type == 'brown':
            a_rollout = sample_continuous_policy(env.action_space, seq_len, 1. / 50)

        s_rollout = []
        r_rollout = []
        d_rollout = []

        ii = None
        t = 0
        while True:
            # Check for keyboard interrupt
            if getkey() != '':
                print("Keyboard interrupt detected, ending rollout...")
                break

            action = a_rollout[t]
            t += 1

            print(env.step(action))
            s, r, done, _, _ = env.step(action)
            # env.viewer.window.dispatch_events()
            s_rollout += [s]
            r_rollout += [r]
            d_rollout += [done]
            if done:
                print("> End of rollout {}, {} frames...".format(i, len(s_rollout)))
                np.savez(join(data_dir, 'rollout_{}'.format(i)),
                         observations=np.array(s_rollout),
                         rewards=np.array(r_rollout),
                         actions=np.array(a_rollout),
                         terminals=np.array(d_rollout))
                break
        env.close()
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--rollouts', type=int, help="Number of rollouts")
    parser.add_argument('--dir', type=str, help="Where to place rollouts")
    parser.add_argument('--policy', type=str, choices=['white', 'brown'],
                        help='Noise type used for action sampling.',
                        default='brown')
    args = parser.parse_args()
    generate_data(args.rollouts, args.dir, args.policy)
