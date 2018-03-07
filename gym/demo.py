#!/usr/bin/env python3

import argparse
import gym
import gym_minos

import time
import sys
import traceback

from minos.config import sim_config
from minos.config.sim_args import add_sim_args_basic

import scipy.misc
from PIL import Image
import numpy as np

actions_dict = {'forwards': [0, 0, 1],  'turnLeft': [1, 0, 0], 'turnRight':[0, 1, 0], 'idle':[0, 0, 0]};

def run_gym(sim_args):
    env = gym.make('indoor-v0')
    env.configure(sim_args)
    try:
        print('Running MINOS gym example')
        for i_episode in range(100):
            time.sleep(1)
            print('Starting episode %d' % i_episode)
            observation = env.reset()
            done = False
            num_steps = 0
            name = sim_args['env_config'].split('_')[-1]
            while not done:
                img_prefix = 'pics/'
                actions_str = [None] + ['turnRight']*5
                for action_i, action_str in enumerate(actions_str):
                    if action_str is not None:
                        observation, reward, done, info = env.step(actions_dict[action_str])
                    env.render(mode='human')
                    time.sleep(1)
                    img = observation['observation']['sensors']['color'].get('data')
                    #print(img, img.shape)
                    #im = Image.fromarray(img)
                    #im.save(name + '_' +img_prefix + str(i_episode) + '_' + str(action_i) + 'img' + '.png')
                    scipy.misc.imsave(name + '_' +img_prefix + str(i_episode) + '_' + str(action_i) + 'img' + '.png', img)
                    np.savetxt(name + '_' +img_prefix + str(i_episode) + '_' + str(action_i) + 'img' + '1.txt', img[:, :, 0].astype(np.uint),  fmt='%d')
                    np.savetxt(name + '_' +img_prefix + str(i_episode) + '_' + str(action_i) + 'img' + '2.txt', img[:, :, 1].astype(np.uint),  fmt='%d')
                    np.savetxt(name + '_' +img_prefix + str(i_episode) + '_' + str(action_i) + 'img' + '3.txt', img[:, :, 2].astype(np.uint),  fmt='%d')
                    depth = observation['observation']['sensors']['depth']['data']
                    #print(depth.shape)
                    depth *= (255.0 / depth.max())  # naive rescaling for visualization
                    depth = depth.astype(np.uint8)
                    scipy.misc.toimage(depth, cmin=0, cmax=255).save(name + '_' +img_prefix + str(i_episode) + '_' + str(action_i) +'depth' + '.png')
                    if 'suncg'==name:
                        pass
                        #objectType = observation['observation']['sensors']['objectType'].get('data_viz')
                        #print(objectType)
                        #scipy.misc.toimage(objectType, cmin=0, cmax=objectType[:,:,0:2].max()).save(name + '_' +img_prefix + str(i_episode)+ '_' + str(action_i)  + 'objectType' + '.png')
                num_steps += 1
                done = True
                # if done:
                #     print("Episode finished after {} steps; success={}".format(num_steps, observation['success']))
                #     break
    except Exception as e:
        print(traceback.format_exc())
    finally:
        env._close()



def main():
    parser = argparse.ArgumentParser(description='MINOS gym wrapper')
    add_sim_args_basic(parser)
    parser.add_argument('--env_config',
                        default='objectgoal_suncg_sf',
                        help='Environment configuration file')
    args = parser.parse_args()
    sim_args = sim_config.get(args.env_config, vars(args))

    run_gym(sim_args)


if __name__ == "__main__":
    main()
