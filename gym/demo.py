#!/usr/bin/env python3

import argparse
import gym
import gym_minos
import matplotlib.pyplot as plt

# kvandy
import time
import sys
import traceback

from minos.config import sim_config
###
from minos.config.sim_args import parse_sim_args

import scipy.misc
from PIL import Image
import numpy as np

actions_dict = {'forwards': [0, 0, 1],  'turnLeft': [1, 0, 0], 'turnRight':[0, 1, 0], 'idle':[0, 0, 0]};

def run_gym(sim_args):
    env = gym.make('indoor-v0')
    env.configure(sim_args)
<<<<<<< HEAD
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
                    img = img.reshape((img.shape[1], img.shape[0], img.shape[2]))
                    scipy.misc.imsave(name + '_' +img_prefix + str(i_episode) + '_' + str(action_i) + '_img' + '.png', img)
                    depth = observation['observation']['sensors']['depth']['data']
                    depth = depth.reshape((depth.shape[1], depth.shape[0]))
                    #print(depth.shape)
                    depth *= (255.0 / depth.max())  # naive rescaling for visualization
                    depth = depth.astype(np.uint8)
                    scipy.misc.toimage(depth, cmin=0, cmax=255).save(name + '_' +img_prefix + str(i_episode) + '_' + str(action_i) +'_depth' + '.png')
                    if 'suncg'==name:
                        pass
                        #objectType = observation['observation']['sensors']['objectType'].get('data_viz')
                        #print(objectType)
                        #scipy.misc.toimage(objectType, cmin=0, cmax=objectType[:,:,0:2].max()).save(name + '_' +img_prefix + str(i_episode)+ '_' + str(action_i)  + 'objectType' + '.png')
                    ## Msavva demo function
                    #if sim_args.save_observations:
                    #    save_observations(observation, sim_args)
                num_steps += 1
                done = True
                # if done:
                #     print("Episode finished after {} steps; success={}".format(num_steps, observation['success']))
                #     break
    except Exception as e:
        print(traceback.format_exc())
    finally:
        env._close()


def save_observations(observation, sim_args):
    if sim_args.observations.get('color'):
        color = observation["observation"]["sensors"]["color"]["data"]
        plt.imsave('color.png', color)

    if sim_args.observations.get('depth'):
        depth = observation["observation"]["sensors"]["depth"]["data"]
        plt.imsave('depth.png', depth, cmap='Greys')

    if sim_args.observations.get('normal'):
        normal = observation["observation"]["sensors"]["normal"]["data"]
        plt.imsave('normal.png', normal)

    if sim_args.observations.get('objectId'):
        object_id = observation["observation"]["sensors"]["objectId"]["data"]
        plt.imsave('object_id.png', object_id)

    if sim_args.observations.get('objectType'):
        object_type = observation["observation"]["sensors"]["objectType"]["data"]
        plt.imsave('object_type.png', object_type)

    if sim_args.observations.get('roomId'):
        room_id = observation["observation"]["sensors"]["roomId"]["data"]
        plt.imsave('room_id.png', room_id)

    if sim_args.observations.get('roomType'):
        room_type = observation["observation"]["sensors"]["roomType"]["data"]
        plt.imsave('room_type.png', room_type)

    if sim_args.observations.get('map'):
        nav_map = observation["observation"]["map"]["data"]
        nav_map.shape = (nav_map.shape[1], nav_map.shape[0], nav_map.shape[2])
        plt.imsave('nav_map.png', nav_map)

    shortest_path = observation["observation"]["measurements"]["shortest_path_to_goal"]
    print(shortest_path)


def main():
    parser = argparse.ArgumentParser(description='MINOS gym wrapper')
    parser.add_argument('--save_observations', action='store_true',
                        default=False,
                        help='Save sensor observations at each step to images')
    args = parse_sim_args(parser)
    run_gym(args)


if __name__ == "__main__":
    main()
