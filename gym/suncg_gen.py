#!/usr/bin/env python3

import argparse
import gym
import gym_minos
import matplotlib.pyplot as plt

# kvandy
import os
import time
import sys
import traceback
import numpy as np

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
    try:
        root_dir = '/root/hdd/data/'
        name = 'mp3d'  if 'mp3d' in sim_args['env_config'] else 'suncg'
        w, h = sim_args['width'], sim_args['height']
        img_suffix = 'pics_{0}x{1}'.format(w, h)
        img_dir = os.path.join(root_dir, name + '_' + img_suffix)
        if not os.path.exists(img_dir):
            os.makedirs(img_dir)
        print('Running MINOS gym generator of semantic segmentation')
        actions_str = ['idle'] + ['turnRight']*3
        offset = 0
        gen_offset = 0 // 4
        for i_episode in range(offset, offset + 20*10**3 // 4 - gen_offset):
            if i_episode % 250:
                print('Starting episode %d' % i_episode)
                sys.stdout.flush()
            observation = env.reset()
            done = False
            num_steps = 0
            while not done:
                for action_i, action_str in enumerate(actions_str):
                    if action_str is not None:
                        observation, reward, done, info = env.step(actions_dict[action_str])
                    #env.render(mode='human')
                    #time.sleep(1)
                    if sim_args.save_observations:
                        save_observations(observation, sim_args,
                             prename = os.path.join(img_dir, str(env._sim.scene_id) + '_' + str(i_episode) + '_' + str(action_i) + '_'))
                num_steps += 1
                done = True
                # if done:
                #     print("Episode finished after {} steps; success={}".format(num_steps, observation['success']))
                #     break
    except Exception as e:
        print(traceback.format_exc())
    finally:
        env._close()


def save_observations(observation, sim_args, prename):
    if sim_args.observations.get('objectType'):
        object_type = observation["observation"]["sensors"]["objectType"]["data"][:, :, 2]
        if len(np.unique(object_type)) < 3:
            print(prename + ' rejected!')
            sys.stdout.flush()
            return
        object_type = object_type.reshape((object_type.shape[1], object_type.shape[0]))
        #np.savetxt(prename + 'object_type_labels.txt', object_type, fmt='%d')
        scipy.misc.imsave(prename + 'object_type_labels.png', object_type)
        object_type = observation["observation"]["sensors"]["objectType"]["data_viz"]
        object_type = object_type.reshape((object_type.shape[1], object_type.shape[0], object_type.shape[2]))
        plt.imsave(prename + 'object_type.png', object_type)

    if sim_args.observations.get('color'):
        color = observation["observation"]["sensors"]["color"]["data"]
        color = color.reshape((color.shape[1], color.shape[0], color.shape[2]))
        plt.imsave(prename + 'color.png', color)

    if sim_args.observations.get('depth'):
        depth = observation["observation"]["sensors"]["depth"]["data"]
        depth = depth.reshape((depth.shape[1], depth.shape[0]))
        plt.imsave(prename + 'depth.png', depth, cmap='Greys')

    if sim_args.observations.get('normal'):
        normal = observation["observation"]["sensors"]["normal"]["data"]
        normal = normal.reshape((normal.shape[1], normal.shape[0], normal.shape[2]))
        plt.imsave(prename + 'normal.png', normal)

    if sim_args.observations.get('objectId'):
        object_id = observation["observation"]["sensors"]["objectId"]["data"]
        object_id = object_id.reshape((object_id.shape[1], object_id.shape[0], object_id.shape[2]))
        plt.imsave(prename + 'object_id.png', object_id)

    if sim_args.observations.get('roomId'):
        room_id = observation["observation"]["sensors"]["roomId"]["data"]
        plt.imsave(prename + 'room_id.png', room_id)

    if sim_args.observations.get('roomType'):
        room_type = observation["observation"]["sensors"]["roomType"]["data"]
        plt.imsave(prename + 'room_type.png', room_type)

    if sim_args.observations.get('map'):
        nav_map = observation["observation"]["map"]["data"]
        plt.imsave(prename+ 'nav_map.png', nav_map)

    #shortest_path = observation["observation"]["measurements"]["shortest_path_to_goal"]
    #print(shortest_path)


def main():
    parser = argparse.ArgumentParser(description='MINOS gym wrapper')
    parser.add_argument('--save_observations', action='store_true',
                        default=False,
                        help='Save sensor observations at each step to images')
    args = parse_sim_args(parser)
    args.visualize_sensors = True
    run_gym(args)


if __name__ == "__main__":
    main()
