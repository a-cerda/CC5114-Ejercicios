import gym_super_mario_bros
import os
import neat
import cv2
import numpy as np
import pickle
import time

env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')

_button_map = {
        'right':  0b10000000,
        'left':   0b01000000,
        'down':   0b00100000,
        'up':     0b00010000,
        'B':      0b00000010,
        'A':      0b00000001,
        'NOP':    0b00000000,
    }
button_array = [value for value in _button_map.values()]

image_array = []

# Our evaluation function, it does all the training
def playback_winner():

    observation = env.reset()


    inx, iny, inc = env.observation_space.shape  #inx = 240, iny = 256, inc = 3

    inx = int(inx / 8)
    iny = int(iny / 8)

    with open('winner_first_complete_training.pkl', 'rb') as winner:
        network = pickle.load(winner)
        local_dir = os.path.dirname(__file__)
        config_path = os.path.join(local_dir, 'config-mario')
        config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                             neat.DefaultSpeciesSet, neat.DefaultStagnation,
                             config_path)
        network = neat.nn.RecurrentNetwork.create(network, config=config)
    current_max_fitness = 0
    current_fitness = 0
    stagnation_counter = 0

    done = False

    while not done:

        env.render()

        observation = cv2.resize(observation, (inx, iny))
        observation = cv2.cvtColor(observation, cv2.COLOR_BGR2GRAY)
        observation = np.reshape(observation, (inx, iny))

        image_array = np.ndarray.flatten(observation)

        network_output = network.activate(image_array)
        output_binary = 0
        for i, value in enumerate(network_output):
            if value > 0.99:
                output_binary |= button_array[i]


        observation, reward, done, info = env.step(output_binary)

        current_fitness += reward

        if info['flag_get']:
            current_fitness += 13000

        if current_fitness > current_max_fitness:
            current_max_fitness = current_fitness
            stagnation_counter = 0
        else:
            stagnation_counter += 1

        if done:
            state = env.reset()
        time.sleep(1/30)

playback_winner()