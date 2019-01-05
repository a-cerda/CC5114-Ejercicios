import gym_super_mario_bros
import os
import neat
import cv2
import numpy as np
import pickle

# Making the environment as specified in the docs for gym-super-mario-bros
env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')

# Button mappings, they are all byte values taken from the emulator (NOP corresponds to doing nothing)
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
# We set the number of frames our player is allowed to try and advance
stagnation_frames = 17 * 4 # 17 frames per game second, 5 seconds for mario to do something


# Our evaluation function, it does all the training
def eval_genomes(genome, config):
        # First observation of the player screen, also initializing the environment
        observation = env.reset()
        # Taking the pixel count for X, Y and the Colors (B,G,R)
        inx, iny, inc = env.observation_space.shape  #inx = 240, iny = 256, inc = 3
        # We downsample the player screen
        inx = int(inx / 8)
        iny = int(iny / 8)
        # Creating a network with the genome that our ParallelEvaluator gives us
        network = neat.nn.RecurrentNetwork.create(genome, config)
        current_max_fitness = 0
        current_fitness = 0
        stagnation_counter = 0

        done = False
        # The training loop
        while not done:
            # We downsample the pixels into 30*34 and change the colors into grayscale
            observation = cv2.resize(observation, (inx, iny))
            observation = cv2.cvtColor(observation, cv2.COLOR_BGR2GRAY)
            # Then we flatten the array so our network can process it
            observation = np.reshape(observation, (inx, iny))

            image_array = np.ndarray.flatten(observation)
            # We give the flattened array to our network
            network_output = network.activate(image_array)
            # We calculate the buttons that we pressed, using our button array
            output_binary = 0
            for i, value in enumerate(network_output):
                if value > 0.99:
                    output_binary |= button_array[i]

            # We take all the variables that gym gives us and store them
            observation, reward, done, info = env.step(output_binary)

            current_fitness += reward
            # If we pass the level, then we finish the execution via fitness
            if info['flag_get']:
                current_fitness += 13000
            # We reset the stagnation counter if we have gained fitness
            if current_fitness > current_max_fitness:
                current_max_fitness = current_fitness
                stagnation_counter = 0
            # if not, we increment it
            else:
                stagnation_counter += 1
            # These are the cases in which the training terminates
            if done or stagnation_counter >= stagnation_frames:
                done = True
                state = env.reset()
                print(current_fitness)

            genome.fitness = current_fitness
        # Our function has to return the fitness of our genome in order for the ParallelEvaluator to continue
        return current_max_fitness

def run(config_file):
    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Create the population, which is the top-level object for a NEAT run.
    # We restore the population that was the fittest
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(15))

    parallel_evaluator = neat.ParallelEvaluator(8, eval_genomes)

    # Run for up to 200 generations.
    winner = p.run(parallel_evaluator.evaluate)

    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))

    # Show output of the most fit genome against training data.
    print('\nOutput:')
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
    # We save the winner network in a pickle container
    with open('winner.pkl', 'wb') as output:
        pickle.dump(winner, output, 1)




if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-mario')
    run(config_path)