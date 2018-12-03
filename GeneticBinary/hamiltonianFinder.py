import GeneticAlgorithm
import matplotlib.pyplot as plt
import random as rand
import string
import time

# first graph tried
graph = {'A': ['B', 'C'],
         'B': ['A', 'C', 'D'],
         'C': ['A', 'B', 'D', 'F'],
         'D': ['B', 'C'],
         'E': ['F'],
         'F': ['C', 'E']}



adjacency_matrix_1 =   [[0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                        [1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0],
                        [0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1],
                        [0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0],
                        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1],
                        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                        [0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
                        [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0],
                        [0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0],
                        [0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                        [0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1],
                        [0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0]]

adjacency_matrix_2 =   [[0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                        [0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1],
                        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0]]


# An auxiliary function to translate an adjacency matrix to a graph represantation
# with dicts, such as the one first given here
def adjacency_matrix_to_graph(adjacency_matrix):
    letters_list = list(string.ascii_uppercase)
    graph_letters = []
    graph_connections = [[] for i in range(len(adjacency_matrix))]
    for i, row in enumerate(adjacency_matrix):
        graph_letters.append(letters_list[i])
        for j, column in enumerate(row):
            if column == 1:
                graph_connections[i].append(letters_list[j])

    new_graph = dict(zip(graph_letters, graph_connections))
    return new_graph


# we change the graph to be one of the adjacency matrixes
graph = adjacency_matrix_to_graph(adjacency_matrix_1)
print(graph.keys())


# make a gene be the next node on the path, starting with a node gotten randomly from the graph dict.
def path_generator(numberOfGenes, populationSize):
    population = []
    for i in range(populationSize):
        chosen_node = rand.choice(alphabet)
        individual = [chosen_node]
        for j in range(1,numberOfGenes):
            individual.append(rand.choice(graph[chosen_node]))
            chosen_node = individual[j]
        population.append(individual)

    return population


def path_fitness(individual):
    # Whenever we visit a new node, the fitness goes up
    fitness = 0
    visited_nodes = []
    last_visited_node = None
    for i, gene in enumerate(individual):
        if i == 0:
            last_visited_node = gene
            visited_nodes.append(gene)
            fitness += 1
            continue
        if gene not in visited_nodes and gene in graph[last_visited_node]:
            visited_nodes.append(gene)
            fitness += 1
            last_visited_node = gene
        else:
            fitness -= 1
    if fitness < 0:
        fitness = 0
    return fitness


fitness_function = path_fitness
generator_function = path_generator
alphabet = (list(graph.keys()))
number_of_genes = len(graph.keys())


# Comment from here if you want to set a single algorithm pass
# initial_mutation_rate = 0.01
# mutation_rates = [initial_mutation_rate + i*0.005 for i in range(20)]
# hamiltonian_algorithms = []
# for mutation_rate in mutation_rates:
#     hamiltonian_algorithms.append(
#         GeneticAlgorithm.GeneticAlgorithm(200, mutation_rate, number_of_genes, fitness_function,
#                                           generator_function, alphabet,
#                                           maxNumberOfIterations=30000, seed = 23)
#     )
# hamiltonian_genetic_algorithm = GeneticAlgorithm.GeneticAlgorithm(200, 0.0675, number_of_genes, fitness_function,
#                                                                   generator_function, alphabet,
#                                                                   maxNumberOfIterations=30000)
#
# execution_times = []
#
# # Plotting
# for algorithm in hamiltonian_algorithms:
#     start = time.time()
#
#     algorithm.runWithAccFitness()
#     end = time.time()
#     execution_times.append(end - start)
#     generations = [i for i in range(algorithm.maximumIterationReached)]
#     fitness = [fitness for fitness in algorithm.fitnessPerGeneration]
#     plt.plot(generations, fitness, '-', label='Fitness')
#
#     plt.legend()
#
#     plt.xlabel('Generation')
#     plt.xscale('log')
#
#     plt.ylabel("Fitness per generation")
#     plt.title("200 Individuals and "+str(algorithm.mutationProbability)+" mutation rate")
#     plt.savefig("mutation_rate" + str(algorithm.mutationProbability) + ".png")
#     plt.show()
#
#
#
# print(execution_times)

#Uncomment here and comment above if you want a single algorithm pass
hamiltonian_genetic_algorithm = GeneticAlgorithm.GeneticAlgorithm(800, 0.06, number_of_genes, fitness_function,
                                                                  generator_function, alphabet,
                                                                  maxNumberOfIterations=30000)

#Plotting
start = time.time()

hamiltonian_genetic_algorithm.runWithAccFitness()
end = time.time()

generations = [i for i in range(hamiltonian_genetic_algorithm.maximumIterationReached)]
fitness = [fitness for fitness in hamiltonian_genetic_algorithm.fitnessPerGeneration]
plt.plot(generations, fitness, '-', label='Fitness')

plt.legend()

plt.xlabel('Generation')


plt.ylabel("Fitness per generation")
plt.title("800 Individuals and "+str(hamiltonian_genetic_algorithm.mutationProbability)+" mutation rate")

plt.show()

running_time = (end - start)
print("the time taken to finish the algorithm was: "+str(running_time))