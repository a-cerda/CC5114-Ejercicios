from GeneticBinary import GeneticAlgorithm
import matplotlib.pyplot as plt
import random as rand

graph = {'A': ['B', 'C'],
         'B': ['A', 'C', 'D'],
         'C': ['A', 'B', 'D', 'F'],
         'D': ['B', 'C'],
         'E': ['F'],
         'F': ['C', 'E']}

alphabet = (list(graph.keys()))


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

number_of_genes = len(graph.keys())

hamiltonian_genetic_algorithm = GeneticAlgorithm.GeneticAlgorithm(100, 0.1, number_of_genes, fitness_function,
                                                                  generator_function, alphabet)
hamiltonian_genetic_algorithm.runWithAccFitness()
