import string
from GeneticBinary import GeneticAlgorithm,Individual
import matplotlib.pyplot as plt

def charFitnessFunc(individual):
    expectedString = "supercalifragilisticoespialidoso"
    i = 0
    fitness = 0
    for gene in individual.genes:
        if gene == expectedString[i]:
            fitness += 1
        i += 1
    individual.changeFitness(fitness)
    return fitness


def strGeneratorFunc(numberOfGenes, populationSize):
    alphabet = list(string.ascii_lowercase)
    population = []
    for i in range(populationSize):
        population.append(Individual.Individual(i,alphabet,numberOfGenes))

    return population


fitnessFunction = charFitnessFunc

generatorFunc = strGeneratorFunc
count = 0
for char in "supercalifragilisticoespialidoso":
    count += 1

alggenetico = GeneticAlgorithm.GeneticAlgorithm(1100,0.01,count,fitnessFunction,
                                                generatorFunc,alphabet = list(string.ascii_lowercase),maxNumberOfIterations=1000)
alggenetico.run()

# #Plotting


generations = [i for i in range(alggenetico.maximumIterationReached+1)]
fitness = [fitness for fitness in alggenetico.fitnessPerGeneration]
plt.plot(generations,fitness,label='Fitness')

plt.legend()

plt.xlabel('Generation')

plt.title("Fitness per generation")
plt.show()
