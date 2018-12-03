import string
from GeneticBinary import GeneticAlgorithm
import matplotlib.pyplot as plt
import random as rand
import time

#expected_string = "porquelatierraesmaspequenaqueelsolylalunaesmaspequenaquelatierra"
expected_string = "supercalifragilisticoespialidoso"

def charFitnessFunc(individual):
    fitness = 0
    for i, gene in enumerate(individual):
        if gene == expected_string[i]:
            fitness += 1
    return fitness


def strGeneratorFunc(numberOfGenes, populationSize):
    alphabet = list(string.ascii_lowercase)
    population = []
    for i in range(populationSize):
        population.append([rand.choice(alphabet) for i in range(numberOfGenes)])

    return population


fitnessFunction = charFitnessFunc

generatorFunc = strGeneratorFunc
count = 0
for char in expected_string:
    count += 1

alggenetico = GeneticAlgorithm.GeneticAlgorithm(800,0.001,count,fitnessFunction,
                                                generatorFunc,alphabet = list(string.ascii_lowercase),maxNumberOfIterations=2000)
start = time.time()
alggenetico.runWithAccFitness()
end = time.time()
running_time = (end - start)
print("the time taken to finish the algorithm was: "+str(running_time))
# #Plotting


generations = [i for i in range(alggenetico.maximumIterationReached)]
fitness = [fitness for fitness in alggenetico.fitnessPerGeneration]
plt.plot(generations,fitness,label='Fitness')

plt.legend()

plt.xlabel('Generation')

plt.title("Fitness per generation")
plt.show()
