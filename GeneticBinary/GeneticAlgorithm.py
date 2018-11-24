import random as rand
import GeneticBinary.Individual as Individual


class GeneticAlgorithm:

    def __init__(self,populationSize, mutationProbability, numberOfGenes,
                 fitnessFunction, generatorFunction,alphabet,maxNumberOfIterations = 100):

        self.maximumIterationReached = 0
        self.maxNumberOfIterations = maxNumberOfIterations
        self.alphabet = alphabet
        self.fitnessFunction = fitnessFunction
        self.generatorFunction = generatorFunction
        self.mutationProbability = mutationProbability
        self.regularFitness = [0 for i in range(populationSize)]
        self.accfitness = [.0 for i in range(populationSize)]
        self.numberOfGenes = numberOfGenes
        self.population = []
        self.generatePopulation()
        self.fitnessPerGeneration = []


    def generatePopulation(self):
        self.population = self.generatorFunction(self.numberOfGenes,len(self.regularFitness))

    def calculateFitness(self):
        for i, individual in enumerate(self.population):
            self.regularFitness[i] = self.fitnessFunction(individual)



    def normalizeAndOrderFitness(self):
        sumOfAllFitness = float(sum(self.regularFitness))
        for i, individual in enumerate(self.population):
            try:
                self.accfitness[i] = (self.regularFitness[i] / sumOfAllFitness)
            except ZeroDivisionError:
                self.accfitness[i] = 0
            individual.changeFitness(self.accfitness[i])
        self.accfitness.sort(reverse=True)
        self.population.sort(key=lambda x : x.fitness,reverse=True)


    def accumulateFitness(self):
        for i in range(1,len(self.accfitness)):
            self.accfitness[i] += self.accfitness[i-1]

    def selectFittest(self):
        n = rand.random()
        if n < self.accfitness[0]:
            return 0
        for i in range(len(self.accfitness)):
            if n < self.accfitness[i]:
                return i
        return len(self.accfitness)-1

    def reproduce(self):
        newPopulation = []
        for i in range(len(self.population)):
            firstParent = self.population[self.selectFittest()]
            secondParent = self.population[self.selectFittest()]

            crossoverPoint = rand.randint(0,len(self.population))
            child = Individual.Individual(i,self.alphabet,self.numberOfGenes)
            childGenes = []
            for j in range(self.numberOfGenes):
                if j < crossoverPoint:
                    childGenes.append(firstParent.genes[j])
                else:
                    childGenes.append(secondParent.genes[j])
                if(rand.random() < self.mutationProbability):
                    childGenes[j] = rand.choice(self.alphabet)
            child.setGenes(childGenes)
            newPopulation.append(child)

        self.population = newPopulation

    def tournamentSelection(self):
        pass

    def runWithAccFitness(self):
        result = []
        for i in range(self.maxNumberOfIterations):
            self.calculateFitness()
            self.normalizeAndOrderFitness()
            self.accumulateFitness()
            if self.fitnessFunction(self.population[0]) == self.numberOfGenes:
                print("The solution has been found on generation "+str(i)+" and it is: "
                      )
                print(self.population[0].genes)
                self.fitnessPerGeneration.append(self.fitnessFunction(self.population[0]))
                self.maximumIterationReached = i
                result = self.population[0].genes
                return result
            self.fitnessPerGeneration.append(self.fitnessFunction(self.population[0]))
            print(self.fitnessPerGeneration[i])

            self.reproduce()

        print("The solution wasn't found on "+str(self.maxNumberOfIterations)+" iterations")

    def runWithTournamentSelection(self):
        pass

