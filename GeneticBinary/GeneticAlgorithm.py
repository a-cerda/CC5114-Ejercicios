import random as rand


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
        self.populationSize = populationSize
        self.fitnessPerGeneration = []
        self.generatePopulation()




    def generatePopulation(self):
        self.population = self.generatorFunction(self.numberOfGenes,self.populationSize)

    def calculateFitness(self):
        for i, individual in enumerate(self.population):
            self.regularFitness[i] = self.fitnessFunction(individual[0])
            individual[1] = self.regularFitness[i]


    def normalizeAndOrderFitness(self):
        sumOfAllFitness = float(sum(self.regularFitness))
        for i in range(self.populationSize):
            try:
                self.accfitness[i] = (self.regularFitness[i] / sumOfAllFitness)
            except ZeroDivisionError:
                self.accfitness[i] = 0
        self.accfitness.sort(reverse=True)
        self.population.sort(key=lambda x : x[1],reverse=True)


    def accumulateFitness(self):
        for i in range(1,len(self.accfitness)):
            self.accfitness[i] += self.accfitness[i-1]

    def selectFittest(self):
        n = rand.random()
        if n < self.accfitness[0]:
            return 0
        for i in range(self.populationSize):
            if n < self.accfitness[i]:
                return i
        return len(self.accfitness)-1

    def reproduce(self):
        newPopulation = []
        for i in range(self.populationSize):
            firstParent = self.population[self.selectFittest()][0]
            secondParent = self.population[self.selectFittest()][0]
            crossoverPoint = rand.randint(0,self.numberOfGenes)
            child = []
            for j in range(self.numberOfGenes):
                if j < crossoverPoint:
                    child.append(firstParent[j])
                else:
                    child.append(secondParent[j])
                if(rand.random() < self.mutationProbability):
                    child[j] = rand.choice(self.alphabet)
            newPopulation.append([child,0])

        self.population = newPopulation

    def tournamentSelection(self):
        pass

    def runWithAccFitness(self):

        for i in range(self.maxNumberOfIterations):
            self.calculateFitness()
            self.normalizeAndOrderFitness()
            self.accumulateFitness()
            if self.fitnessFunction(self.population[0][0]) == self.numberOfGenes:
                print("The solution has been found on generation "+str(i)+" and it is: "
                      +''.join(self.population[0][0]))
                self.fitnessPerGeneration.append(self.fitnessFunction(self.population[0][0]))
                self.maximumIterationReached = i
                return self.population[0][0]
            self.fitnessPerGeneration.append(self.fitnessFunction(self.population[0][0]))
            print("The current generation is: "+str(i)+" and it's max fitness is : "+str(self.fitnessPerGeneration[i]))

            self.reproduce()

        print("The solution wasn't found on "+str(self.maxNumberOfIterations)+" iterations")

    def runWithTournamentSelection(self):
        pass

