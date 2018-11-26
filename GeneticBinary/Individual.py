
import random as rand

class Individual:

    def __init__(self,index,alphabet,numberofgenes, fitness=-1):
        self.index = index
        self.fitness = fitness
        self.alphabet = alphabet
        self.numberOfGenes = numberofgenes
        self.genes = [rand.choice(alphabet) for i in range(numberofgenes)]
