from src.Module.Species import Species
from src.Blueprint.Blueprint import BlueprintNode
from src.Graph import Node
from src.NeuralNetwork.ANN import ModuleNet
import torch
import math

import torch.nn as nn
from src.Learner import Evaluator
import torch.tensor


class Generation:
    numBlueprints = 1
    numModules = 1

    speciesCollection = {}  # hashmap from species number to species
    speciesNumbers = []
    blueprintCollection = set()

    def __init__(self, firstGen=False, previousGeneration=None):
        self.speciesNumbers = []
        self.speciesCollection = {}
        if (firstGen):
            self.initialisePopulation()
        else:
            self.generateFromPreviousGeneration(previousGeneration)

    def initialisePopulation(self):
        print("initialising random population")

        for b in range(self.numBlueprints):
            blueprint = Node.genNodeGraph(BlueprintNode, "linear")
            self.blueprintCollection.add(blueprint)

        species = Species()
        species.initialiseModules(self.numModules)
        self.speciesCollection[species.speciesNumber] = species
        self.speciesNumbers.append(species.speciesNumber)

    def generateFromPreviousGeneration(self, previousGen):
        pass

    def evaluate(self):
        print("evaluating blueprints")

        for blueprint in self.blueprintCollection:
            print("parsing blueprint to module")

            x = torch.randn(1, 5)

            moduleGraph = blueprint.parseToModule(self)
            moduleGraph.createLayers(inChannels=1)
            net = moduleGraph.getOutputNode().to_nn()

            print(net)

            print(x)
            print('my net:', net(x))

            moduleGraph1 = blueprint.parseToModule(self)
            moduleGraph1.createLayers(inChannels=1)
            moduleGraph1.insertAggregatorNodes()
            net1 = ModuleNet(moduleGraph1)  # .to(torch.device("cuda:0"))
            print(x)
            print('shanes net:', net1(x))
            # Evaluator.evaluate(net, 15, dataset='mnist', path='../../data', device=torch.device("cpu"))
