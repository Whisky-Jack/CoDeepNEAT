from src.Module.Species import Species
from src.Blueprint.Blueprint import BlueprintNode
from src.Graph import Node
from src.NeuralNetwork.ANN import ModuleNet
import torch
import math

import torch.nn as nn
from src.Learner import Evaluator, Net, Layers
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
            blueprint = Node.genNodeGraph(BlueprintNode, "single")
            self.blueprintCollection.add(blueprint)

        species = Species()
        species.initialiseModules(self.numModules)
        self.speciesCollection[species.speciesNumber] = species
        self.speciesNumbers.append(species.speciesNumber)

    def generateFromPreviousGeneration(self, previousGen):
        pass

    def print_children(self, net):
        for n in net.children():
            self.print_children(n)

        print('an item:', net)

    def evaluate(self):
        print("evaluating blueprints")

        for blueprint in self.blueprintCollection:
            print("parsing blueprint to module")

            x = torch.randn(1, 8)

            moduleGraph = blueprint.parseToModule(self)
            moduleGraph.createLayers(inFeatures=1)
            net = Net.BlueprintNet(nn.Sequential(moduleGraph.getOutputNode().to_nn(),
                                                 Layers.Reshape(64, -1),
                                                 nn.Linear(500, 500),
                                                 nn.ReLU(),
                                                 nn.Linear(500, 10),
                                                 nn.ReLU(),
                                                 nn.LogSoftmax(dim=1)
                                                 )).cuda()

            # moduleGraph.plotTree()
            print(net)
            # self.print_children(net)

            Evaluator.evaluate(net, 10)

            # moduleGraph1 = blueprint.parseToModule(self)
            # moduleGraph1.createLayers(inChannels=1)
            # moduleGraph1.insertAggregatorNodes()
            # net1 = ModuleNet(moduleGraph1)  # .to(torch.device("cuda:0"))
            # print(x)
            # print('shanes net:', net1(x))
            # Evaluator.evaluate(net, 15, dataset='mnist', path='../../data', device=torch.device("cpu"))
