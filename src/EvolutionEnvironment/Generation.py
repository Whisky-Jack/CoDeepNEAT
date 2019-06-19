import torch

from src.Module.Species import Species
from src.Blueprint.Blueprint import BlueprintNode
from src.Graph import Node
from src.Learner import Evaluator, Net, Layers

import torch.nn as nn


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
            blueprint = Node.genNodeGraph(BlueprintNode, "linear", linearCount=3)
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

            batch_size = 64
            lin_size = int(3276800 / batch_size)

            moduleGraph = blueprint.parseToModule(self)
            moduleGraph.createLayers(inFeatures=1)
            net = Net.BlueprintNet(nn.Sequential(moduleGraph.getOutputNode().to_nn(),
                                                 Layers.Reshape(-1, lin_size),
                                                 nn.Linear(lin_size, 500),
                                                 nn.ReLU(),
                                                 nn.Linear(500, 10),
                                                 nn.LogSoftmax(dim=1)
                                                 )).cuda()

            # moduleGraph.plotTree()
            print('Generated network:\n', net)
            torch.manual_seed(1)

            Evaluator.evaluate(net, 10, batch_size=batch_size)
