import random

from src.Graph.Node import Node
import torch.nn as nn

from src.Learner.Layers import MergeSum, MergeCat, SequentialMemory
from src.Learner.Net import BlueprintNet


class ModuleNode(Node):
    """
    ModuleNode represents a node in a module
    The whole  Module is represented as the input Module, followed by its children
    Each Module should have one input node and one output node
    All module children are closer to the output node than their parent (one way/ feed forward)

    Modules get parsed into neural networks via traversals
    """

    layerType = None
    regularisation = None

    def __init__(self):
        Node.__init__(self)
        self.deepLayer = None  # an nn layer object such as    nn.Conv2d(3, 6, 5) or nn.Linear(84, 10)
        self.activation = nn.ReLU()  # should we have an activation here?
        self.inFeatures = -1
        self.outFeatures = -1
        self.traversalID = ""

        self.combined = None

    # Why is this not just the constructor?
    def createLayers(self, inFeatures=None, outFeatures=20, debug=False):
        self.outFeatures = outFeatures
        if self.deepLayer is None:
            if inFeatures is None:
                self.inFeatures = self.parents[0].outFeatures  # only aggregator nodes should have more than one parent
            else:
                self.inFeatures = inFeatures

            # Maybe just pass the entire layer and activation in?
            self.deepLayer = nn.Conv2d(self.inFeatures, self.outFeatures, 3, 1)
            if debug:
                self.deepLayer.weight.data.fill_(0.2341)
                self.deepLayer.bias.data.fill_(0.341)

            self.combined = [self.deepLayer, self.activation, nn.MaxPool2d(2, 2)]

            # if random.randint(0, 1) == 0:
            #     self.combined.add_module('batch_norm-' + str(self.traversalID), nn.BatchNorm2d(outFeatures))

            for child in self.children:
                child.createLayers()

    # could be made more efficient as a breadth first instead of depth first because of duplicate paths
    def insertAggregatorNodes(self, state="start"):
        from src.Module.AggregatorNode import AggregatorNode as Aggregator
        """
        *NB  -- must have called getTraversalIDs on root node to use this function
        traverses the module graph from the input up to output, and inserts aggregators
        which are module nodes which take multiple inputs and combine them
        """

        if state == "start":  # method is called from the root node - but must be traversed from the output node
            outputNode = self.getOutputNode()
            outputNode.insertAggregatorNodes("fromTop")
            return

        # print("inserting agg nodes from " , self.traversalID)

        numParents = len(self.parents)

        if (numParents > 1):
            # must insert an aggregator node
            aggregator = Aggregator()  # all parents of self must be rerouted to this aggregatorNode
            # self.getInputNode().registerAggregator(aggregator)

            for parent in self.parents:
                aggregator.addParent(parent)
                parent.children = [aggregator if x == self else x for x in
                                   parent.children]  # replaces self as a child with the new aggregator node

            self.parents = []  # none of the parents are now a parent to this node - instead only the aggregator is
            # self.parents.append(aggregator)
            aggregator.addChild(self)

            for previousParent in aggregator.parents:
                previousParent.insertAggregatorNodes("fromTop")

        elif numParents == 1:
            self.parents[0].insertAggregatorNodes("fromTop")

        if self.isOutputNode():
            self.getInputNode().getTraversalIDs('_')

    def passANNInputUpGraph(self, input, parentID=""):
        """
        called by the forward method of the NN - traverses the module graph
        passes the nn's input all the way up through the graph
        aggregator nodes wait for all inputs before passing outputs
        """

        # if(self.isOutputNode()):
        #     print("final node received input:", input)

        output = self.passInputThroughLayer(input)  # will call aggregation if is aggregator node
        # if(input is None):
        #     print("out of agg(",type(self), "):",output)

        childOut = None
        for child in self.children:
            # if (input is None):
            #     print("passing output to child:",child)
            co = child.passANNInputUpGraph(output, self.traversalID)
            if co is not None:
                childOut = co

        if childOut is not None:
            # print("bubbling up output nodes output:", childOut.size())
            return childOut

        if self.isOutputNode():
            # print("output node returning output:",output)
            return output  # output of final output node must be bubbled back up to the top entry point of the nn

    def passInputThroughLayer(self, input):
        # print("inshape:",input.size())
        # print("outshape:", self.deepLayer(input).size())
        return self.deepLayer(input)

    def getParameters(self, parametersDict):
        if self not in parametersDict:
            myParams = self.deepLayer.parameters()
            parametersDict[self] = myParams

            for child in self.children:
                child.getParameters(parametersDict)

            if self.isInputNode():
                # print("input node params: ", parametersDict)

                params = None
                for param in parametersDict.values():

                    if params is None:
                        params = list(param)
                    else:
                        params += list(param)
                return params

    def printNode(self, printToConsole=True):
        out = " " * (len(self.traversalID)) + self.traversalID
        if printToConsole:
            print(out)
        else:
            return out

    def getDimensionality(self):
        # TODO
        print("need to implement get dimensionality")
        # print(torch.cumprod(self.deepLayer.shape()))
        return 10 * 10 * self.deepLayer.out_channels  # 10*10 because by the time the 28*28 has gone through all the convs - it has been reduced to 10810

    def getOutFeatures(self, deepLayer=None):  # why is this param here? We want to check this node's deepLayer
        if deepLayer is None:
            deepLayer = self.deepLayer

        if type(deepLayer) == nn.Conv2d:
            numFeatures = deepLayer.out_channels
        elif type(deepLayer) == nn.Linear:
            numFeatures = deepLayer.out_features
        else:
            print("cannot extract num features from layer type:", type(deepLayer))
            return None

        return numFeatures

    def to_nn(self):
        """
        Parses a CDN module to a pytorch nn.Module
        Must be called on the output node to fully parse the module

        :return: nn.Module representing this CDN module
        """
        # list of all the parents as nn.Modules
        input_layers = nn.ModuleList([parent.to_nn() for parent in self.parents])

        # comb = (nn.Conv2d(self.inFeatures, self.outFeatures, 3, 1), nn.ReLU(), nn.MaxPool2d(2, 2))

        if self.isInputNode():
            return nn.Sequential(*self.combined)  # self.combined

        # if not input node then a node's input is the merge of its parents outputs
        merged = self.intelligent_merge(input_layers, self.deepLayer)

        # finding the largest integer key in the module map
        key = 0
        for name, _ in merged.named_children():
            if name.isdigit() and int(name) > key:
                key = int(name)

        for i, layer in enumerate(self.combined, 1):
            merged.add_module(str(i + key), layer)

        # return nn.Sequential(self.intelligent_merge(input_layers, self.deepLayer), self.combined)
        return merged

    def intelligent_merge(self, parents, child):
        """Chooses correct type of merge for all parents to output for a single child"""
        # TODO choose merge type depending on parent and child dims
        if len(parents) == 1:
            return parents[0]

        return MergeSum(parents)
