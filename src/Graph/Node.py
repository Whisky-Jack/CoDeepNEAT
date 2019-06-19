import matplotlib.pyplot as plt
import math


class Node:
    """
    All children lead to the leaf node
    """

    children = []
    parents = []
    value = None

    traversalID = ""  # a string structured as '1,1,3,2,0' where each number represents which child to move to along the path from input to output

    def __init__(self, val=None):
        self.value = val
        self.children = []
        self.parents = []

    def addChild(self, value=None):
        self.addChild(Node(value))

    def addChild(self, childNode):
        """
        :param childNode: Node to be added - can have subtree underneath
        """
        self.children.append(childNode)
        childNode.parents.append(self)

    def getChild(self, childNum):
        return self.children[childNum]

    def getOutputNode(self):
        if len(self.children) == 0:
            return self

        return self.children[0].getOutputNode()

    def getInputNode(self):
        if len(self.parents) == 0:
            return self

        return self.parents[0].getInputNode()

    def getTraversalIDs(self, currentID=""):
        """
        should be called on root node
        calculates all nodes traversal IDs
        """
        # TODO if len(self.partents) != 0 exception bc not root node?
        self.traversalID = currentID
        # print(self,"num children:", len(self.children))
        # print("Me:",self,"child:",self.children[0])
        for childNo in range(len(self.children)):
            newID = currentID + (',' if not currentID == "" else "") + repr(childNo)
            # print(newID)
            self.children[childNo].getTraversalIDs(newID)

    def isInputNode(self):
        return len(self.parents) == 0

    def isOutputNode(self):
        return len(self.children) == 0

    def hasSiblings(self):
        for parent in self.parents:
            if len(parent.children) > 1:
                return True

        return False

    def printTree(self, nodesPrinted=set()):
        if self in nodesPrinted:
            return
        nodesPrinted.add(self)
        self.printNode()

        for child in self.children:
            child.printTree(nodesPrinted)

    def printNode(self, printToConsole=True):
        pass

    def plotTree(self, nodesPlotted=None, rotDegree=0):
        if nodesPlotted is None:
            nodesPlotted = set()

        arrowScaleFactor = 1

        y = len(self.traversalID)
        x = 0

        for i in range(4):
            x += self.traversalID.count(repr(i)) * i

        # x +=y*0.05

        x = x * math.cos(rotDegree) - y * math.sin(rotDegree)
        y = y * math.cos(rotDegree) + x * math.sin(rotDegree)

        if self in nodesPlotted:
            return x, y

        nodesPlotted.add(self)

        plt.plot(x, y, self.getPlotColour(), markersize=10)

        for child in self.children:
            c = child.plotTree(nodesPlotted, rotDegree)
            if (not c == None):
                cx, cy = c
                plt.arrow(x, y, (cx - x) * arrowScaleFactor, (cy - y) * 0.8 * arrowScaleFactor, head_width=0.13,
                          length_includes_head=True)

                # print("plotting from:",(x,y),"to",(cx,cy))
            # print(child.plotTree(nodesPlotted,xs,ys))

        if self.isInputNode():
            plt.show()

        return x, y

    def getPlotColour(self):
        return 'ro'

    def dfs(self, other_id):
        if other_id == self.traversalID:
            return True

        for child in self.children:
            return child.dfs(other_id)

        return False

    def is_linear(self):
        """Determines if you and your child are linear"""
        return len(self.children) == 1 and len(self.children[0].children == 1)

    # TODO test this!
    def is_diamond_right(self, child_index):
        """Checks for a diamond structure with child at the given index and next index"""
        if len(self.children) < 2 or len(self.children) >= child_index:
            return False

        left_id = self.children[child_index].traversalID
        return not self.children[child_index + 1].dfs(left_id)

    def is_tri_right(self, child_index):
        return not (self.is_diamond_right(child_index) and self.is_linear())


def genNodeGraph(nodeType, graphType="diamond", linearCount=3):
    """the basic starting points of both blueprints and modules"""
    inp = nodeType()

    if graphType == "linear":
        curr = inp
        for _ in range(linearCount - 1):
            curr.addChild(nodeType())
            curr = curr.children[0]

    if graphType == "diamond":
        inp.addChild(nodeType())
        inp.addChild(nodeType())
        inp.children[0].addChild(nodeType())
        inp.children[1].addChild(inp.children[0].children[0])

    if graphType == "triangle":
        """feeds input node to a child and straight to output node"""
        inp.addChild(nodeType())
        inp.children[0].addChild(nodeType())
        inp.addChild(inp.children[0].children[0])

    if graphType == "single":
        pass

    return inp
