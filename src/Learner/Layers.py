"""Custom layers that need to be added to an instance of Net"""

from torch import nn, cat
import torch


class HiddenMemory(nn.Module):
    def __init__(self):
        super(HiddenMemory, self).__init__()
        self.memory = None

    def forward(self, input):
        if self.memory is not None:
            return self.memory


class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, input):
        return input.view(self.shape)


# This is a base class should never be called direct because it doesn't have a forward method
class Merge(HiddenMemory):
    def __init__(self, children):
        super(Merge, self).__init__()
        if len(children) < 2:
            raise Exception("Cannot merge less than two children")
        self.childs = children

    def forward(self, input):
        raise Exception('Use a specific type of merge, this is the base class')


class MergeSum(Merge):
    def __init__(self, children):
        super(MergeSum, self).__init__(children)

    def forward(self, input):
        if self.memory is not None:
            print('reusing memory:', self.memory)
            return self.memory

        res = [y(input) for y in self.childs]
        joined = torch.sum(torch.stack(res), dim=0)  # TODO how to choose the dim!?

        self.memory = joined
        return joined


class MergeCat(Merge):
    def __init__(self, children):
        super(MergeCat, self).__init__(children)

    def forward(self, input):
        if self.memory is not None:
            print('reusing memory:', self.memory)
            return self.memory

        res = [y(input) for y in self.childs]
        joined = cat(res, dim=0)  # TODO how to choose the dim!?

        self.memory = joined
        return joined


class SequentialMemory(HiddenMemory):
    def __init__(self, *layers):
        super(SequentialMemory, self).__init__()
        self.model = nn.Sequential(*layers)

    def forward(self, input):
        if self.memory is not None:
            print('reusing memory:', self.memory)
            return self.memory

        self.memory = self.model(input)
        return self.memory
