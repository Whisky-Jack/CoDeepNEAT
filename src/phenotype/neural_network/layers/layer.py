from __future__ import annotations

import copy

import math
from functools import reduce
from typing import Optional, List, Tuple, TYPE_CHECKING

from torch import nn, zeros

from src.phenotype.neural_network.layers.base_layer import BaseLayer
from src.phenotype.neural_network.layers.custom_layer_types.depthwise_separable_conv import DepthwiseSeparableConv
from src.phenotype.neural_network.layers.custom_layer_types.pad_up import PadUp
from src.phenotype.neural_network.layers.custom_layer_types.reshape import Reshape

if TYPE_CHECKING:
    from src.genotype.cdn.nodes.module_node import ModuleNode


class Layer(BaseLayer):
    def __init__(self, module: ModuleNode, name, feature_multiplier: float = 1):
        super().__init__(name)
        self.module_node: ModuleNode = copy.deepcopy(module)

        self.out_features = round(module.layer_type.get_subvalue('out_features') * feature_multiplier)
        if self.module_node.is_conv():
            self.out_features = max(self.out_features, 1)
        if self.module_node.is_linear():
            self.out_features = max(self.out_features, 10)

        self.sequential: Optional[nn.Sequential] = None
        self.activation: Optional[nn.Module] = self.module_node.activation.value
        self.has_deep_layer : bool = True # false if the layer is an identity or regulariser only

    def forward(self, x):
        try:
            if self.has_deep_layer:  # dont activate unless passing through deep layer
                return self.activation(self.sequential(x))
            else:
                return self.sequential(x)
        except Exception as e:
            print("error passing shape", x.size(), "through ", self.sequential, '\nModule layer type',
                  self.module_node.layer_type.value)
            raise Exception(e)

    def _create_regularisers(self, in_shape: List[int]) -> Tuple[nn.Module]:
        """Creates and returns regularisers given mutagens in self.module_node.layer_type"""
        regularisation: Optional[nn.Module] = None
        reduction: Optional[nn.Module] = None
        dropout: Optional[nn.Module] = None

        neat_regularisation = self.module_node.layer_type.get_submutagen('regularisation')
        neat_dropout = self.module_node.layer_type.get_submutagen('dropout')
        neat_reduction = None
        if self.module_node.is_conv():
            neat_reduction = self.module_node.layer_type.get_submutagen('reduction')

        if neat_regularisation is not None and neat_regularisation.value is not None:
            # Can use either batchnorm 1D or 2D must decide based on input shape
            if neat_regularisation.value == 'batchnorm':
                if len(in_shape) == 4:
                    regularisation = nn.BatchNorm2d(in_shape[1])
                else:
                    regularisation = nn.BatchNorm1d(in_shape[1])
            else:  # input size is known at the genome level
                regularisation = neat_regularisation()(self.out_features)

        if neat_reduction is not None and neat_reduction.value is not None:
            pool_size = neat_reduction.get_subvalue('pool_size')
            if neat_reduction.value == nn.MaxPool2d or neat_reduction.value == nn.AvgPool2d:
                reduction = neat_reduction.value(pool_size, pool_size, padding=pool_size // 2)  # TODO should be stride
            elif neat_reduction.value == nn.MaxPool1d or neat_reduction.value == nn.AvgPool1d:
                reduction = neat_reduction.value(pool_size, padding=pool_size // 2)

        if neat_dropout is not None and neat_dropout.value is not None:
            dropout = neat_dropout.value(neat_dropout.get_subvalue('dropout_factor'))

        return tuple(r for r in [regularisation, reduction, dropout] if r is not None)

    def create_layer(self, in_shape: List[int]) -> List[int]:
        """
        Creates a layer of type nn.Linear or nn.Conv2d according to its module_node and gives it the correct shape.
        Populates the self.sequential attribute with created layers and values returned from self.create_regularisers.
        """
        if 0 in in_shape:
            raise Exception("Parent shape contains has a dim of size 0: " + repr(in_shape))

        if len(in_shape) == 4:  # parent node is a conv
            batch, channels, h, w = in_shape
        elif len(in_shape) == 2:  # parent node  is a linear
            batch, channels = in_shape
        else:
            raise Exception('Invalid input with shape: ' + str(in_shape))

        self.has_deep_layer = True

        reshape_layer: Optional[Reshape] = None
        deep_layer: Optional[nn.Module] = None
        img_flat_size = int(reduce(lambda x, y: x * y, in_shape) / batch)

        # Calculating out feature size, creating deep layer and reshaping if necessary
        if self.module_node.is_conv():  # conv layer
            if len(in_shape) == 2:  # need a reshape if parent layer is linear because conv input needs 4 dims
                h = w = math.ceil(math.sqrt(img_flat_size / channels))
                if h * w != img_flat_size / channels:
                    raise Exception("lossy reshape of linear output (" + repr(in_shape) + ") to conv input (" +
                                    str(batch) + ", " + channels + ", " + str(h) * 2 + ")")
                reshape_layer = Reshape(batch, channels, h, w)

            # TODO make kernel size and stride a tuple
            # gathering conv params from module
            window_size = self.module_node.layer_type.get_subvalue('conv_window_size')
            stride = self.module_node.layer_type.get_subvalue('conv_stride')
            padding = math.ceil((window_size - h) / 2)  # just-in-time padding
            padding = padding if padding >= 0 else 0
            if self.module_node.layer_type.get_subvalue("pad_output"):  # Preemptive padding
                padding = max(padding, (window_size - 1) // 2)

            # creating conv layer
            deep_layer = nn.Conv2d(channels, self.out_features, window_size, stride, padding)
        elif self.module_node.is_linear():  # Linear layer
            if len(in_shape) != 2 or channels != img_flat_size:  # linear must be reshaped
                reshape_layer = Reshape(batch, img_flat_size)

            # creating linear layer
            deep_layer = nn.Linear(img_flat_size, self.out_features)
        elif self.module_node.layer_type.value == DepthwiseSeparableConv:
            if len(in_shape) == 2:  # need a reshape if parent layer is linear because conv input needs 4 dims
                h = w = math.ceil(math.sqrt(img_flat_size / channels))
                if h * w != img_flat_size / channels:
                    raise Exception("lossy reshape of linear output (" + repr(in_shape) + ") to conv input (" +
                                    str(batch) + ", " + channels + ", " + str(h) * 2 + ")")
                reshape_layer = Reshape(batch, channels, h, w)

            window_size = self.module_node.layer_type.get_subvalue('conv_window_size')
            kernels_per_layer = self.module_node.layer_type.get_subvalue('conv_window_size')

            deep_layer = DepthwiseSeparableConv(channels, self.out_features, kernels_per_layer, window_size)
        elif self.module_node.layer_type.value is None:  # No deep layer
            deep_layer = None
            self.has_deep_layer = False
        else:
            raise Exception("unrecognised module type: " + repr(self.module_node.layer_type.value))

        # packing reshape, deep layer and regularisers into a sequential
        modules = [module for module in [reshape_layer, deep_layer, *self._create_regularisers(in_shape)] if
                   module is not None]
        # print('layer order:', modules)

        if not modules:
            modules = [nn.Identity()]
        self.sequential = nn.Sequential(*modules)

        self.out_shape = list(self.forward(zeros(in_shape)).size())
        return self.out_shape

    def get_layer_info(self) -> str:
        """for dnn visualization"""
        features = list(self.sequential.children()) + ( [self.activation, "out features: " + repr(self.out_features)] if self.has_deep_layer else [])
        return '\n'.join(map(lambda x: repr(x), features))
