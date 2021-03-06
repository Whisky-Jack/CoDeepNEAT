from __future__ import annotations

from typing import TYPE_CHECKING

import random
from enum import Enum

from configuration import config
from src.genotype.mutagen.option import Option
from src.genotype.neat.gene import Gene

if TYPE_CHECKING:
    pass


class NodeType(Enum):
    INPUT = 0
    HIDDEN = 1
    OUTPUT = 2


class Node(Gene):
    """General neat node"""

    def __init__(self, id, type: NodeType = NodeType.HIDDEN):
        super().__init__(id)
        self.node_type: NodeType = type
        # TODO
        self.lossy_aggregation = Option('lossy', False, True,
                                        current_value=random.choices([False, True], weights=[1-config.lossy_chance,
                                                                                             config.lossy_chance])[0],
                                        mutation_chance=0.3 if config.mutate_lossy_values else 0)
        self.try_conv_aggregation = Option('conv_aggregation', False, True, current_value=random.choice([False, True]))
        mult_chance = config.element_wise_multiplication_chance
        mult_weights = [1-mult_chance, mult_chance]
        self.element_wise_multiplication_aggregation = \
            Option('element_wise_multiplication_aggregation', False, True, current_value=
            random.choices([False, True], weights=mult_weights)[0],
                   mutation_chance= 0.2 if mult_chance > 0 else 0,
                   probability_weighting=mult_weights)

    def is_output_node(self):
        return self.node_type == NodeType.OUTPUT

    def is_input_node(self):
        return self.node_type == NodeType.INPUT

    def get_all_mutagens(self):
        return [self.lossy_aggregation, self.try_conv_aggregation]

    def convert_node(self, **kwargs):
        raise NotImplemented()
