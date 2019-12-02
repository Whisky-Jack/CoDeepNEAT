import threading
from typing import List, Tuple

from runs import RunsManager
from src2.Configuration import config
from src2.Phenotype.NeuralNetwork.Layers.Layer import Layer
from src2.Genotype.NEAT.Connection import Connection
from src2.Genotype.NEAT.Genome import Genome
from src2.Genotype.NEAT.Node import Node
from src2.Visualisation.GenomeVisualiser import get_graph_of


class ModuleGenome(Genome):
    def __int__(self, nodes: List[Node], connections: List[Connection]):
        super().__init__(nodes, connections)

    def to_phenotype(self, **kwargs) -> Tuple[Layer, Layer]:
        return super().to_phenotype(**kwargs)

    def visualize(self):
        get_graph_of(self).render(directory=RunsManager.get_graphs_folder_path(), view=config.view_graph_plots)

    def __getstate__(self):
        d = dict(self.__dict__)
        if 'lock' in d:
            del d['lock']
        return d

    def __setstate__(self, state):
        self.__dict__ = state
        self.__dict__['lock'] = threading.RLock()
