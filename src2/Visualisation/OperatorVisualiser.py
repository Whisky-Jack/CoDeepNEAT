import random
from typing import List

from Genotype.NEAT.Genome import Genome
from Genotype.NEAT.Operators.Mutations import MutationRecord
from Genotype.NEAT.Operators.Mutations.GenomeMutator import GenomeMutator
from Visualisation.GenomeVisualiser import get_graph_of

"""
    class to visualise the various genome operations:
"""

def visualise_mutation(genomes: List[Genome], mutator: GenomeMutator, mutation_record: MutationRecord, count = 1):
    """
    samples genomes and plots them before and after mutation
    """

    for i in range(count):
        genome:Genome = random.choice(genomes)
        before_graph = get_graph_of(genome, node_names="before")
        mutant_genome = mutator.mutate(genome,mutation_record)
        both_graph = get_graph_of(mutant_genome, node_names="after", append_graph= before_graph)
        both_graph.view()

def visualise_crossover(genomes: List[Genome], count = 1):
    """
    pairs genomes, plots parents along with their children
    """
    pass
