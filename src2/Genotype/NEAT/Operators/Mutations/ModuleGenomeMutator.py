from src2.Genotype.NEAT.Genome import Genome
from src2.Genotype.NEAT.Operators.Mutations import MutationRecord
from src2.Genotype.NEAT.Operators.Mutations.GenomeMutator import GenomeMutator


class ModuleGenomeMutator(GenomeMutator):

    def mutate(self, genome: Genome, mutation_record: MutationRecord):
        """
            performs base NEAT genome mutations, as well as node and genome property mutations
            as well as all mutations specific to module genomes
        """
        self.mutate_base_genome(genome, mutation_record, 0, 0)  # todo get these mutation chances from config