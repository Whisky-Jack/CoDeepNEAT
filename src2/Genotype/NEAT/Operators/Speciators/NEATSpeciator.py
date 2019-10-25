from typing import List, TYPE_CHECKING

from Genotype.NEAT.Operators.Speciators.Speciator import Speciator
from src2.Configuration import config

if TYPE_CHECKING:
    from Genotype.NEAT.Species import Species
    from Genotype.NEAT.Genome import Genome


class NEATSpeciator(Speciator):
    def speciate(self, species: List[Species]) -> None:
        self.adjust_speciation_threshold(len(species))
        individuals: List[Genome] = [member for spc in species for member in spc.members]

        for individual in individuals:
            found = False
            for spc in species:
                if individual.distance_to(spc.representative) <= self.threshold:
                    spc.add(individual)
                    found = True
                    break

            if not found:
                species.append(Species(individual))

    def adjust_speciation_threshold(self, n_species: int):
        if self.target_num_species == 1:
            return

        if n_species < self.target_num_species:
            self.current_threshold_dir = -1  # decrease thresh
        elif n_species > self.target_num_species:
            self.current_threshold_dir = 1  # increase thresh
        else:
            self.current_threshold_dir = 0
            return

        self.threshold += max(config.species_distance_thresh_mod_min,
                              config.species_distance_thresh_mod_base * self.current_threshold_dir)