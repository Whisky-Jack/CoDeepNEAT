from __future__ import annotations

import argparse
import os
import sys
from typing import TYPE_CHECKING

from runs import RunsManager

# For importing project files
dir_path = os.path.dirname(os.path.realpath(__file__))
dir_path_1 = os.path.split(os.path.split(dir_path)[0])[0]
dir_path_2 = os.path.split(dir_path)[0]
sys.path.append(dir_path_1)
sys.path.append(dir_path_2)
print(os.path.join(dir_path_1, 'test'))
sys.path.append(os.path.join(dir_path_1, 'test'))
sys.path.append(os.path.join(dir_path_1, 'src'))
sys.path.append(os.path.join(dir_path_1, 'runs'))


from src2.Configuration import config
import src2.Configuration as Configuration
from src2.main.Generation import Generation
import src2.main.Singleton as Singleton

if TYPE_CHECKING:
    from src2.Phenotype import NeuralNetwork


def main():
    if not RunsManager.does_run_folder_exist():
        """"fresh run"""
        parse_config()
        RunsManager.set_up_run_folder()
        RunsManager.save_config(config)
        generation = Generation()
    else:
        """continuing run"""
        generation = RunsManager.load_latest_generation()
        loaded_config = RunsManager.load_config()
        if loaded_config is not None:
            Configuration.config = loaded_config

    set_up_operators()
    Singleton.instance = generation

    while generation.generation_number < config.n_generations:
        print('\n\nStarted generation:', generation.generation_number)
        RunsManager.save_generation(generation)
        evolve_generation(generation)


def parse_config():
    parser = argparse.ArgumentParser(description='CoDeepNEAT')
    parser.add_argument('-c', '--configs', nargs='+', type=str,
                        help='Path to all config files that will be used. (Earlier configs are given preference)',
                        required=False)
    args = parser.parse_args()
    # Reading configs in reverse order so that initial config files overwrite others
    if args.configs is None:
        return

    for cfg_file in reversed(args.configs):
        config.read(cfg_file)


def evolve_generation(generation: Generation):
    generation.step()


def fully_train_nn(model: NeuralNetwork):
    pass


def set_up_operators():
    from src2.Genotype.NEAT.Operators.PopulationRankers.SingleObjectiveRank import SingleObjectiveRank
    from src2.Genotype.NEAT.Operators.RepresentativeSelectors.BestRepSelector import BestRepSelector
    from src2.Genotype.NEAT.Operators.RepresentativeSelectors.CentroidRepSelector import CentroidRepSelector
    from src2.Genotype.NEAT.Operators.RepresentativeSelectors.RandomRepSelector import RandomRepSelector
    from src2.Genotype.NEAT.Operators.ParentSelectors.RouletteSelector import RouletteSelector
    from src2.Genotype.NEAT.Operators.ParentSelectors.TournamentSelector import TournamentSelector
    from src2.Genotype.NEAT.Operators.ParentSelectors.UniformSelector import UniformSelector
    from src2.Genotype.NEAT.Species import Species
    from src2.Genotype.NEAT.Population import Population

    if not config.multiobjective:
        Population.ranker = SingleObjectiveRank()
    else:
        # TODO multiobjective rank
        raise NotImplemented('Multi-objectivity is not yet implemented')

    if config.parent_selector.lower() == "uniform":
        Species.selector = UniformSelector()
    elif config.parent_selector.lower() == "roulette":
        Species.selector = RouletteSelector()
    elif config.parent_selector.lower() == "tournament":
        Species.selector = TournamentSelector(5)
    else:
        raise Exception("unrecognised parent selector in config: " + str(config.parent_selector).lower() +
                        " expected either: uniform | roulette | tournament")

    if config.representative_selector.lower() == "centroid":
        Species.representative_selector = CentroidRepSelector()
    elif config.representative_selector.lower() == "random":
        Species.representative_selector = RandomRepSelector()
    elif config.representative_selector.lower() == "best":
        Species.representative_selector = BestRepSelector()
    else:
        raise Exception("unrecognised representative selector in config: " + config.representative_selector.lower()
                        + " expected centroid | random | best")


if __name__ == '__main__':
    main()
