from __future__ import annotations
import argparse
from typing import TYPE_CHECKING

from src2.Configuration import config
from src2.main.Generation import Generation

if TYPE_CHECKING:
    from src2.Phenotype import NeuralNetwork


def main():
    parser = argparse.ArgumentParser(description='CoDeepNEAT')
    parser.add_argument('-c', '--configs', nargs='+', type=str,
                        help='Path to all config files that will be used. (Earlier configs are given preference)',
                        required=False)
    args = parser.parse_args()
    # Reading configs in reverse order so that initial config files overwrite others
    for cfg_file in reversed(args.configs):
        config.read(cfg_file)

    generation = Generation()
    for _ in range(config.n_generations):
        evolve_generation(generation)


def evolve_generation(generation: Generation):
    generation.step()


def fully_train_nn(model: NeuralNetwork):
    pass


if __name__ == '__main__':
    main()
