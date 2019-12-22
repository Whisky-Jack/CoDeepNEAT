"""
    the generation class is a container for the 3 cdn populations.
    It is also responsible for stepping the evolutionary cycle.
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
import torch.multiprocessing as mp
from typing import Optional, List

import wandb

from runs import runs_manager
from src2.configuration import config
from src2.genotype.cdn.genomes.blueprint_genome import BlueprintGenome
from src2.genotype.cdn.genomes.da_genome import DAGenome
from src2.genotype.cdn.genomes.module_genome import ModuleGenome
from src2.genotype.cdn.mutators.blueprint_genome_mutator import BlueprintGenomeMutator
from src2.genotype.cdn.mutators.module_genome_mutator import ModuleGenomeMutator
from src2.genotype.cdn.nodes.blueprint_node import BlueprintNode
from src2.genotype.cdn.nodes.da_node import DANode
from src2.genotype.cdn.nodes.module_node import ModuleNode
from src2.genotype.cdn.population_initializer import create_population, create_mr
from src2.genotype.neat.operators.speciators.most_similar_speciator import MostSimilarSpeciator
from src2.genotype.neat.operators.speciators.neat_speciator import NEATSpeciator
from src2.genotype.neat.population import Population
from src2.phenotype.neural_network.evaluator.data_loader import get_data_shape
from src2.phenotype.neural_network.neural_network import Network
from src2.phenotype.phenotype_evaluator import evaluate_blueprint, evaluate_blueprints


class Generation:
    def __init__(self):
        self.genome_id_counter = 0  # max genome id of all genomes contained in this generation obj
        import src2.main.singleton as Singleton
        Singleton.instance = self

        self.module_population: Optional[Population] = None
        self.blueprint_population: Optional[Population] = None
        self.da_population: Optional[Population] = None

        self.initialise_populations()
        self.generation_number = 0

    def step_evaluation(self):
        eval_start_time = time.time()
        model_sizes = self.evaluate_blueprints()  # may be parallel
        num_evals = len(self.blueprint_population) * config.n_evaluations_per_bp
        time_taken = time.time() - eval_start_time
        print("finished ", num_evals, "evals in", time_taken, "seconds, av:", (time_taken / num_evals), "num threads:",
              config.n_gpus)

        # Aggregate the fitnesses immediately after they have all been recorded
        self.module_population.aggregate_fitness()
        self.blueprint_population.aggregate_fitness()

        if config.use_wandb:
            self.wandb_report(model_sizes)

    def step_evolution(self):
        """
            Runs cdn for one generation. Calls the evaluation of all individuals. Prepares population objects for the
            next step.
        """
        # TODO move this to visualization method
        most_accurate_blueprint: BlueprintGenome = self.blueprint_population.get_most_accurate()
        if config.plot_best_genotypes:
            most_accurate_blueprint.visualize(prefix="best_g" + str(self.generation_number) + "_")
        if config.plot_best_phenotype:
            model: Network = Network(most_accurate_blueprint, get_data_shape(),
                                     sample_map=most_accurate_blueprint.best_module_sample_map)
            model.visualize(prefix="best_g" + str(self.generation_number) + "_")

        print("Num blueprint species:", len(self.blueprint_population.species), self.blueprint_population.species)
        print("Most accurate graph:", self.blueprint_population.get_most_accurate().fitness_values)

        self.module_population.step()
        self.blueprint_population.step()

        # self.module_population.visualise(suffix="_" + str(self.generation_number) + "_module_species")
        # self.blueprint_population.visualise(suffix="_" + str(self.generation_number) + "blueprint_species")

        self.generation_number += 1

        self.module_population.end_step()
        self.blueprint_population.end_step()
        if config.evolve_data_augmentations:
            self.da_population.end_step()

        print('Step ended')
        print('Module species:', [len(spc.members) for spc in self.module_population.species])

    def evaluate_blueprints(self) -> List[int]:
        """Evaluates all blueprints"""
        # Multiplying the blueprints so that each blueprint is evaluated config.n_evaluations_per_bp times

        self.module_population.before_step()
        self.blueprint_population.before_step()

        # blueprints choosing DA schemes from population
        if config.evolve_data_augmentations:
            self.da_population.before_step()
            for blueprint_individual in self.blueprint_population:
                blueprint_individual.pick_da_scheme()

        manager = mp.Manager()
        blueprints = list(self.blueprint_population) * config.n_evaluations_per_bp
        blueprints_q = manager.Queue(len(self.blueprint_population) * config.n_evaluations_per_bp)

        for bp in blueprints:
            blueprints_q.put(bp, False)

        in_size = get_data_shape()
        model_sizes: List[int] = []

        print("num blueprints:", len(self.blueprint_population), "num evals:", len(blueprints))
        print("num modules:", len(self.module_population))

        consumers = []
        for gpu in range(config.n_gpus):
            consumers.append(mp.Process(target=evaluate_blueprints, args=(blueprints_q, in_size, self.generation_number), name=str(gpu)))
            consumers[-1].start()

        for consumer in consumers:
            consumer.join()

        return model_sizes

    def initialise_populations(self):
        """Starts off the populations of a new evolutionary run"""
        # TODO this is using the old method
        if config.module_speciation.lower() == "similar":
            module_speciator = MostSimilarSpeciator(config.species_distance_thresh_mod_base, config.n_module_species,
                                                    ModuleGenomeMutator())
        elif config.module_speciation.lower() == "neat":
            module_speciator = NEATSpeciator(config.species_distance_thresh_mod_base, config.n_module_species,
                                             ModuleGenomeMutator())
        else:
            raise Exception(
                "speciation method in config not recognised: " + str(config.module_speciation).lower()
                + " expected: similar | neat")

        bp_speciator = NEATSpeciator(config.species_distance_thresh_mod_base, config.n_blueprint_species,
                                     BlueprintGenomeMutator())

        self.module_population = Population(create_population(config.module_pop_size, ModuleNode, ModuleGenome),
                                            create_mr(), config.module_pop_size, module_speciator)

        self.blueprint_population = Population(
            create_population(config.bp_pop_size, BlueprintNode, BlueprintGenome),
            create_mr(), config.bp_pop_size, bp_speciator)

        print("initialised pops, bps:", len(self.blueprint_population), "mods:", len(self.module_population))

        # TODO DA pop
        if config.evolve_data_augmentations:
            self.da_population = Population(create_population(config.da_pop_size, DANode, DAGenome),
                                            create_mr(), config.da_pop_size, bp_speciator)

    # TODO move this to a wandb manager or something similar
    def wandb_report(self, model_sizes: List[int]):
        module_accs = sorted([module.accuracy for module in self.module_population])
        bp_accs = sorted([bp.accuracy for bp in self.blueprint_population])

        n_unevaluated_bps = 0
        raw_bp_accs = []
        for bp in self.blueprint_population:
            n_unevaluated_bps += sum(fitness[0] == 0 for fitness in bp.fitness_raw)
            raw_bp_accs.extend(bp.fitness_raw[0])

        n_unevaluated_mods = 0
        raw_mod_accs = []
        for mod in self.module_population:
            n_unevaluated_mods += 1 if mod.n_evaluations == 0 else 0
            raw_mod_accs.extend(mod.fitness_raw[0])

        mod_acc_tbl = wandb.Table(['module accuracies'],
                                  data=raw_mod_accs)
        bp_acc_tbl = wandb.Table(['blueprint accuracies'],
                                 data=raw_bp_accs)
        # Saving the pickle file for further inspection
        wandb.save(runs_manager.get_generation_file_path(self.generation_number, config.run_name))

        wandb.log({'module accuracy table': mod_acc_tbl, 'blueprint accuracy table': bp_acc_tbl,
                   'module accuracies aggregated': module_accs, 'blueprint accuracies aggregated': bp_accs,
                   'module accuracies raw': raw_mod_accs, 'blueprint accuracies raw': raw_bp_accs,
                   'avg module accuracy': sum(module_accs) / len(module_accs),
                   'avg blueprint accuracy': sum(bp_accs) / len(bp_accs),
                   'best module accuracy': max(raw_mod_accs), 'best blueprint accuracy': max(raw_bp_accs),
                   'num module species': len(self.module_population.species),
                   'species sizes': [len(spc.members) for spc in self.module_population.species],
                   'unevaluated blueprints': n_unevaluated_bps, 'n_unevaluated_mods': n_unevaluated_mods,
                   'speciation threshold': self.module_population.speciator.threshold,
                   'model sizes': model_sizes})

    def __getitem__(self, genome_id: int):
        if config.evolve_data_augmentations:
            populations: List[Population] = [self.blueprint_population, self.module_population, self.da_population]
        else:
            populations: List[Population] = [self.blueprint_population, self.module_population]

        for pop in populations:
            mem = pop[genome_id]
            if mem is not None:
                return mem

        return None
