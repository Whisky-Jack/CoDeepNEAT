from __future__ import annotations

import multiprocessing as mp
# import threading
from typing import TYPE_CHECKING, List

from src2.configuration import config
from src2.phenotype.neural_network.evaluator.evaluator import evaluate
from src2.phenotype.neural_network.neural_network import Network

if TYPE_CHECKING:
    from src2.genotype.cdn.genomes.blueprint_genome import BlueprintGenome

bp_lock = mp.Lock()


def evaluate_blueprints(blueprints: mp.Queue,
                        completed_blueprints: mp.List,
                        input_size: List[int],
                        generation_num: int,
                        num_epochs: int = config.epochs_in_evolution):
    """
    Consumes blueprints off the blueprints queue, evaluates them and adds them back to the queue if all of their
    evaluations have not been completed for the current generation. If all their evaluations have been completed, add
    them to the completed_blueprints list.

    :param blueprints:
    :param completed_blueprints:
    :param input_size:
    :param generation_num:
    :param num_epochs:
    :return:
    """
    print(mp.current_process().name)
    while blueprints.qsize() != 0:
        blueprint = blueprints.get()
        print("proc %s got bp %i from q with %i evals" % (
        mp.current_process().name, blueprint.id, blueprint.n_evaluations))

        blueprint = evaluate_blueprint(blueprint, input_size, generation_num, num_epochs)

        if blueprint.n_evaluations == config.n_evaluations_per_bp:
            completed_blueprints.append(blueprint)
            continue

        blueprints.put(blueprint)
        print('proc %s put bp %i back on q with %i evals' % (
        mp.current_process().name, blueprint.id, blueprint.n_evaluations))


def evaluate_blueprint(blueprint: BlueprintGenome, input_size: List[int], generation_num: int,
                       num_epochs=config.epochs_in_evolution) -> BlueprintGenome:
    """
    Parses the blueprint into its phenotype NN
    Handles the assignment of the single/multi obj finesses to the blueprint in parallel
    """
    device = config.get_device()

    model: Network = Network(blueprint, input_size).to(device)

    model_size = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if model_size > config.max_model_params:
        accuracy = 0
    else:
        accuracy = evaluate(model, num_epochs=num_epochs)

    blueprint.update_best_sample_map(model.sample_map, accuracy)
    blueprint.report_fitness([accuracy], module_sample_map=model.sample_map)
    parse_number = blueprint.n_evaluations

    print("Evaluation of genome:", blueprint.id, "complete with accuracy:", accuracy, "by thread",
          mp.current_process().name)

    if config.plot_every_genotype:
        blueprint.visualize(parse_number=parse_number,
                            prefix="g" + str(generation_num) + "_" + str(blueprint.id))

    if config.plot_every_phenotype:
        model.visualize(parse_number=parse_number,
                        prefix="g" + str(generation_num) + "_" + str(blueprint.id))

    return blueprint
