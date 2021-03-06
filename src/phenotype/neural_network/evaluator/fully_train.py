from __future__ import annotations

import os
from typing import TYPE_CHECKING, List, Tuple
import wandb

import src.main.singleton as S

from src.analysis.run import get_run
from configuration import config, internal_config
from src.genotype.cdn.genomes.blueprint_genome import BlueprintGenome
from src.phenotype.neural_network.evaluator.data_loader import get_data_shape
from src.phenotype.neural_network.evaluator.evaluator import evaluate, RETRY
from src.phenotype.neural_network.feature_multiplication import get_model_of_target_size
from src.phenotype.neural_network.neural_network import Network
from src.utils.mp_utils import create_eval_pool
from src.utils.wandb_utils import resume_ft_run, new_ft_run

if TYPE_CHECKING:
    from src.analysis.run import Run

MAX_RETRIES = 5


def fully_train(run_name, n=1):
    """
    Loads and trains from a saved run.
    This will parallelize all training of all the best networks across the given config.ft_feature_multipliers. i.e each
    different feature multiplier will gets its own process and own gpu if available.

    :param run_name: name of the old run
    :param n: number of the best networks to train
    """
    print('Fully training...')
    internal_config.ft_started = True
    internal_config.state = 'ft'

    run: Run = get_run(run_name)
    best_blueprints = run.get_most_accurate_blueprints(n)
    in_size = get_data_shape()

    with create_eval_pool(None) as pool:
        for feature_mul in config.ft_feature_multipliers:
            pool.submit(setup_and_evaluate, run, best_blueprints, in_size, feature_mul)

    internal_config.finished = True
    internal_config.state = 'finished'


def setup_and_evaluate(run: Run, blueprints: List[Tuple[BlueprintGenome, int]], in_size: List[int], feature_mul: int):
    if config.use_wandb:
        if config.resume_fully_train:
            resume_ft_run(True)
        else:
            new_ft_run(True)

    for blueprint, gen_num in blueprints:
        eval_with_retries(run, blueprint, gen_num, in_size, feature_mul)

    wandb.join()


def eval_with_retries(run: Run, blueprint: BlueprintGenome, gen_num: int, in_size: List[int], feature_mul: int):
    """
    Evaluates a run and automatically retries is accuracy is not keeping up with the accuracy achieved in evolution
    """
    accuracy = RETRY
    remaining_retries = MAX_RETRIES
    while accuracy == RETRY and remaining_retries >= 0:
        model: Network = _create_model(run, blueprint, gen_num, in_size, feature_mul)

        if config.resume_fully_train and os.path.exists(model.save_location()):
            model = _load_model(blueprint, run, gen_num, in_size)

        if config.use_wandb:
            wandb.watch(model, criterion=model.loss_fn, log='all', idx=blueprint.id)

        if remaining_retries > 0:
            attempt_number = MAX_RETRIES - remaining_retries
            accuracy = evaluate(model, config.fully_train_max_epochs, blueprint.max_acc, attempt_number)
        else:  # give up retrying, take whatever is produced from training
            accuracy = evaluate(model, config.fully_train_max_epochs)

        if accuracy == RETRY:
            print('retrying fully training')
            internal_config.ft_epoch = 0

        remaining_retries -= 1

    print(f'Achieved a final accuracy of: {accuracy * 100}')


def _create_model(run: Run, blueprint: BlueprintGenome, gen_num, in_size, target_feature_multiplier) -> Network:
    S.instance = run.generations[gen_num]
    modules = run.get_modules_for_blueprint(blueprint)
    model: Network = Network(blueprint, in_size, sample_map=blueprint.best_module_sample_map,
                             allow_module_map_ignores=False, feature_multiplier=1,
                             target_feature_multiplier=target_feature_multiplier).to(config.get_device())
    model_size = sum(p.numel() for p in model.parameters() if p.requires_grad)
    sample_map = model.sample_map

    if target_feature_multiplier != 1:
        model = get_model_of_target_size(blueprint, sample_map, model_size, in_size,
                                         target_size=model_size * target_feature_multiplier)
        model.target_feature_multiplier = target_feature_multiplier

    print("Blueprint: {}\nModules: {}\nSample map: {}\n Species used: {}"
          .format(blueprint,
                  modules,
                  blueprint.best_module_sample_map,
                  list(set(
                      [blueprint.nodes[node_id].species_id for node_id in blueprint.get_fully_connected_node_ids()]))))
    print("Training model which scored: {} in evolution , with {} parameters with feature mult: {}\n"
          .format(blueprint.max_acc, model.size(), target_feature_multiplier))

    return model


def _load_model(dummy_bp: BlueprintGenome, run: Run, gen_num: int, in_size) -> Network:
    if not config.resume_fully_train:
        raise Exception('Calling resume training, but config.resume_fully_train is false')

    S.instance = run.generations[gen_num]
    model: Network = Network(dummy_bp, in_size, sample_map=dummy_bp.best_module_sample_map,
                             allow_module_map_ignores=False).to(config.get_device())
    model.load()

    return model
