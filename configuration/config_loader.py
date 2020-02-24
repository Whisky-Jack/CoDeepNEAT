from typing import Tuple, Union

from configuration import config, batch_runner
from src.utils.wandb_data_fetcher import download_run
from runs import runs_manager

import argparse

"""
    there are 3 possible levels of configs to be loaded:
    1: a saved config which is attached to an existing run which has been executed before
        this config does not exist when starting a fresh run, only when continuing an existing one
    2: a scheduled config. If a run scheduler is used, it will point to a one of the configs in its schedule
    3: the cli config, which is specified as a run arg to the main program
    
    when no run schedule is used, the cli config values overwrite the saved config (if one exists)
        an example of when this is desirable is to change the num gpu's when continuing a run, or 
        to change the man num of generations, to evolve a population for longer
        
    when a run schedule is specified, it will fetch a config file eg: elite.json
    It may be desirable to override certain properties of all runs in a schedule
        An example of this is schedule = {elite,base} - we may want to turn on DataAug for bot
        ie: transform the schedule into {da_elite,da_base}
    
    thus when a run schedule is used, the cli config starting the schedule may contain overriding config values (eg: da=true)
    
    therefore the priority of configs when a schedule is being used is:
        saved config (if exists)    - lowest
        scheduled config            - middle
        cli config                  - highest
"""


def load_config():
    cli_cfg_file_name, cli_n_gpus = get_cli_args()

    effective_run_name, scheduled_cfg_file_name = check_and_execute_batch_scheduler(cli_cfg_file_name)
    effective_run_name = load_saved_config(effective_run_name, cli_cfg_file_name)

    if scheduled_cfg_file_name:
        print("reading scheduled config: ", scheduled_cfg_file_name)
        config.read(scheduled_cfg_file_name)

    print('Reading cli config', cli_cfg_file_name)
    config.read(cli_cfg_file_name)  # final authority on config values

    if scheduled_cfg_file_name:
        # must detect whether the scheduler is calling for a fully train, or an evolutionary run
        fully_train, resume_fully_train = batch_runner.get_fully_train_state(effective_run_name)
        print("scheduler is starting run with FT=",fully_train, "continue train:",resume_fully_train)
        config.fully_train = fully_train
        config.resume_fully_train = resume_fully_train

    config.run_name = effective_run_name
    if cli_n_gpus is not None:
        config.n_gpus = cli_n_gpus
    else:
        print("no gpu argument given, using cli config value of", config.n_gpus)


def check_and_execute_batch_scheduler(cli_cfg_file_name) -> Tuple[str,Union[str, None]]:
    """
        finds out the effective run name to be used
        if there is a run scheduler, the run config it is scheduling is fetched, but not parsed
        the effective name, as well as the Optional scheduled_run_name are returned
    """
    batch_run_scheduler = config.read_option(cli_cfg_file_name, 'batch_run_scheduler')
    cli_cfg_run_name = config.read_option(cli_cfg_file_name, 'run_name')

    if batch_run_scheduler:  # there is a batch run scheduler so must use the config specified in it
        scheduled_cfg_file_name, scheduled_run_name = batch_runner.get_config_path(batch_run_scheduler, cli_cfg_run_name)
        return scheduled_run_name, scheduled_cfg_file_name

    if not cli_cfg_run_name:
        raise Exception("no run name given. configs for non scheduled runs must provide run name")

    # this is a standard - non scheduled run, so just the cli run name is used
    return cli_cfg_run_name, None


def load_saved_config(effective_run_name, cli_cfg_file_name):
    wandb_run_path = config.read_option(cli_cfg_file_name, 'wandb_run_path')

    if wandb_run_path:
        # wandb downloaded runs are not used by the batch scheduler.
        # this must be a standalone run
        print('wandb run path:', wandb_run_path)
        print('downloading run...')
        # if a wandb run path is specified - the cli configs run name is ignored, instead the run name
        # is determined by the saved config
        effective_run_name = download_run(run_path=wandb_run_path, replace=True)

    print('Run name', effective_run_name)
    if runs_manager.run_folder_exists(effective_run_name):
        print('Run folder already exists, reading its config')
        runs_manager.load_config(effective_run_name)

    return effective_run_name


def get_cli_args() -> Tuple[str, int]:
    parser = argparse.ArgumentParser(description='CoDeepNEAT')
    parser.add_argument('-c', '--config', type=str, help='Config file that will be used', required=True)
    parser.add_argument('-g', '--ngpus', type=int, help='Number of GPUs available', required=False)

    args = parser.parse_args()
    return args.config, args.ngpus