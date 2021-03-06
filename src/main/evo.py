from __future__ import annotations

import atexit
import os
import sys

# For importing project files
dir_path = os.path.dirname(os.path.realpath(__file__))
dir_path_1 = os.path.split(os.path.split(dir_path)[0])[0]
sys.path.append(dir_path_1)
sys.path.append(os.path.join(dir_path_1, 'src'))
sys.path.append(os.path.join(dir_path_1, 'test'))
sys.path.append(os.path.join(dir_path_1, 'runs'))
sys.path.append(os.path.join(dir_path_1, 'configuration'))

from src.utils.main_common import _force_cuda_device_init, evolve
from configuration import internal_config
from src.utils.wandb_utils import resume_evo_run, new_evo_run
from configuration.config_loader import load_simple_config, get_cli_args


def main():
    args = get_cli_args()
    load_simple_config(args.config, resume_evo_run, new_evo_run, args.ngpus)

    _force_cuda_device_init()

    evolve()


if __name__ == '__main__':
    atexit.register(internal_config.on_exit)
    main()
