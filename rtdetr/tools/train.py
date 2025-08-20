"""by lyuwenyu"""

import sys
import os

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..'))
""" Inserts that parent directory path at the front of sys.path, which is the list of directories Python checks when you do import ..., to have the highest priority during module lookup. """

from src.core import YAMLConfig
import src.misc.dist as dist
from src.solver import TASKS
import argparse


def main(args, ) -> None:
    '''main
    '''
    dist.init_distributed()
    """ to initialize the distributed training environment, which sets up communication across multiple GPUs/machines if the program runs in parallel. """

    assert not all([args.tuning, args.resume]), \
        'Only support from_scratch(new training) or resume or tuning at one time'

    cfg = YAMLConfig(
        args.config,
        resume=args.resume,
        use_amp=args.amp,
        tuning=args.tuning
    )
    # print(cfg.model)
    '''
    Creates a configuration object from a YAML file (args.config) which is the model configuration file.
    resume, amp, and tuning are flags passed in.
        - resume=True → load checkpoint.
        - amp=True → use automatic mixed precision (faster training with half-precision floats).
        - tuning=True → fine-tuning mode.
    So cfg now holds all model/training settings.
    '''
    
    solver = TASKS[cfg.yaml_cfg['task']](cfg)
    '''
    cfg.yaml_cfg['task'] → gets the task name from the model's config.
    TASKS[...] → picks the right solver class.
    (cfg) → creates an instance of that solver with the configuration.
    So now solver is the main object that knows how to train/validate the model.
    '''

    if args.show_flops:
        solver.show_flops()
        '''to measure model complexity, how heavy it is'''

    if args.test_only:
        solver.val()
        '''runs validation only, that is evaluates the model without training.'''

    else:
        solver.fit()
        '''training process starts'''


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, )
    parser.add_argument('--resume', '-r', type=str, )
    parser.add_argument('--tuning', '-t', type=str, )
    parser.add_argument('--test-only', action='store_true', default=False,)
    parser.add_argument('--amp', action='store_true', default=False,)
    parser.add_argument("--show-flops", '-f', action='store_true',default=False)

    args = parser.parse_args()

    main(args)


"""Summary:
Setup distributed training.
Check you’re not mixing modes.
Load YAML config.
Pick the correct solver (trainer) for the task.
Optionally show FLOPs.
Either validate (test-only) or train (fit).
"""