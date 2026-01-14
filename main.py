import logging
import os
import sys

import torch
from torch_geometric import seed_everything
from exp.exp_attack import Exp_Attack
from exp.exp_fairgu import ExpFairGU
from parameter_parser import parameter_parser
import warnings

import csv

warnings.filterwarnings("ignore")
seed_everything(2019816)


def config_logger(save_name):
    # create logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(levelname)s:%(asctime)s: - %(name)s - : %(message)s')

    # create console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(ch)


if __name__ == "__main__":
    args = parameter_parser()
    csv_path = args['csv_name'] + ".csv"
    csv_dir = os.path.dirname(csv_path)
    if csv_dir and not os.path.exists(csv_dir):
        os.makedirs(csv_dir, exist_ok=True)
    if not os.path.exists(csv_path):
        with open(csv_path, 'w', newline='') as csvfile:
            fieldnames = [
                'exp', 'dataset_name', 'target_model', 'unlearn_task',
                'unlearn_ratio',
                'f1_avg', 'f1_std', 'acc_avg', 'acc_std', 'sp_avg',
                'sp_std', 'eo_avg', 'eo_std',
                'f1_avg_unlearn', 'f1_std_unlearn', 'acc_avg_unlearn', 'acc_std_unlearn',
                'sp_avg_unlearn', 'sp_std_unlearn', 'eo_avg_unlearn', 'eo_std_unlearn',
                'unlearning_time_avg'
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()


    # config the logger
    logger_name = "_".join((args['method'], args['target_model'], args['dataset_name'],
                            args['unlearn_task'], str(args['unlearn_ratio'])))
    config_logger(logger_name)
    logging.info(logger_name)

    torch.set_num_threads(args["num_threads"])
    torch.cuda.set_device(args["cuda"])
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args["cuda"])
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    if args["exp"].lower() == "unlearn":
        if args["method"].lower() == "fairgu":
            ExpFairGU(args)
        else:
            raise NotImplementedError
    elif args["exp"].lower() == "attack":
        Exp_Attack(args)
    else:
        raise NotImplementedError
