import csv
import logging
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
from sklearn.decomposition import PCA

from exp.mia.MLG_TSTF import MIA
from lib_dataset.data_store import DataStore
from parameter_parser import parameter_parser

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOGGER = logging.getLogger("ExpAttack")

def _load_dataset(args: Dict[str, Any]):
    data_store = DataStore(args)
    data = data_store.load_raw_data()
    return data

def _resolve_checkpoint_path(args: Dict[str, Any]) -> Path:
    prefix = Path(args["attack_file_prefix"])
    if not prefix.is_absolute():
        prefix = PROJECT_ROOT / prefix

    file_path = (
        f"{prefix}_{args['dataset_name']}_{args['unlearn_task']}_"
        f"{args['unlearn_ratio']}_{args['target_model']}.pth"
    )
    checkpoint = Path(file_path)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Attack material not found: {checkpoint}")

    return checkpoint


def _ensure_csv_header(csv_path: Path):
    # Create result directory if it doesn't exist
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if not csv_path.exists():
        with open(csv_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(
                [
                    "Dataset",
                    "Exp",
                    "Unlearn Ratio",
                    "Target Model",
                    "AUC Mean",
                    "ACC Mean",
                    "ACC_POS Mean",
                    "F1_POS Mean",
                ]
            )


def Exp_Attack(args: Dict[str, Any]):
    data = _load_dataset(args)
    LOGGER.info(
        "Attack task: dataset=%s",
        args["dataset_name"],
    )

    checkpoint_path = _resolve_checkpoint_path(args)
    model = torch.load(checkpoint_path, map_location="cpu")

    auc_list, acc_list, acc_pos_list, f1_pos_list = [], [], [], []

    for _ in range(args.get("attack_num_runs", 1)):
        train_indices = model["train_indices"].nonzero().view(-1)
        test_indices = model["test_indices"].nonzero().view(-1)
        mia = MIA(
            args,
            model["predicted_prob"],
            model["removed_nodes"],
            train_indices,
            test_indices,
        )
        acc_pos, f1_pos, auc, acc = mia.get_results()
        auc_list.append(auc)
        acc_list.append(acc)
        acc_pos_list.append(acc_pos)
        f1_pos_list.append(f1_pos)

    auc_mean = float(np.mean(auc_list))
    acc_mean = float(np.mean(acc_list))
    acc_pos_mean = float(np.mean(acc_pos_list))
    f1_pos_mean = float(np.mean(f1_pos_list))

    csv_path = Path(args["attack_csv_name"])
    if csv_path.suffix.lower() != ".csv":
        csv_path = csv_path.with_suffix(".csv")
    _ensure_csv_header(csv_path)
    with open(csv_path, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            [
                args["dataset_name"],
                args["unlearn_task"],
                args["unlearn_ratio"],
                args["target_model"],
                auc_mean,
                acc_mean,
                acc_pos_mean,
                f1_pos_mean,
            ]
        )
    LOGGER.info(
        "Attack Result: AUC=%.4f, ACC=%.4f, ACC_POS=%.4f, F1_POS=%.4f -> %s",
        auc_mean,
        acc_mean,
        acc_pos_mean,
        f1_pos_mean,
        csv_path,
    )


if __name__ == "__main__":
    cli_args = parameter_parser()
    Exp_Attack(cli_args)