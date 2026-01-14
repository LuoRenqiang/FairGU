import logging
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

torch.cuda.empty_cache()
import numpy as np
from exp.exp import Exp
from lib_gnn_model.node_classifier import NodeClassifier
from sklearn.metrics import f1_score, accuracy_score
import csv
import os

from exp.fim_graph import FIMUnlearner

class ExpFairGU(Exp):
    def __init__(self, args):
        super(ExpFairGU, self).__init__(args)

        self.logger = logging.getLogger('ExpFairGU')

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.load_data()
        self.num_feats = self.data.num_features
        self.train_test_split()
        self.unlearning_request()

        self.target_model_name = self.args['target_model']

        self.determine_target_model()

        self.num_layers = 2

        self.run_f1 = np.empty(0)
        self.run_acc = np.empty(0)
        self.run_sp = np.empty(0)
        self.run_eo = np.empty(0)

        self.run_f1_unlearning = np.empty(0)
        self.run_acc_unlearning = np.empty(0)
        self.run_sp_unlearning = np.empty(0)
        self.run_eo_unlearning = np.empty(0)

        self.unlearning_times = np.empty(0)
        self.training_times = np.empty(0)
        for run in range(self.args['num_runs']):
            # self.logger.info("Run %d" % run)
            run_training_time, _ = self._train_model(run)
            f1, acc, sp_score, eo_score = self.evaluate(run)
            self.run_f1 = np.append(self.run_f1, f1)
            self.run_acc = np.append(self.run_acc, acc)
            self.run_sp = np.append(self.run_sp, sp_score)
            self.run_eo = np.append(self.run_eo, eo_score)
            self.training_times = np.append(self.training_times, run_training_time)

            unlearning_time, f1_unlearning, acc_unlearning, sp_unlearn, eo_unlearn = self.fim_unlearning()

            self.unlearning_times = np.append(self.unlearning_times, unlearning_time)
            self.run_f1_unlearning = np.append(self.run_f1_unlearning, f1_unlearning)
            self.run_acc_unlearning = np.append(self.run_acc_unlearning, acc_unlearning)
            self.run_sp_unlearning = np.append(self.run_sp_unlearning, sp_unlearn)
            self.run_eo_unlearning = np.append(self.run_eo_unlearning, eo_unlearn)

        self.run_f1_avg = np.average(self.run_f1) * 100
        self.run_f1_std = np.std(self.run_f1) * 100
        self.f1_score_avg = f"{self.run_f1_avg:.2f}"
        self.f1_score_std = f"{self.run_f1_std:.2f}"

        self.run_acc_avg = np.average(self.run_acc) * 100
        self.run_acc_std = np.std(self.run_acc) * 100
        self.acc_score_avg = f"{self.run_acc_avg:.2f}"
        self.acc_score_std = f"{self.run_acc_std:.2f}"

        self.run_sp_avg = np.average(self.run_sp) * 100
        self.run_sp_std = np.std(self.run_sp) * 100
        self.sp_score_avg = f"{self.run_sp_avg:.2f}"
        self.sp_score_std = f"{self.run_sp_std:.2f}"

        self.run_eo_avg = np.average(self.run_eo) * 100
        self.run_eo_std = np.std(self.run_eo) * 100
        self.eo_score_avg = f"{self.run_eo_avg:.2f}"
        self.eo_score_std = f"{self.run_eo_std:.2f}"

        self.run_f1_unlearning_avg = np.average(self.run_f1_unlearning) * 100
        self.run_f1_unlearning_std = np.std(self.run_f1_unlearning) * 100
        self.f1_score_unlearning_avg = f"{self.run_f1_unlearning_avg:.2f}"
        self.f1_score_unlearning_std = f"{self.run_f1_unlearning_std:.2f}"

        self.run_acc_unlearning_avg = np.average(self.run_acc_unlearning) * 100
        self.run_acc_unlearning_std = np.std(self.run_acc_unlearning) * 100
        self.acc_score_unlearning_avg = f"{self.run_acc_unlearning_avg:.2f}"
        self.acc_score_unlearning_std = f"{self.run_acc_unlearning_std:.2f}"

        self.run_sp_unlearning_avg = np.average(self.run_sp_unlearning) * 100
        self.run_sp_unlearning_std = np.std(self.run_sp_unlearning) * 100
        self.sp_score_unlearning_avg = f"{self.run_sp_unlearning_avg:.2f}"
        self.sp_score_unlearning_std = f"{self.run_sp_unlearning_std:.2f}"

        self.run_eo_unlearning_avg = np.average(self.run_eo_unlearning) * 100
        self.run_eo_unlearning_std = np.std(self.run_eo_unlearning) * 100
        self.eo_score_unlearning_avg = f"{self.run_eo_unlearning_avg:.2f}"
        self.eo_score_unlearning_std = f"{self.run_eo_unlearning_std:.2f}"

        self.unlearning_time_avg = np.average(self.unlearning_times)

        self.logger.info(
            f"|Unlearn| F1: {self.f1_score_unlearning_avg}+-{self.f1_score_unlearning_std} | "
            f"ACC: {self.acc_score_unlearning_avg}+-{self.acc_score_unlearning_std} | "
            f"SP: {self.sp_score_unlearning_avg}+-{self.sp_score_unlearning_std} | "
            f"EO: {self.eo_score_unlearning_avg}+-{self.eo_score_unlearning_std} | "
            f"Time: {self.unlearning_time_avg:.4f}s")

        self.log_to_csv()

    def load_data(self):
        self.data = self.data_store.load_raw_data()

    def train_test_split(self):
        if self.args['is_split']:
            # self.logger.info('splitting train/test data')
            # use the dataset's default split
            self.train_indices, self.test_indices = train_test_split(np.arange(self.data.num_nodes),
                                                                     test_size=self.args['test_ratio'],
                                                                     random_state=100)

            self.data_store.save_train_test_split(self.train_indices, self.test_indices)

            self.data.train_mask = torch.from_numpy(np.isin(np.arange(self.data.num_nodes), self.train_indices))
            self.data.test_mask = torch.from_numpy(np.isin(np.arange(self.data.num_nodes), self.test_indices))
        else:
            self.train_indices, self.test_indices = self.data_store.load_train_test_split()

            self.data.train_mask = torch.from_numpy(np.isin(np.arange(self.data.num_nodes), self.train_indices))
            self.data.test_mask = torch.from_numpy(np.isin(np.arange(self.data.num_nodes), self.test_indices))

    def unlearning_request(self):
        # self.logger.debug("Train data  #.Nodes: %f, #.Edges: %f" % (
        #     self.data.num_nodes, self.data.num_edges))

        self.data.x_unlearn = self.data.x.clone()
        self.data.edge_index_unlearn = self.data.edge_index.clone()
        edge_index = self.data.edge_index.numpy()
        unique_indices = np.where(edge_index[0] < edge_index[1])[0]

        if self.args["unlearn_task"] == 'node':
            unique_nodes = np.random.choice(len(self.train_indices),
                                            int(len(self.train_indices) * self.args['unlearn_ratio']),
                                            replace=False)
            self.data.edge_index_unlearn = self.update_edge_index_unlearn(unique_nodes)

        if self.args["unlearn_task"] == 'edge':
            remove_indices = np.random.choice(
                unique_indices,
                int(unique_indices.shape[0] * self.args['unlearn_ratio']),
                replace=False)
            remove_edges = edge_index[:, remove_indices]
            unique_nodes = np.unique(remove_edges)

            self.data.edge_index_unlearn = self.update_edge_index_unlearn(unique_nodes, remove_indices)

        if self.args["unlearn_task"] == 'feature':
            unique_nodes = np.random.choice(len(self.train_indices),
                                            int(len(self.train_indices) * self.args['unlearn_ratio']),
                                            replace=False)
            self.data.x_unlearn[unique_nodes] = 0.

        self.temp_node = unique_nodes

    def update_edge_index_unlearn(self, delete_nodes, delete_edge_index=None):
        edge_index = self.data.edge_index.numpy()

        unique_indices = np.where(edge_index[0] < edge_index[1])[0]
        unique_indices_not = np.where(edge_index[0] > edge_index[1])[0]

        if self.args["unlearn_task"] == 'edge':
            remain_indices = np.setdiff1d(unique_indices, delete_edge_index)
        else:
            unique_edge_index = edge_index[:, unique_indices]
            delete_edge_indices = np.logical_or(np.isin(unique_edge_index[0], delete_nodes),
                                                np.isin(unique_edge_index[1], delete_nodes))
            remain_indices = np.logical_not(delete_edge_indices)
            remain_indices = np.where(remain_indices == True)

        remain_encode = edge_index[0, remain_indices] * edge_index.shape[1] * 2 + edge_index[1, remain_indices]
        unique_encode_not = edge_index[1, unique_indices_not] * edge_index.shape[1] * 2 + edge_index[
            0, unique_indices_not]
        sort_indices = np.argsort(unique_encode_not)
        remain_indices_not = unique_indices_not[
            sort_indices[np.searchsorted(unique_encode_not, remain_encode, sorter=sort_indices)]]
        remain_indices = np.union1d(remain_indices, remain_indices_not)

        return torch.from_numpy(edge_index[:, remain_indices])

    def determine_target_model(self):
        # self.logger.info('target model: %s' % (self.args['target_model'],))
        num_classes = self.data.num_classes

        self.target_model = NodeClassifier(self.num_feats, num_classes, self.args)

    def evaluate(self, run):
        # self.logger.info('model evaluation')
        start_time = time.time()
        self.target_model.model.eval()
        out = self.target_model.model(self.data.x, self.data.edge_index)
        y = self.data.y.cpu().numpy()

        if self.args['target_model'] in ['FGCN', 'FGAT']:
            y_pred_tensor = (out > 0).long()
            y_pred = y_pred_tensor.cpu().numpy()
        else:
            y_hat = F.log_softmax(out, dim=1).cpu().detach().numpy()
            y_pred = np.argmax(y_hat, axis=1)
        
        mask = self.data.test_mask.cpu().numpy()
        test_f1 = f1_score(y[mask], y_pred[mask], average="macro")
        test_acc = accuracy_score(y[mask], y_pred[mask])
        evaluate_time = time.time() - start_time
        
        test_sp, test_eo = self.target_model.fair_metric(
            y_pred_tensor.unsqueeze(1),
            self.data.test_mask,
            self.data.sens,
            self.data.y
        )

        return test_f1, test_acc, test_sp, test_eo

    def _train_model(self, run):
        # self.logger.info('training target models, run %s' % run)

        start_time = time.time()
        self.target_model.data = self.data
        res = self.target_model.train_model()
        train_time = time.time() - start_time

        # self.data_store.save_target_model(run, self.target_model)
        # self.logger.info(f"Model training time: {train_time:.4f}")

        return train_time, res

    def _fair_metric(self, y_true, sens, pred_y):
        idx_s0 = sens == 0
        idx_s1 = sens == 1

        idx_s0_y1 = idx_s0 & (y_true == 1)
        idx_s1_y1 = idx_s1 & (y_true == 1)

        epsilon = 1e-8
        parity = abs((pred_y[idx_s0].sum() + epsilon) / (idx_s0.sum() + epsilon) -
                     (pred_y[idx_s1].sum() + epsilon) / (idx_s1.sum() + epsilon))
        equality = abs((pred_y[idx_s0_y1].sum() + epsilon) / (idx_s0_y1.sum() + epsilon) -
                       (pred_y[idx_s1_y1].sum() + epsilon) / (idx_s1_y1.sum() + epsilon))

        return parity, equality

    def log_to_csv(self):
        csv_path = self.args['csv_name'] + ".csv"
        csv_dir = os.path.dirname(csv_path)
        if csv_dir and not os.path.exists(csv_dir):
            os.makedirs(csv_dir, exist_ok=True)
        file_exists = os.path.isfile(csv_path)

        need_header = not file_exists or os.path.getsize(csv_path) == 0

        with open(csv_path, 'a' if file_exists else 'w', newline='') as csvfile:
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

            if need_header:
                writer.writeheader()

            writer.writerow({
                'exp': self.args['exp'],
                'dataset_name': self.args['dataset_name'],
                'target_model': self.args['target_model'],
                'unlearn_task': self.args['unlearn_task'],
                'unlearn_ratio': self.args['unlearn_ratio'],
                'f1_avg': self.f1_score_avg,
                'f1_std': self.f1_score_std,
                'acc_avg': self.acc_score_avg,
                'acc_std': self.acc_score_std,
                'sp_avg': self.sp_score_avg,
                'sp_std': self.sp_score_std,
                'eo_avg': self.eo_score_avg,
                'eo_std': self.eo_score_std,
                'f1_avg_unlearn': self.f1_score_unlearning_avg,
                'f1_std_unlearn': self.f1_score_unlearning_std,
                'acc_avg_unlearn': self.acc_score_unlearning_avg,
                'acc_std_unlearn': self.acc_score_unlearning_std,
                'sp_avg_unlearn': self.sp_score_unlearning_avg,
                'sp_std_unlearn': self.sp_score_unlearning_std,
                'eo_avg_unlearn': self.eo_score_unlearning_avg,
                'eo_std_unlearn': self.eo_score_unlearning_std,
                'unlearning_time_avg': self.unlearning_time_avg
            })

    def prepare_forget_data(self):
        forget_mask = torch.zeros(self.data.num_nodes, dtype=torch.bool, device=self.device)
        forget_mask[self.temp_node] = True
        return forget_mask


    def prepare_retain_data(self):
        retain_mask = torch.ones(self.data.num_nodes, dtype=torch.bool, device=self.device)
        retain_mask[self.temp_node] = False
        retain_mask = retain_mask & self.data.train_mask
        return retain_mask


    def fim_unlearning(self):
        start_time = time.time()

        parameters = {
            "lower_bound": self.args.get('fim_lower_bound', 1.0),
            "exponent": self.args.get('fim_exponent', 1.0),
            "dampening_constant": self.args.get('fim_dampening_constant', 1.0),
            "selection_weighting": self.args.get('fim_selection_weighting', 10.0)
        }

        fim_unlearner = FIMUnlearner(self.target_model.model, parameters, self.device)

        forget_mask = self.prepare_forget_data()
        retain_mask = self.prepare_retain_data()

        sample_importance = fim_unlearner.calc_importance(forget_mask, self.data)
        original_importance = fim_unlearner.calc_importance(retain_mask, self.data)

        fim_unlearner.modify_weight(original_importance, sample_importance)

        unlearn_time = time.time() - start_time

        test_f1, test_acc, test_sp, test_eo = self.evaluate_unlearning()

        self.save_unlearned_model_for_attack()

        return unlearn_time, test_f1, test_acc, test_sp, test_eo


    def evaluate_unlearning(self):
        self.target_model.model.eval()
        with torch.no_grad():
            out = self.target_model.model(self.data.x_unlearn, self.data.edge_index_unlearn)
            if self.args['target_model'] in ['FGCN', 'FGAT']:
                prob = torch.sigmoid(out).cpu().detach().numpy().reshape(-1)
                sel_idx = getattr(self, 'threshold_select_indices', None)
                if sel_idx is None or len(sel_idx) == 0:
                    sel_idx = np.where(self.data.train_mask.cpu().numpy())[0]
                y_all = self.data.y.cpu().numpy()
                best_thr, best_acc = 0.5, -1.0
                for thr in np.linspace(0.48, 0.52, 21):
                    pred_sel = (prob[sel_idx] >= thr).astype(np.int64)
                    acc_sel = accuracy_score(y_all[sel_idx], pred_sel)
                    if acc_sel > best_acc:
                        best_acc, best_thr = acc_sel, thr
                y_pred = (prob >= best_thr).astype(np.int64)
            else:
                y_hat = torch.log_softmax(out, dim=1).cpu().detach().numpy()
                y_pred = np.argmax(y_hat, axis=1)
            y = self.data.y.cpu().numpy()

            mask = self.data.test_mask.cpu().numpy()
            test_f1 = f1_score(y[mask], y_pred[mask], average="macro")
            test_acc = accuracy_score(y[mask], y_pred[mask])

            sens = self.data.sens.cpu().numpy()
            test_sp, test_eo = self._fair_metric(y[mask], sens[mask], y_pred[mask])

        return test_f1, test_acc, test_sp, test_eo


    def save_unlearned_model_for_attack(self):
        with torch.no_grad():
            self.target_model.model.eval()
            test_out = self.target_model.model(self.data.x_unlearn, self.data.edge_index_unlearn)
            if self.args['target_model'] in ['FGCN', 'FGAT']:
                predicted_prob = torch.sigmoid(test_out).cpu()
            else:
                predicted_prob = F.softmax(test_out, dim=1).cpu()

            state = {
                'model_state_dict': self.target_model.model.state_dict(),
                'train_indices': self.data.train_mask,
                'test_indices': self.data.test_mask,
                'removed_nodes': self.temp_node,
                'predicted_prob': predicted_prob,
                'args': self.args,
            }

            exp_marker = [self.args['dataset_name'], self.args['unlearn_task'],
                          str(self.args['unlearn_ratio']), self.args['target_model']]
            exp_marker_string = "_".join(exp_marker)

            os.makedirs('attack_materials', exist_ok=True)

            file_path = f'attack_materials/FairGU_{exp_marker_string}.pth'
            torch.save(state, file_path)
            self.logger.info(f"Model for attack saved to {file_path}")