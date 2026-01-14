import logging
import os
import pickle
import shutil

import numpy as np
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
import config

import pandas as pd
import scipy.sparse as sp
from torch_geometric.utils import subgraph
from torch_geometric.data import Data
import torch

from sklearn.preprocessing import MinMaxScaler

import csv

class DataStore:
    def __init__(self, args):
        self.logger = logging.getLogger('data_store')
        self.args = args

        self.dataset_name = self.args['dataset_name']
        self.num_features = {
            "cora": 1433,
            "citeseer": 3703,
            "credit": 13,
            "pokec_z": 276,
            "pokec_n": 265,
            "income": 13,
        }
        self.target_model = self.args['target_model']

        self.determine_data_path()

    def determine_data_path(self):
        embedding_name = '_'.join(('embedding', self.args['unlearn_task'], str(self.args['unlearn_ratio'])))

        processed_data_prefix = config.PROCESSED_DATA_PATH + self.dataset_name + "/"
        self.train_test_split_file = processed_data_prefix + "train_test_split" + str(self.args['test_ratio'])
        self.train_data_file = processed_data_prefix + "train_data"
        self.train_graph_file = processed_data_prefix + "train_graph"
        self.embedding_file = processed_data_prefix + embedding_name

        self.unlearned_file = processed_data_prefix + '_'.join(
            ('unlearned', self.args['unlearn_task'], str(self.args['unlearn_ratio'])))

        dir_lists = [s + self.dataset_name for s in [config.PROCESSED_DATA_PATH,
                                                     config.MODEL_PATH]]
        for dir in dir_lists:
            self._check_and_create_dirs(dir)

    def _check_and_create_dirs(self, folder):
        if not os.path.exists(folder):
            try:
                # self.logger.info("checking directory %s", folder)
                os.makedirs(folder, exist_ok=True)
                # self.logger.info("new directory %s created", folder)
            except OSError as error:
                # self.logger.info("deleting old and creating new empty %s", folder)
                shutil.rmtree(folder)
                os.mkdir(folder)
                # self.logger.info("new empty directory %s created", folder)
        else:
            # self.logger.info("folder %s exists, do not need to create again.", folder)
            pass

    def load_raw_data(self):
        # self.logger.info('loading raw data')

        if self.dataset_name in ["cora", "citeseer"]:
            dataset = Planetoid(config.RAW_DATA_PATH, self.dataset_name, transform=T.NormalizeFeatures())
            labels = np.unique(dataset.data.y.numpy())
            data = dataset[0]

        elif self.dataset_name == "pokec_z":
            data = self._load_pokec_z()
        elif self.dataset_name == "pokec_n":
            data = self._load_pokec_n()
        elif self.dataset_name == "credit":
            data = self._load_credit_data()
        elif self.dataset_name == "income":
            data = self._load_income()
        else:
            raise Exception('unsupported dataset')

        data.name = self.dataset_name
        # data.num_classes = dataset.num_classes

        return data

    def save_train_data(self, train_data):
        # self.logger.info('saving train data')
        pickle.dump(train_data, open(self.train_data_file, 'wb'))

    def load_train_data(self):
        # self.logger.info('loading train data')
        return pickle.load(open(self.train_data_file, 'rb'))

    def save_train_graph(self, train_data):
        # self.logger.info('saving train graph')
        pickle.dump(train_data, open(self.train_graph_file, 'wb'))

    def load_train_graph(self):
        # self.logger.info('loading train graph')
        return pickle.load(open(self.train_graph_file, 'rb'))

    def save_train_test_split(self, train_indices, test_indices):
        # self.logger.info('saving train test split data')
        pickle.dump((train_indices, test_indices), open(self.train_test_split_file, 'wb'))

    def load_train_test_split(self):
        # self.logger.info('loading train test split data')
        return pickle.load(open(self.train_test_split_file, 'rb'))

    def save_embeddings(self, embeddings):
        # self.logger.info('saving embedding data')
        pickle.dump(embeddings, open(self.embedding_file, 'wb'))

    def load_embeddings(self):
        # self.logger.info('loading embedding data')
        return pickle.load(open(self.embedding_file, 'rb'))

    def load_unlearned_data(self, suffix):
        file_path = '_'.join((self.unlearned_file, suffix))
        # self.logger.info('loading unlearned data from %s' % file_path)
        return pickle.load(open(file_path, 'rb'))

    def save_unlearned_data(self, data, suffix):
        # self.logger.info('saving unlearned data %s' % suffix)
        pickle.dump(data, open('_'.join((self.unlearned_file, suffix)), 'wb'))

    def _load_credit_data(self):
        idx_features_labels = pd.read_csv(os.path.join(config.RAW_DATA_PATH, 'credit/credit.csv'))
        edges_unordered = np.genfromtxt(os.path.join(config.RAW_DATA_PATH, 'credit/credit_edges.txt')).astype('int')

        sens_attr = "Age"
        predict_attr = "NoDefaultNextMonth"
        header = list(idx_features_labels.columns)
        header.remove('Single')
        header.remove(predict_attr)
        header.remove(sens_attr)

        sens = torch.FloatTensor(idx_features_labels[sens_attr].values.astype(int))

        features = idx_features_labels[header].values

        scaler = MinMaxScaler(feature_range=(-1, 1))
        features = scaler.fit_transform(features)

        features = torch.FloatTensor(features)


        labels = idx_features_labels[predict_attr].values
        labels = torch.LongTensor(labels)
        labels[labels > 1] = 1

        idx = np.arange(features.shape[0])
        idx_map = {j: i for i, j in enumerate(idx)}
        edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=int).reshape(edges_unordered.shape)
        edge_index = torch.tensor(edges.T, dtype=torch.long)

        data = Data(
            x=features,
            edge_index=edge_index,
            y=labels,
            sens=sens
        )
        data.num_classes = 2
        return data

    def _load_pokec_z(self):
        datapath = os.path.join(config.RAW_DATA_PATH, "pokec_z/")
        if not os.path.exists(datapath):
            raise FileNotFoundError(f"Dataset directory not found: {datapath}")

        edges_file = os.path.join(datapath, 'region_job_relationship.txt')
        if not os.path.exists(edges_file):
            raise FileNotFoundError(f"File not found: {edges_file}")

        edges_unordered = np.genfromtxt(edges_file).astype('int')
        idx_features_labels = pd.read_csv(os.path.join(datapath, 'region_job.csv'))

        predict_attr = 'I_am_working_in_field'
        sens_attr = 'region'

        header = list(idx_features_labels.columns)
        header.remove(predict_attr)
        header.remove(sens_attr)
        header.remove("user_id")


        feature = idx_features_labels[header]

        scaler = MinMaxScaler(feature_range=(-1, 1))
        feature = scaler.fit_transform(feature)

        labels = idx_features_labels[predict_attr].values

        idx = np.array(idx_features_labels["user_id"], dtype=int)
        idx_map = {j: i for i, j in enumerate(idx)}
        edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=int).reshape(edges_unordered.shape)

        adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                            shape=(labels.shape[0], labels.shape[0]), dtype=np.float32)
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        if self.args.get('self_loop', True):
            adj = adj + sp.eye(adj.shape[0])
        adj = adj.tocoo()

        sens = idx_features_labels[sens_attr].values.astype(int)
        if sens_attr == 'AGE':
            sens = (sens >= 25)
        sens = torch.FloatTensor(sens)

        feature = np.array(feature)
        feature = torch.FloatTensor(feature)

        labels = torch.LongTensor(labels)
        labels[labels > 1] = 1

        edge_index = torch.tensor(np.vstack([adj.row, adj.col]), dtype=torch.long)

        valid_mask = (labels == 0) | (labels == 1)
        valid_indices = torch.where(valid_mask)[0]

        feature = feature[valid_mask]
        labels = labels[valid_mask]
        sens = sens[valid_mask]

        edge_index, _ = subgraph(
            subset=valid_indices,
            edge_index=edge_index,
            relabel_nodes=True
        )

        data = Data(x=feature, edge_index=edge_index, y=labels, sens=sens)

        data.num_classes = 2

        return data

    def _load_pokec_n(self):
        datapath = os.path.join(config.RAW_DATA_PATH, "pokec_n/")
        if not os.path.exists(datapath):
            raise FileNotFoundError(f"Dataset directory not found: {datapath}")

        edges_file = os.path.join(datapath, 'region_job_2_relationship.txt')
        if not os.path.exists(edges_file):
            raise FileNotFoundError(f"File not found: {edges_file}")

        edges_unordered = np.genfromtxt(edges_file).astype('int')
        idx_features_labels = pd.read_csv(os.path.join(datapath, 'region_job_2.csv'))

        predict_attr = 'I_am_working_in_field'
        sens_attr = 'region'

        header = list(idx_features_labels.columns)
        header.remove(predict_attr)
        header.remove(sens_attr)
        header.remove("user_id")

        feature = idx_features_labels[header]

        scaler = MinMaxScaler(feature_range=(-1, 1))
        feature = scaler.fit_transform(feature)

        labels = idx_features_labels[predict_attr].values

        idx = np.array(idx_features_labels["user_id"], dtype=int)
        idx_map = {j: i for i, j in enumerate(idx)}
        edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=int).reshape(edges_unordered.shape)

        adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                            shape=(labels.shape[0], labels.shape[0]), dtype=np.float32)
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        if self.args.get('self_loop', True):
            adj = adj + sp.eye(adj.shape[0])
        adj = adj.tocoo()

        sens = idx_features_labels[sens_attr].values.astype(int)
        if sens_attr == 'AGE':
            sens = (sens >= 25)
        sens = torch.FloatTensor(sens)

        feature = np.array(feature)
        feature = torch.FloatTensor(feature)

        labels = torch.LongTensor(labels)
        labels[labels > 1] = 1

        edge_index = torch.tensor(np.vstack([adj.row, adj.col]), dtype=torch.long)

        valid_mask = (labels == 0) | (labels == 1)
        valid_indices = torch.where(valid_mask)[0]

        feature = feature[valid_mask]
        labels = labels[valid_mask]
        sens = sens[valid_mask]

        edge_index, _ = subgraph(
            subset=valid_indices,
            edge_index=edge_index,
            relabel_nodes=True
        )

        data = Data(x=feature, edge_index=edge_index, y=labels, sens=sens)

        data.num_classes = 2

        return data

    def _load_income(self):

        datapath = os.path.join(config.RAW_DATA_PATH, "income/")
        if not os.path.exists(datapath):
            raise FileNotFoundError(f"Dataset directory not found: {datapath}")

        data_file = os.path.join(datapath, 'income.csv')
        edges_file = os.path.join(datapath, 'income_edges.txt')

        if not os.path.exists(data_file):
            raise FileNotFoundError(f"File not found: {data_file}")
        if not os.path.exists(edges_file):
            raise FileNotFoundError(f"File not found: {edges_file}")

        edges_unordered = np.genfromtxt(edges_file).astype('int')

        idx_features_labels = pd.read_csv(data_file)

        predict_attr = 'income'
        sens_attr = 'race'

        header = list(idx_features_labels.columns)
        header.remove(predict_attr)
        header.remove(sens_attr)  # Remove sens_attr

        feature = idx_features_labels[header]

        scaler = MinMaxScaler(feature_range=(-1, 1))
        feature = scaler.fit_transform(feature)

        labels = idx_features_labels[predict_attr].values

        idx = np.arange(feature.shape[0])
        idx_map = {j: i for i, j in enumerate(idx)}
        edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=int).reshape(edges_unordered.shape)


        adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                            shape=(labels.shape[0], labels.shape[0]), dtype=np.float32)

        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

        if self.args.get('self_loop', True):
            adj = adj + sp.eye(adj.shape[0])
        adj = adj.tocoo()

        sens = idx_features_labels[sens_attr].values.astype(int)
        sens = torch.FloatTensor(sens)

        feature = np.array(feature)
        feature = torch.FloatTensor(feature)

        labels = torch.LongTensor(labels)
        labels[labels > 1] = 1

        edge_index = torch.tensor(np.vstack([adj.row, adj.col]), dtype=torch.long)

        valid_mask = (labels == 0) | (labels == 1)
        valid_indices = torch.where(valid_mask)[0]

        feature = feature[valid_mask]
        labels = labels[valid_mask]
        sens = sens[valid_mask]

        edge_index, _ = subgraph(
            subset=valid_indices,
            edge_index=edge_index,
            relabel_nodes=True
        )

        data = Data(x=feature, edge_index=edge_index, y=labels, sens=sens)
        data.num_classes = 2

        return data