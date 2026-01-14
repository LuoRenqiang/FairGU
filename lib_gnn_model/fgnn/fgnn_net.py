import torch
import torch.nn as nn
import torch.nn.functional as F
from lib_gnn_model.gcn.gcn_net_batch import GCNNet
from lib_gnn_model.gat.gat_net_batch import GATNet
import os

class FGNN(nn.Module):
    def __init__(self, num_feats, num_classes, args):
        super(FGNN, self).__init__()

        self.args = args
        nhid = args.get('num_hidden', 128)
        dropout = args.get('dropout', 0.5)

        self.num_classes = num_classes
        # Sensitive attribute estimator
        self.estimator = GCNNet(num_feats, 1, num_layers=2)
        if args['target_model'] == 'FGCN':
            self.GNN = GCNNet(num_feats, nhid, num_layers=2)
        elif args['target_model'] == 'FGAT':
            self.GNN = GATNet(num_feats, nhid, num_layers=2, dropout=dropout)

        self.classifier = nn.Linear(nhid, 1)
        self.adv = nn.Linear(nhid, 1)

        self.criterion = nn.BCEWithLogitsLoss()
        self.estimator_pretrained_loaded = False
        self.load_pretrained_estimator(args['dataset_name'], args['sens_number'])
        self.return_all = True
        # Optimizer will be created during training

    def forward(self, x, edge_index, return_all=None):
        if return_all is None:
            return_all = self.return_all

        s = self.estimator(x, edge_index)
        z = self.GNN(x, edge_index)
        y = self.classifier(z)

        if return_all:
            return y, s
        else:
            return y

    def load_pretrained_estimator(self, dataset_name, sens_number):
        try:
            checkpoint_path = f"exp/checkpoint/GCN_sens_{dataset_name}_ns_{sens_number}"
            if os.path.exists(checkpoint_path):
                self.estimator.load_state_dict(
                    torch.load(checkpoint_path, map_location='cpu')
                )
                print(f"Loaded pretrained estimator from {checkpoint_path}")
                self.estimator_pretrained_loaded = True
            else:
                print(f"Pretrained estimator not found at {checkpoint_path}")
        except Exception as e:
            print(f"Failed to load pretrained estimator: {e}")

    def reset_parameters(self):
        if hasattr(self.estimator, 'reset_parameters'):
            if not getattr(self, 'estimator_pretrained_loaded', False):
                self.estimator.reset_parameters()

        if hasattr(self.GNN, 'reset_parameters'):
            self.GNN.reset_parameters()

        if hasattr(self.classifier, 'reset_parameters'):
            self.classifier.reset_parameters()
        else:
            nn.init.xavier_uniform_(self.classifier.weight)
            if self.classifier.bias is not None:
                nn.init.zeros_(self.classifier.bias)

        if hasattr(self.adv, 'reset_parameters'):
            self.adv.reset_parameters()
        else:
            nn.init.xavier_uniform_(self.adv.weight)
            if self.adv.bias is not None:
                nn.init.zeros_(self.adv.bias)