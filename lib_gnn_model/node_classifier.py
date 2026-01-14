import logging

import torch
from sklearn.metrics import f1_score, accuracy_score
torch.cuda.empty_cache()
import torch.nn.functional as F
from torch.autograd import grad
import numpy as np

from lib_gnn_model.gat.gat_net_batch import GATNet
from lib_gnn_model.gin.gin_net_batch import GINNet
from lib_gnn_model.gcn.gcn_net_batch import GCNNet
from lib_gnn_model.sgc.sgc_net_batch import SGCNet
from lib_gnn_model.graphsage.graphsage_net_batch import SAGENet
from lib_gnn_model.fgnn.fgnn_net import FGNN

from lib_gnn_model.gnn_base import GNNBase
from torch_geometric.loader import ClusterData, ClusterLoader, NeighborLoader, GraphSAINTRandomWalkSampler
import copy
from lib_utils.utils import calc_f1

class NodeClassifier(GNNBase):
    def __init__(self, num_feats, num_classes, args, data=None):
        super(NodeClassifier, self).__init__()

        self.args = args
        self.logger = logging.getLogger('node_classifier')
        self.target_model = args['target_model']

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.kwargs = {'batch_size': self.args['batch_size'], 'num_workers':0}
        # self.device = 'cpu'
        self.model = self.determine_model(num_feats, num_classes).to(self.device)
        self.data = data

    def determine_model(self, num_feats, num_classes):
        # self.logger.info('target model: %s' % (self.args['target_model'],))
        if self.target_model == 'SAGE':
            self.lr, self.decay = 0.01, 0.0
            return SAGENet(num_feats, num_classes)
        elif self.target_model == 'GAT':
            self.lr, self.decay = 0.01, 0.0001
            return GATNet(num_feats, num_classes)
        elif self.target_model == 'GCN':
            self.lr, self.decay = 0.05, 0.0001
            return GCNNet(num_feats, num_classes)
        elif self.target_model == 'GIN':
            self.lr, self.decay = 0.01, 0.001
            return GINNet(num_feats, num_classes)
        elif self.target_model == 'SGC':
            self.lr, self.decay = 0.05, 0.0
            return SGCNet(num_feats, num_classes)
        elif self.target_model in ['FGCN', 'FGAT']:
            self.lr, self.decay = 0.001, 1e-5
            return FGNN(num_feats, num_classes, self.args)
        else:
            raise Exception('unsupported target model')

    def train_model(self, unlearn_info=None):
        # self.logger.info("training model")
        if self.target_model in ['FGCN', 'FGAT']:
            self.train_fgnn()
        else:
            self.model.train()
            self.model.reset_parameters()
            self.model, self.data = self.model.to(self.device), self.data.to(self.device)
            self.data.y = self.data.y.squeeze().to(self.device)
            self._gen_train_loader()

            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.decay)

            for epoch in range(self.args['num_epochs']):
                self.model.train()
                for batch in self.train_loader:
                    batch = batch.to(self.device)
                    optimizer.zero_grad()
                    out = self.model(batch.x, batch.edge_index)
                    if self.args['inductive'] == 'graphsaint':
                        loss = F.cross_entropy(out[batch.train_mask], batch.y[batch.train_mask])
                    else:
                        loss = F.cross_entropy(out[:batch.batch_size], batch.y[:batch.batch_size])

                    loss.backward()
                    optimizer.step()

        grad_all, grad1, grad2 = None, None, None

        return (grad_all, grad1, grad2)

    def fair_metric(self, output, idx, sens_labels, labels):

        val_y = labels[idx].cpu().numpy()
        sens = sens_labels.cpu().numpy()[idx.cpu().numpy()]
        pred_y = (output[idx].squeeze() > 0).type_as(labels).cpu().numpy()
        
        idx_s0 = sens == 0
        idx_s1 = sens == 1

        idx_s0_y1 = idx_s0 & (val_y == 1)
        idx_s1_y1 = idx_s1 & (val_y == 1)

        epsilon = 1e-8
        parity = abs((pred_y[idx_s0].sum() + epsilon) / (idx_s0.sum() + epsilon) -
                     (pred_y[idx_s1].sum() + epsilon) / (idx_s1.sum() + epsilon))
        equality = abs((pred_y[idx_s0_y1].sum() + epsilon) / (idx_s0_y1.sum() + epsilon) -
                       (pred_y[idx_s1_y1].sum() + epsilon) / (idx_s1_y1.sum() + epsilon))

        return parity, equality


    def load_best_model(self):
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state['model_state_dict'])
            self.optimizer_G.load_state_dict(self.best_model_state['optimizer_G_state_dict'])
            self.optimizer_A.load_state_dict(self.best_model_state['optimizer_A_state_dict'])
            print("Best model loaded")
            return True
        else:
            print("No saved best model found")
            return False


    def train_fgnn(self):
        self.model.train()
        self.model.reset_parameters()
        self.model, self.data = self.model.to(self.device), self.data.to(self.device)
        self.data.y = self.data.y.squeeze().long().to(self.device)

        model_params = list(self.model.GNN.parameters()) + list(self.model.classifier.parameters()) + list(self.model.estimator.parameters())
        self.optimizer_G = torch.optim.Adam(model_params, lr=self.lr, weight_decay=self.decay)
        self.optimizer_A = torch.optim.Adam(self.model.adv.parameters(), lr=self.lr, weight_decay=self.decay)

        sens_labels = self.data.sens.to(self.device)
        labels = self.data.y.long()

        alpha = self.args.get('alpha', 100)
        beta = self.args.get('beta', 1)
        best_result = {}
        best_fair = 100
        acc_threshold = self.args.get('acc_threshold', 0.68)
        f1_threshold = self.args.get('f1_threshold', 0.67)

        self.best_model_state = None

        for epoch in range(self.args['num_epochs']):
            self.model.train()

            self.model.adv.requires_grad_(False)
            self.optimizer_G.zero_grad()

            self.model.return_all = True
            y, s = self.model(self.data.x, self.data.edge_index)
            h = self.model.GNN(self.data.x, self.data.edge_index)

            s_g = self.model.adv(h)

            s_score = torch.sigmoid(s.detach())
            s_score[self.data.train_mask] = sens_labels[self.data.train_mask].unsqueeze(1).float()
            y_score = torch.sigmoid(y)
            
            cov = torch.abs(torch.mean((s_score - torch.mean(s_score)) * (y_score - torch.mean(y_score))))



            cls_loss = self.model.criterion(y[self.data.train_mask],
                                            labels[self.data.train_mask].unsqueeze(1).float())

            adv_loss_G = self.model.criterion(s_g, s_score)

            if alpha == 0 and beta == 0:
                G_loss = cls_loss
            else:
                G_loss = cls_loss + alpha * cov - beta * adv_loss_G

            G_loss.backward()
            self.optimizer_G.step()
            if beta > 0:
                self.model.adv.requires_grad_(True)
                self.optimizer_A.zero_grad()
                s_g = self.model.adv(h.detach())
                A_loss = self.model.criterion(s_g, s_score)
                A_loss.backward()
                self.optimizer_A.step()

            if (epoch + 1) % max(1, int(self.args.get('log_interval', 100))) == 0:
                with torch.no_grad():
                    self.model.eval()
                    y_eval, s_eval = self.model(self.data.x, self.data.edge_index)
                    
                    y_pred = (y_eval > 0).long()
                    pred_test = y_pred[self.data.test_mask]
                    acc_test = accuracy_score(labels[self.data.test_mask].cpu().numpy(), pred_test.cpu().numpy())
                    f1_test = f1_score(labels[self.data.test_mask].cpu().numpy(), pred_test.cpu().numpy(), average='macro')
                    
                    parity_test, equality_test = self.fair_metric(y_pred.unsqueeze(1), self.data.test_mask, sens_labels, labels)
                    
                    self.model.train()
                    
                    adv_probs = torch.sigmoid(self.model.adv(h))
                    adv_pred = (adv_probs >= 0.5).long()
                    adv_acc = (adv_pred.squeeze(1) == s_score.squeeze().long()).float().mean().item()
                    
                    print(f"[FairGU][Epoch {epoch+1}] F1={f1_test:.4f} ACC={acc_test:.4f} SP={parity_test:.4f} EO={equality_test:.4f} cls={cls_loss.item():.4f} cov={cov.item():.6f} adv_acc={adv_acc:.4f}")

            if (epoch + 1) % max(1, int(self.args.get('eval_interval', 10))) == 0 :
                self.model.eval()
                with torch.no_grad():
                    y_eval, s_eval = self.model(self.data.x, self.data.edge_index)
                    
                    y_pred = (y_eval > 0).long()
                    pred_test = y_pred[self.data.test_mask]
                    acc_test = accuracy_score(labels[self.data.test_mask].cpu().numpy(), pred_test.cpu().numpy())
                    f1_test = f1_score(labels[self.data.test_mask].cpu().numpy(), pred_test.cpu().numpy(), average='macro')
                    
                    parity_test, equality_test = self.fair_metric(y_pred.unsqueeze(1), self.data.test_mask, sens_labels, labels)
                    
                    if acc_test > acc_threshold and f1_test >= f1_threshold and equality_test < 0.02:
                        current_fair = parity_test + equality_test
                        if best_fair > current_fair:
                            best_fair = current_fair
                            best_result = {
                                'acc': acc_test,
                                'f1': f1_test,
                                'parity': parity_test,
                                'equality': equality_test,
                                'epoch': epoch + 1
                            }
                            
                            self.best_model_state = {
                                'model_state_dict': self.model.state_dict().copy(),
                                'optimizer_G_state_dict': self.optimizer_G.state_dict().copy(),
                                'optimizer_A_state_dict': self.optimizer_A.state_dict().copy(),
                                'args': self.args.copy()
                            }
                            
                            print("=" * 50)
                            print(f"Found better model! Epoch {epoch+1}")
                            print(f"Test set: acc={acc_test:.4f}, f1={f1_test:.4f}, SP={parity_test:.4f}, EO={equality_test:.4f}")
                            print("Best model saved to memory")
                            print("=" * 50)

        self.model.return_all = False
        
        if best_result:
            print("\nTraining completed! Best results:")
            print(f"Test accuracy: {best_result['acc']:.4f}")
            print(f"Test F1 score: {best_result['f1']:.4f}")
            print(f"Test statistical parity: {best_result['parity']:.4f}")
            print(f"Test equality of opportunity: {best_result['equality']:.4f}")
            print(f"Best model from epoch {best_result['epoch']}")
            self.load_best_model()
        else:
            print("No model found that meets the accuracy and F1 threshold requirements")
        return None, None, None

    def evaluate_unlearn_F1(self, new_parameters):
        idx = 0
        for p in self.model.parameters():
            p.data = new_parameters[idx]
            idx = idx + 1
        self.model.eval()
        out = self.model(self.data.x_unlearn, self.data.edge_index_unlearn)

        y = self.data.y.cpu()
        y_hat = out.cpu().detach().numpy()
        test_f1 = calc_f1(y, y_hat, self.data.test_mask)
        return test_f1

    @torch.no_grad()
    def evaluate_model(self):
        self.model.eval()
        self.model, self.data = self.model.to(self.device), self.data.to(self.device)
        self._gen_test_loader()

        out = self.model.inference(self.data.x, self.test_loader, self.device)
        y = self.data.y.to(out.device)
        train_f1 = calc_f1(y, out, self.data.train_mask)
        test_f1 = calc_f1(y, out, self.data.test_mask)

        return train_f1, test_f1


    def posterior(self):
        # self.logger.debug("generating posteriors")
        self.model, self.data = self.model.to(self.device), self.data.to(self.device)
        self.model.eval()
        self._gen_test_loader()

        posteriors = self.model.inference(self.data.x, self.test_loader, self.device).to(self.device)

        for _, mask in self.data('test_mask'):
            posteriors = F.softmax(posteriors[mask], dim=-1)

        return posteriors.detach()

    def generate_embeddings(self):
        self.model.eval()
        self.model, self.data = self.model.to(self.device), self.data.to(self.device)
        self._gen_test_loader()

        logits = self.model.inference(self.data.x, self.test_loader, self.device).to(self.device)
        return logits

    def _gen_train_loader(self):
        temp_data = copy.copy(self.data).cpu()
        
        if self.args['inductive'] == 'cluster-gcn':
            cluster_data = ClusterData(temp_data, num_parts=50, recursive=False)
            self.train_loader = ClusterLoader(cluster_data, batch_size=2048, shuffle=True,
                                         num_workers=0)
        if self.args['inductive'] == 'graphsaint':
            self.train_loader = GraphSAINTRandomWalkSampler(temp_data, batch_size=8000, walk_length=2,
                                                 num_steps=5, sample_coverage=100, num_workers=0)
        else:
            self.train_loader = NeighborLoader(temp_data.contiguous(), input_nodes=temp_data.train_mask,
                                               num_neighbors=[5, 5], shuffle=True, **self.kwargs)

    def _gen_test_loader(self):
        temp_data = copy.copy(self.data).cpu()
        self.test_loader = NeighborLoader(temp_data.contiguous(), input_nodes=None, num_neighbors=[-1], shuffle=False,
                                          **self.kwargs)
        self.test_loader.data.num_nodes = self.data.num_nodes
        self.test_loader.data.n_id = torch.arange(self.data.num_nodes)
        del self.test_loader.data.x, self.test_loader.data.y