import torch
import torch.nn as nn
from copy import deepcopy

class FIMUnlearner:
    def __init__(self, model, parameters, device):
        self.model = model
        self.parameters = parameters
        self.device = device
        self.init_weights = deepcopy(model.state_dict())

    def calc_importance(self, node_mask, data):
        self.model.eval()
        importance = {}

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.importance = torch.zeros_like(param.data, dtype=torch.float32)
                importance[name] = param.importance

        output = self.model(data.x, data.edge_index)
        is_binary = (output.dim() == 2 and output.size(1) == 1)

        if data.y.dim() == 1:
            if is_binary:
                criterion = nn.BCEWithLogitsLoss()
            else:
                criterion = nn.CrossEntropyLoss()

            for node_idx in torch.where(node_mask)[0]:
                if node_idx < len(data.y):
                    if is_binary:
                        target = data.y[node_idx:node_idx + 1].float().unsqueeze(1)
                        loss = criterion(
                            output[node_idx:node_idx + 1],
                            target
                        )
                    else:
                        loss = criterion(
                            output[node_idx:node_idx + 1],
                            data.y[node_idx:node_idx + 1]
                        )

                    self.model.zero_grad()
                    loss.backward(retain_graph=True)

                    for name, param in self.model.named_parameters():
                        if param.grad is not None:
                            importance[name] += param.grad.data.clone().pow(2)
        else:
            criterion = nn.BCEWithLogitsLoss()
            for node_idx in torch.where(node_mask)[0]:
                if node_idx < len(data.y):
                    loss = criterion(output[node_idx:node_idx + 1], data.y[node_idx:node_idx + 1].float())
                    self.model.zero_grad()
                    loss.backward(retain_graph=True)

                    for name, param in self.model.named_parameters():
                        if param.grad is not None:
                            importance[name] += param.grad.data.clone().pow(2)

        num_nodes = node_mask.sum().item()
        if num_nodes > 0:
            for name in importance:
                importance[name] /= num_nodes

        return importance

    def modify_weight(self, original_importance, sample_importance):
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in original_importance and name in sample_importance:
                    oimp = original_importance[name]
                    fimp = sample_importance[name]

                    oimp_norm = oimp.mul(self.parameters["selection_weighting"])
                    locations = torch.where(fimp > oimp_norm)

                    weight = ((oimp.mul(self.parameters["dampening_constant"]) / fimp)).pow(
                        self.parameters["exponent"]
                    )
                    update = weight[locations]

                    min_locs = torch.where(update > self.parameters["lower_bound"])
                    update[min_locs] = self.parameters["lower_bound"]

                    # Apply update
                    param.data[locations] *= update