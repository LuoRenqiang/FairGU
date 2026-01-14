# train_sens_pyg.py
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
import argparse
import sys
import pandas as pd
import scipy.sparse as sp

# Add project path
sys.path.append('.')

from lib_gnn_model.gcn.gcn_net_batch import GCNNet
from lib_dataset.data_store import DataStore
from torch_geometric.utils import from_scipy_sparse_matrix


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='pokec_z',
                        choices=['pokec_z', 'pokec_n', 'credit', 'income'])
    parser.add_argument('--hidden', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--epochs', type=int, default=2000)
    parser.add_argument('--sens_number', type=int, default=200)
    parser.add_argument('--seed', type=int, default=41)
    parser.add_argument('--patience', type=int, default=500)

    args = parser.parse_args()

    # Set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # Create parameter dictionary to match DataStore's expected format
    data_args = {
        'dataset_name': args.dataset,
        'target_model': 'GCN',  # Arbitrary value, only need data loading
        'test_ratio': 0.2,
        'is_split': True,
        'unlearn_task': 'node',
        'unlearn_ratio': 0.05
    }

    # Load data
    print(f"Loading dataset: {args.dataset}")
    data_store = DataStore(data_args)
    data = data_store.load_raw_data()

    # Get features and labels
    features = data.x
    sens = data.sens

    print(f"Dataset loaded. Features shape: {features.shape}")
    print(f"Sensitive attribute shape: {sens.shape}")

    # Reuse main project's train/test split; split validation set from training set
    try:
        train_indices, test_indices = data_store.load_train_test_split()
        print("Loaded existing train/test split from DataStore.")
    except Exception as e:
        print(f"Failed to load train/test split, will create a new split here. Error: {e}")
        # Fallback: if not exists, create and save for future reuse
        num_nodes = features.shape[0]
        all_indices = np.arange(num_nodes)
        from sklearn.model_selection import train_test_split
        train_indices, test_indices = train_test_split(all_indices, test_size=0.2, random_state=100)
        data_store.save_train_test_split(train_indices, test_indices)

    # Only use samples with sensitive attribute labels
    valid_indices = np.where(sens.cpu().numpy() >= 0)[0]
    print(f"Valid samples with sensitive attributes: {len(valid_indices)}")

    # Intersect with main split to get train/test for sensitive attribute estimator
    inter_train = np.intersect1d(train_indices, valid_indices)
    inter_test = np.intersect1d(test_indices, valid_indices)

    # Split validation set from training set (10%)
    rng = np.random.default_rng(42)
    rng.shuffle(inter_train)
    num_val = max(1, int(0.1 * len(inter_train)))
    idx_val = inter_train[:num_val]
    idx_train = inter_train[num_val:]

    if args.sens_number and len(idx_train) > args.sens_number:
        idx_train = idx_train[:args.sens_number]
    idx_test = inter_test

    # Convert to PyTorch tensors
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)
    sens = sens.long()

    # Create sensitive attribute estimator model (PyG version)
    # Note: GCNNet implementation uses 2 layers with intermediate layer size fixed at 32
    model = GCNNet(
        in_channels=features.shape[1],
        out_channels=1,  # Binary classification for sensitive attribute
        num_layers=2,
        hidden_channels=args.hidden
    )

    # Check if GPU is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Move model and data to device
    model = model.to(device)
    features = features.to(device)
    sens = sens.to(device)
    idx_train = idx_train.to(device)
    idx_val = idx_val.to(device)
    idx_test = idx_test.to(device)

    # Create optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    # Construct imbalanced pos_weight (based on training set labels)
    with torch.no_grad():
        train_labels = sens[idx_train].float()
        pos = (train_labels == 1).sum().item()
        neg = (train_labels == 0).sum().item()
        pos_weight = torch.tensor([neg / max(pos, 1)], device=device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Train model
    print("Start training sensitive attribute estimator...")
    best_acc = 0.0
    best_test = 0.0
    best_epoch = -1
    patience_counter = 0

    for epoch in range(args.epochs + 1):
        # Training phase
        model.train()
        optimizer.zero_grad()

        output = model(features, data.edge_index.to(device))
        loss_train = criterion(
            output[idx_train].squeeze(),
            sens[idx_train].float()
        )

        # Calculate training accuracy
        train_preds = (torch.sigmoid(output[idx_train].squeeze()) > 0.5).long()
        acc_train = (train_preds == sens[idx_train]).float().mean()

        loss_train.backward()
        optimizer.step()

        # Validation phase (every 100 epochs)
        if epoch % 100 == 0:
            model.eval()
            with torch.no_grad():
                output = model(features, data.edge_index.to(device))

                # Validation set accuracy
                val_preds = (torch.sigmoid(output[idx_val].squeeze()) > 0.5).long()
                acc_val = (val_preds == sens[idx_val]).float().mean()

                # Test set accuracy
                test_preds = (torch.sigmoid(output[idx_test].squeeze()) > 0.5).long()
                acc_test = (test_preds == sens[idx_test]).float().mean()

                print(f"Epoch [{epoch}] Train acc: {acc_train:.4f}, "
                      f"Val acc: {acc_val:.4f}, Test acc: {acc_test:.4f}")

                # Save best model
                if acc_val > best_acc:
                    best_acc = acc_val
                    best_test = acc_test
                    best_epoch = epoch
                    patience_counter = 0

                    # Create checkpoint directory
                    os.makedirs('checkpoint', exist_ok=True)

                    # Save model weights
                    checkpoint_path = f"exp/checkpoint/GCN_sens_{args.dataset}_ns_{args.sens_number}"
                    torch.save(model.state_dict(), checkpoint_path)
                    print(f"Saved best model to {checkpoint_path}")
                else:
                    patience_counter += 10

                # Early stopping check
                if patience_counter >= args.patience:
                    print(f"Early stopping at epoch {epoch}. Best val acc: {best_acc:.4f} (epoch {best_epoch})")
                    break

    print(f"Training finished!")
    print(f"Best validation accuracy: {best_acc:.4f}")
    print(f"Test accuracy with best validation model: {best_test:.4f}")


if __name__ == '__main__':
    main()
