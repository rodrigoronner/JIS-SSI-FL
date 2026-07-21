import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
import copy

class MimicDataset(Dataset):
    """
    Custom PyTorch Dataset wrapper for the processed MIMIC-IV data.
    """
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        # Unsqueeze adds a dimension to match binary output shape (N, 1)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
        
    def __len__(self): 
        return len(self.y)
        
    def __getitem__(self, idx): 
        return self.X[idx], self.y[idx]

class MLP(nn.Module):
    """
    Multi-Layer Perceptron (MLP) architecture used for the mortality prediction task.
    Structure: Input -> Linear(64) -> ReLU -> Dropout(0.2) -> Linear(32) -> ReLU -> Linear(1) -> Sigmoid
    """
    def __init__(self, input_dim):
        super(MLP, self).__init__()
        self.layer1 = nn.Linear(input_dim, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        self.layer2 = nn.Linear(64, 32)
        self.output = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.dropout(x)
        x = self.relu(self.layer2(x))
        x = self.sigmoid(self.output(x))
        return x

def train_client_fedprox(net, X_client, y_client, args, global_net):
    """
    Executes the local training round for a specific client using the FedProx algorithm.

    FedProx addresses data heterogeneity (Non-IID) by adding a proximal term to the
    loss function, limiting the divergence between the local model and the global model.

    Args:
        net (nn.Module): The local model copy to be trained.
        X_client, y_client (np.ndarray): This client's local training fold
            (already Dirichlet-partitioned and SMOTETomek-balanced upstream
            in data_loader.load_and_process_mimic — never the raw global set).
        args (dict): Hyperparameters (learning rate, batch size, mu, epochs).
        global_net (nn.Module): The frozen global model state for reference.

    Returns:
        state_dict: The updated weights of the local model.
        avg_loss: The average training loss for this round.
    """
    net.train()

    train_loader = DataLoader(
        MimicDataset(X_client, y_client), batch_size=args["bs"], shuffle=True
    )

    # SGD with momentum + weight decay, per the paper's optimization setup.
    optimizer = optim.SGD(
        net.parameters(),
        lr=args["lr"],
        momentum=args.get("momentum", 0.9),
        weight_decay=args.get("weight_decay", 1e-5),
    )
    criterion = nn.BCELoss()  # Binary Cross Entropy for mortality prediction

    epoch_loss = []

    for epoch in range(args["local_ep"]):
        batch_loss = []
        for batch_idx, (features, labels) in enumerate(train_loader):
            net.zero_grad()
            log_probs = net(features)

            # 1. Calculate Empirical Loss (Standard ERM)
            empirical_loss = criterion(log_probs, labels)

            # 2. Calculate FedProx Proximal Term
            # Formula: (mu / 2) * ||w - w_t||^2 (squared L2 norm)
            # This penalizes local weights (w) from drifting too far from global weights (w_t)
            proximal_term = 0.0
            for w, w_t in zip(net.parameters(), global_net.parameters()):
                proximal_term += (w - w_t).norm(2) ** 2

            # 3. Total Loss
            loss = empirical_loss + (args["mu"] / 2) * proximal_term

            loss.backward()
            optimizer.step()
            batch_loss.append(loss.item())

        epoch_loss.append(sum(batch_loss) / len(batch_loss))

    return net.state_dict(), sum(epoch_loss) / len(epoch_loss)
