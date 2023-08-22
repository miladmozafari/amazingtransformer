import torch
import numpy as np
from Solver import CNNSolver
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from maze_dataset import MazeDatasetSnapshots
from experiment_conf import MAZE, DATASET, CNNS, GLOBAL

def train(net, train_loader, epoch_num, loss_fn, optim):
    net.train()
    total = 0
    correct = 0

    for x, y, _, _ in train_loader:
        x = x.unsqueeze(1).to(DEVICE)
        y = y.to(DEVICE)

        optim.zero_grad()

        pred = net(x)
        loss = loss_fn(pred,y)

        loss.backward()
        optim.step()

        _, predicted = torch.max(pred, 1)
        total   += y.shape[0]
        correct += (predicted == y).sum().item()
        acc      = 100 * correct / total

        print(f"Epoch: {epoch_num:03d}\t loss: {loss.item():5.5f}\t acc: {acc:6.3f} %", end="\r")
    print()
    return loss, acc

@torch.no_grad()
def evaluate(net, eval_loader, epoch_num):
    net.eval()
    total = 0
    correct = 0

    for x, y, _, _ in eval_loader:
        x = x.unsqueeze(1).to(DEVICE)
        y = y.to(DEVICE)

        pred = net(x)
        _, predicted = torch.max(pred, 1)
        total   += y.shape[0]
        correct += (predicted == y).sum().item()

    acc = 100 * correct / total
    print(f"Epoch: {epoch_num:03d}\t loss: {np.nan:5.5f}\t acc: {acc:6.3f} %")

    return acc

GRID_SIZE       = MAZE['GRID_SIZE']
MAX_PATH_LENGTH = MAZE['MAX_PATH_LENGTH']
SHORTEST_PATH   = MAZE['SHORTEST_PATH']

NUM_TRAIN       = DATASET['NUM_TRAIN']
NUM_EVAL        = DATASET['NUM_EVAL']

BATCH_SIZE      = CNNS['BATCH_SIZE']
MAX_EPOCH       = CNNS['MAX_EPOCH']
D_MODEL         = CNNS['D_MODEL']
FC_DIM          = CNNS['FC_DIM']
LR              = CNNS['LR']
WD              = CNNS['WD']

DEVICE          = GLOBAL['DEVICE']
SEED            = GLOBAL['SEED']

if __name__ == "__main__":
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)

    train_dataset = MazeDatasetSnapshots(NUM_TRAIN, GRID_SIZE, MAX_PATH_LENGTH, SHORTEST_PATH)
    eval_dataset  = MazeDatasetSnapshots(NUM_EVAL,  GRID_SIZE, MAX_PATH_LENGTH, SHORTEST_PATH)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=8)
    eval_loader  = DataLoader(eval_dataset,  batch_size=BATCH_SIZE, shuffle=False, num_workers=8)

    solver = CNNSolver(d_model=D_MODEL, dim_feedforward=FC_DIM).double()
    pytorch_total_params = sum(p.numel() for p in solver.parameters() if p.requires_grad)
    print(f"Number of parameters: {pytorch_total_params}")
    solver = solver.to(DEVICE)

    optim = Adam(solver.parameters(), lr=LR, weight_decay=WD)
    loss_fn = nn.CrossEntropyLoss()

    best_acc   = 0
    for epoch in range(MAX_EPOCH):
        train_loss, train_acc = train(solver, train_loader, epoch, loss_fn, optim)
        eval_acc = evaluate(solver, eval_loader, epoch)

        if eval_acc >= best_acc:
            if epoch > 100:
                torch.save(solver.state_dict(), f"./cnn_solver_best.pt")
            best_acc = eval_acc

