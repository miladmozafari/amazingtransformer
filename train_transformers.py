import torch
import numpy as np
from Solver import TransformerSolver
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from maze_dataset import MazeDatasetSequential
from experiment_conf import MAZE, DATASET, TRANSFORMERS, GLOBAL

def train(net, train_loader, epoch_num, loss_fn, optim):
    net.train()
    total = 0
    correct = 0

    for in_seq, out_seq, _, _ in train_loader:
        x = in_seq.to(DEVICE)
        y_in  = out_seq[:,:-1,:].to(DEVICE)
        y_out = out_seq[:,1: ,0].type(torch.LongTensor).to(DEVICE)

        optim.zero_grad()

        pred = net(x, y_in)
        loss = loss_fn(pred.reshape(-1,5),y_out.reshape(-1))

        loss.backward()
        optim.step()

        _, predicted = torch.max(pred.reshape(-1,5), 1)
        total   += y_out.shape[0]*y_out.shape[1]
        correct += (predicted == y_out.reshape(-1)).sum().item()
        acc      = 100 * correct / total

        print(f"Epoch: {epoch_num:03d}\t loss: {loss.item():5.5f}\t acc: {acc:6.3f} %", end="\r")
    print()
    return loss, acc

@torch.no_grad()
def evaluate(net, eval_loader, epoch_num):
    net.eval()
    total = 0
    correct = 0

    for in_seq, out_seq, _, _ in eval_loader:
        x = in_seq.to(DEVICE)
        y_in  = out_seq[:,:-1,:].to(DEVICE)
        y_out = out_seq[:,1: ,0].type(torch.LongTensor).to(DEVICE)

        pred = net(x, y_in)
        _, predicted = torch.max(pred.reshape(-1,5), 1)
        total   += y_out.shape[0]*y_out.shape[1]
        correct += (predicted == y_out.reshape(-1)).sum().item()

    acc = 100 * correct / total
    print(f"Epoch: {epoch_num:03d}\t loss: {np.nan:5.5f}\t acc: {acc:6.3f} %")

    return acc

GRID_SIZE       = MAZE['GRID_SIZE']
MAX_PATH_LENGTH = MAZE['MAX_PATH_LENGTH']
SHORTEST_PATH   = MAZE['SHORTEST_PATH']

NUM_TRAIN       = DATASET['NUM_TRAIN']
NUM_EVAL        = DATASET['NUM_EVAL']

BATCH_SIZE      = TRANSFORMERS['BATCH_SIZE'] 
MAX_EPOCH       = TRANSFORMERS['MAX_EPOCH']
D_MODEL         = TRANSFORMERS['D_MODEL']
N_HEAD          = TRANSFORMERS['N_HEAD']
N_ENCODER       = TRANSFORMERS['N_ENCODER']
N_DECODER       = TRANSFORMERS['N_DECODER']
FC_DIM          = TRANSFORMERS['FC_DIM']
LR              = TRANSFORMERS['LR']

DEVICE          = GLOBAL['DEVICE']
SEED            = GLOBAL['SEED']

if __name__ == "__main__":
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)

    train_dataset = MazeDatasetSequential(NUM_TRAIN, GRID_SIZE, MAX_PATH_LENGTH, D_MODEL, SHORTEST_PATH)
    eval_dataset  = MazeDatasetSequential(NUM_EVAL,  GRID_SIZE, MAX_PATH_LENGTH, D_MODEL, SHORTEST_PATH)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=8)
    eval_loader  = DataLoader(eval_dataset,  batch_size=BATCH_SIZE, shuffle=False, num_workers=8)

    solver = TransformerSolver(d_model=D_MODEL, nhead=N_HEAD, num_encoder_layers=N_ENCODER, num_decoder_layers=N_DECODER, dim_feedforward=FC_DIM, batch_first=True)
    pytorch_total_params = sum(p.numel() for p in solver.parameters() if p.requires_grad)
    print(f"Number of parameters: {pytorch_total_params}")
    # solver.load_state_dict(torch.load(f"/home/milad/projects/Transrouter/transformers/transformer_solver_best_no_relu_input_fc_v6.pt", map_location=DEVICE))
    solver = solver.to(DEVICE)

    optim = Adam(solver.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()

    best_acc   = 0
    for epoch in range(MAX_EPOCH):
        train_loss, train_acc = train(solver, train_loader, epoch, loss_fn, optim)
        eval_acc = evaluate(solver, eval_loader, epoch)

        if eval_acc >= best_acc:
            if epoch > 100:
                torch.save(solver.state_dict(), f"./transformer_solver_best.pt")
            best_acc = eval_acc

