"""
Create torch dataset for maze
"""

import torch
from torch.utils.data import Dataset
from maze_util import generate_maze_data_sample, CELL_SIZE, GO, MOVE_DOWN, MOVE_LEFT, MOVE_RIGHT, MOVE_STOP, MOVE_UP

class MazeDataset(Dataset):
    r"""
    Base Maze dataset.

    Args:
        `n_sample` (int): Number of samples
        `grid_size` (int): Size of the maze grid
        `max_path_length` (int): Maximum solution length
        `shortest_path` (boolean): If `True`, the solution will be the shortest possible path
    """
    def __init__(self, n_sample:int, grid_size:int, max_path_length:int, shortest_path:bool) -> None:
        super().__init__()
        self.n_sample  = n_sample
        self.grid_size = grid_size
        self.max_path_length = max_path_length

        self.samples = []
        for i in range(self.n_sample):
            grid, moves, route = generate_maze_data_sample(self.grid_size, self.max_path_length, shortest_path)
            self.samples.append({'grid':torch.from_numpy(grid),'moves':moves,'route':route})
    
    def __getitem__(self, index):
        return self.samples[index]['grid'], self.samples[index]['moves'], self.samples[index]['route']
    
    def __len__(self):
        return len(self.samples)
    
    def count_similar_samples(self, dataset):
        cnt = 0
        for i in range(len(dataset)):
            grid, moves, route = dataset[i]
            for j in range(len(self)):
                self_grid, self_moves, self_route = self[j]

                if len(moves) != len(self_moves):
                    continue
                if torch.any(grid != self_grid):
                    continue
                route_m = 1
                for m,self_m in zip(moves, self_moves):
                    if m != self_m:
                        route_m = 0
                        break
                cnt += route_m

        return cnt

class MazeDatasetSnapshots(MazeDataset):
    r"""
    Converts maze samples into step-by-step snapshots (suitable for convnets)
    """
    def __init__(self, n_sample: int, grid_size: int, max_path_length: int, shortest_path: bool) -> None:
        super().__init__(n_sample, grid_size, max_path_length, shortest_path)
        self.snapshots = []
        self.labels = []
        self.start_points = []
        self.end_points = []

        for i in range(super().__len__()):
            grid, moves, route = super().__getitem__(i)
            
            Xs = [grid.clone()]
            self.start_points.append(route[0])
            self.end_points.append(route[-1])
            Ys = []

            for m, (r,c) in zip(moves[1:-1],route[:-1]):
                Ys.append(m)
                self.add_move_to_grid(grid, r, c, m)
                Xs.append(grid.clone())
                self.start_points.append(route[0])
                self.end_points.append(route[-1])
            Ys.append(MOVE_STOP)
            
            self.snapshots.extend(Xs)
            self.labels.extend(Ys)

    @staticmethod
    def add_move_to_grid(grid, r, c, move):
        cs = CELL_SIZE
        
        dr,dc = GO[move]
        nr,nc = r+dr, c+dc
        center  = grid[r*cs+cs//2,c*cs+cs//2].item()
        ncenter = grid[nr*cs+cs//2,nc*cs+cs//2].item()

        if move == MOVE_DOWN:
            grid[r*cs+cs//2:r*cs+cs,c*cs+cs//2] = 4
            grid[nr*cs:nr*cs+cs//2+1,nc*cs+cs//2] = 4
        elif move == MOVE_LEFT:
            grid[r*cs+cs//2,c*cs:c*cs+cs//2+1] = 4
            grid[nr*cs+cs//2,nc*cs+cs//2:nc*cs+cs] = 4
        elif move == MOVE_RIGHT:
            grid[r*cs+cs//2,c*cs+cs//2:c*cs+cs] = 4
            grid[nr*cs+cs//2,nc*cs:nc*cs+cs//2+1] = 4
        elif move == MOVE_UP:
            grid[r*cs:r*cs+cs//2+1,c*cs+cs//2] = 4
            grid[nr*cs+cs//2:nr*cs+cs,nc*cs+cs//2] = 4
        if center > 1:
            grid[r*cs+cs//2,c*cs+cs//2] = center
        if ncenter > 1:
            grid[nr*cs+cs//2,nc*cs+cs//2] = ncenter

    def __getitem__(self, index):
        return self.snapshots[index], self.labels[index], self.start_points[index], self.end_points[index]
    
    def __len__(self):
        return len(self.snapshots)
    
class MazeDatasetSnapshotsTest(MazeDataset):
    r"""
    Converts maze samples into step-by-step snapshots (suitable for convnets during the text phase)
    """
    def __init__(self, n_sample: int, grid_size: int, max_path_length: int, shortest_path: bool) -> None:
        super().__init__(n_sample, grid_size, max_path_length, shortest_path)
        self.snapshots = []
        self.start_points = []
        self.end_points = []

        for i in range(super().__len__()):
            grid, moves, route = super().__getitem__(i)
            
            Xs = [grid.clone()]
            self.start_points.append(route[0])
            self.end_points.append(route[-1])

            for m, (r,c) in zip(moves[1:-1],route[:-1]):
                MazeDatasetSnapshots.add_move_to_grid(grid, r, c, m)
            
            Xs.append(grid.clone())
            self.snapshots.append(Xs)

    def __getitem__(self, index):
        return self.snapshots[index][0], self.snapshots[index][1], self.start_points[index], self.end_points[index]
    
    def __len__(self):
        return len(self.snapshots)

class MazeDatasetSequential(MazeDataset):
    r"""
    Converts maze samples into sequential format (suitable for transformers)

    Args:
        `d_embed` (int): Number of embedding dimensions
    """
    def __init__(self, n_sample: int, grid_size: int, max_path_length: int, d_embed:int, shortest_path: bool) -> None:
        super().__init__(n_sample, grid_size, max_path_length, shortest_path)
        assert d_embed >= CELL_SIZE * CELL_SIZE + 2, f"d_embed should be >= {CELL_SIZE * CELL_SIZE + 2}"
        self.d_embed = d_embed
        self.in_sequences = []
        self.out_sequences = []
        self.start_points = []
        self.end_points = []
        for i in range(super().__len__()):
            in_seq = []
            out_seq = []
            grid, moves, route = super().__getitem__(i)
            cs = CELL_SIZE      # cell size

            for row in range(grid_size):
                for col in range(grid_size):
                    x = torch.zeros((self.d_embed,))
                    x[:cs*cs]  = grid[row*cs:row*cs+cs, col*cs:col*cs+cs].clone().reshape(-1)    # grid cell info
                    x[cs*cs]   = row                                                     # row info
                    x[cs*cs+1] = col                                                     # column info
                    in_seq.append(x)
            self.in_sequences.append(torch.stack(in_seq))
            self.start_points.append(route[0])
            self.end_points.append(route[-1])

            for m in range(self.max_path_length):
                y = torch.zeros((self.d_embed,))
                if m < len(moves):
                    y[0] = moves[m]
                    y[1] = m
                    if m == len(moves) - 1:     # last move is always stop
                        y[2] = route[m-1][0]
                        y[3] = route[m-1][1]
                    else:
                        y[2] = route[m][0]
                        y[3] = route[m][1]
                out_seq.append(y)               # will be zeros after the last move
            self.out_sequences.append(torch.stack(out_seq))

    def sequence_to_grid(self, seq):
        idx = 0
        grid = torch.zeros(self.grid_size*CELL_SIZE,self.grid_size*CELL_SIZE)
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                grid[CELL_SIZE*i:CELL_SIZE*i+CELL_SIZE, CELL_SIZE*j:CELL_SIZE*j+CELL_SIZE] = seq[idx][:CELL_SIZE*CELL_SIZE].clone().reshape(CELL_SIZE,CELL_SIZE)
                idx+=1
        return grid

    def __getitem__(self, index):
        return self.in_sequences[index], self.out_sequences[index], self.start_points[index], self.end_points[index]
    
    def __len__(self):
        return len(self.in_sequences)