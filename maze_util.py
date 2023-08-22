"""
Utilities to create Maze samples
"""

import numpy as np
import queue

MOVE_STOP  = 0
MOVE_UP    = 1
MOVE_RIGHT = 2
MOVE_DOWN  = 3
MOVE_LEFT  = 4
MOVE_START = 5

CELL_SIZE  = 5

GO = {
    0: ( 0, 0),          # MOVE_STOP 
    1: (-1, 0),          # MOVE_UP   
    2: ( 0, 1),          # MOVE_RIGHT
    3: ( 1, 0),          # MOVE_DOWN 
    4: ( 0,-1),          # MOVE_LEFT 
    5: ( 0, 0),          # MOVE_START
}

MOVE2NAME = {
    0: "STOP",          # MOVE_STOP 
    1: "UP",            # MOVE_UP   
    2: "RIGHT",         # MOVE_RIGHT
    3: "DOWN",          # MOVE_DOWN 
    4: "LEFT",          # MOVE_LEFT 
    5: "START",         # MOVE_START
}

def generate_random_path(grid_size:int, max_path_length:int):
    r"""
    Generates a random path with a given maximum length.

    Args:
        `grid_size` (int): Size of the maze grid. Maze grid is always a square.
        `max_path_length` (int): Maximum length of the randomly generated path.

    Return:
        `random_path` (List[(int,int)]): A list containing row and col indices of each tile that is included in the path from start to end.
        `moves` (List[int]): A list that contains the id of the move that is performed to reach the current tile from the previous tile in the path.
    """
    sr, sc = np.random.randint(0, grid_size), np.random.randint(0, grid_size)   # starting point
    moves = [MOVE_START]
    random_path = []
    
    cr, cc = sr, sc     # current position
    last_move = 0       # moves: 0 (nothing) | 1 (up) | 2 (right) | 3 (down) | 4 (left)
    visited = set()
    visited.add((cr,cc))
    random_path.append((cr,cc))
    max_length = np.random.randint(2, max_path_length-1)
    for i in range(max_length):
        possible_moves = []
        if last_move != MOVE_DOWN and (cr-1) >= 0 and (cr-1,cc) not in visited:
            possible_moves.append(MOVE_UP)
        if last_move != MOVE_LEFT and (cc+1) < grid_size and (cr, cc+1) not in visited:
            possible_moves.append(MOVE_RIGHT)
        if last_move != MOVE_UP and (cr+1) < grid_size and (cr+1, cc) not in visited:
            possible_moves.append(MOVE_DOWN)
        if last_move != MOVE_RIGHT and (cc-1) >= 0 and (cr, cc-1) not in visited:
            possible_moves.append(MOVE_LEFT)
        if len(possible_moves) == 0:
            break
        random_move = np.random.randint(0, len(possible_moves))
        random_move = possible_moves[random_move]
        moves.append(random_move)
        dr, dc = GO[random_move]
        cr = cr + dr
        cc = cc + dc
        visited.add((cr,cc))
        random_path.append((cr,cc))
    moves.append(MOVE_STOP)
    return random_path, moves

def get_tile(in_move:int, out_move:int, start_end:int=None):
    r"""
    Generates a compatible tile according to a particular inward and outward move.

    Args:
        `in_move` (int): The move that has been done to enter the tile
        `out_move` (int): The move that has been done to exit the tile
        `start_end` (int): A number to indicate if the tile denotes the start or end of a path. Default: `None`
    
    Return:
        Numpy array containing tile information.
    """
    block = np.zeros((CELL_SIZE, CELL_SIZE))
    if in_move == MOVE_RIGHT or out_move == MOVE_LEFT:
        block[CELL_SIZE//2,:CELL_SIZE//2+1] = 1
    if in_move == MOVE_DOWN or out_move == MOVE_UP:
        block[:CELL_SIZE//2+1,CELL_SIZE//2] = 1
    if in_move == MOVE_LEFT or out_move == MOVE_RIGHT:
        block[CELL_SIZE//2,CELL_SIZE//2:] = 1
    if in_move == MOVE_UP or out_move == MOVE_DOWN:
        block[CELL_SIZE//2:,CELL_SIZE//2] = 1

    if start_end != None:
        block[CELL_SIZE//2,CELL_SIZE//2] = start_end

    return block

def get_random_tile():
    r"""
    Generates a random tile.
    """
    
    block = np.zeros((CELL_SIZE,CELL_SIZE))
    p = 0.65

    # up
    coin = np.random.rand()
    if coin <= p:
        block[:CELL_SIZE//2+1,CELL_SIZE//2] = 1

    # right
    coin = np.random.rand()
    if coin <= p:
        block[CELL_SIZE//2,CELL_SIZE//2:] = 1

    # down
    coin = np.random.rand()
    if coin <= p:
        block[CELL_SIZE//2:,CELL_SIZE//2] = 1

    # left
    coin = np.random.rand()
    if coin <= p:
        block[CELL_SIZE//2,:CELL_SIZE//2+1] = 1

    return block
    
def generate_random_grid(grid_size:int):
    r"""
    Generates a grid with random tiles. Solution is not guaranteed.

    Args:
        `grid_size` (int): Size of the grid.

    Return:
        Numpy array containing the random maze grid.
    """
    grid = np.zeros((grid_size*CELL_SIZE, grid_size*CELL_SIZE))
    for i in range(grid_size):
        for j in range(grid_size):
            grid[CELL_SIZE*i:CELL_SIZE*i+CELL_SIZE, CELL_SIZE*j:CELL_SIZE*j+CELL_SIZE] = get_random_tile()
    return grid

def find_shortest_path(grid_size:int, grid:np.array, sr:int, sc:int, er:int, ec:int):
    r"""
    Finds the shortest path in a maze grid.

    Args:
        `grid_size` (int): Size of the maze grid.
        `grid` (np.array): Maze grid
        `sr` (int): Row index of the starting tile
        `sc` (int): Column index of the starting tile
        `er` (int): Row index of the ending tile
        `ec` (int): Column index of the ending tile
    """

    template_h = np.zeros((CELL_SIZE,CELL_SIZE+1))
    template_h[CELL_SIZE//2,:] = 1
    template_v = np.zeros((CELL_SIZE+1,CELL_SIZE))
    template_v[:,CELL_SIZE//2] = 1

    visited = set()
    Q = queue.Queue()
    Q.put((sr, sc))
    par = {(sr, sc):None}
    while not Q.empty():
        cr, cc = Q.get()
        if cr == er and cc == ec:
            break
        visited.add((cr, cc))
        if cr - 1 >= 0 and (cr-1,cc) not in visited:
            connection = np.sign(grid[(cr-1)*CELL_SIZE+CELL_SIZE//2:cr*CELL_SIZE+CELL_SIZE//2+1,cc*CELL_SIZE:cc*CELL_SIZE+CELL_SIZE])
            if (connection * template_v).sum() >= CELL_SIZE:
                Q.put((cr-1,cc))
                par[(cr-1,cc)] = cr,cc,MOVE_UP
        if cc + 1 < grid_size and (cr,cc+1) not in visited:
            connection = np.sign(grid[cr*CELL_SIZE:cr*CELL_SIZE+CELL_SIZE,cc*CELL_SIZE+CELL_SIZE//2:(cc+1)*CELL_SIZE+CELL_SIZE//2+1])
            if (connection * template_h).sum() >= CELL_SIZE:
                Q.put((cr,cc+1))
                par[(cr,cc+1)] = cr,cc,MOVE_RIGHT
        if cr + 1 < grid_size and (cr+1,cc) not in visited:
            connection = np.sign(grid[cr*CELL_SIZE+CELL_SIZE//2:(cr+1)*CELL_SIZE+CELL_SIZE//2+1,cc*CELL_SIZE:cc*CELL_SIZE+CELL_SIZE])
            if (connection * template_v).sum() >= CELL_SIZE:
                Q.put((cr+1,cc))
                par[(cr+1,cc)] = cr,cc,MOVE_DOWN
        if cc - 1 >= 0 and (cr,cc-1) not in visited:
            connection = np.sign(grid[cr*CELL_SIZE:cr*CELL_SIZE+CELL_SIZE,(cc-1)*CELL_SIZE+CELL_SIZE//2:cc*CELL_SIZE+CELL_SIZE//2+1])
            if (connection * template_h).sum() >= CELL_SIZE:
                Q.put((cr,cc-1))
                par[(cr,cc-1)] = cr,cc,MOVE_LEFT
    
    route = [(er,ec)]
    moves = [MOVE_STOP]
    cr, cc = er, ec
    while par[(cr,cc)] is not None:
        cr, cc, m = par[(cr,cc)]
        route.append((cr,cc))
        moves.append(m)
    moves.append(MOVE_START)
    route = list(reversed(route))
    moves = list(reversed(moves))

    return grid, moves, route

def generate_maze_data_sample(grid_size:int, max_path_length:int, shortest:bool=False):
    r"""
    Generates a random maze grid with guaranteed solution.

    Args:
        `grid_size` (int): Size of the maze grid.
        `max_path_length` (int): Maximum length of the solution.
        `shortest` (boolean): If `True`, the random solution will be replaced with the shortest path.
    
    Return:
        `random_grid` (np.array): Numpy array containing the random maze
        `random_moves` (list[int]): Sequence of move ids along the solution path
        `random_path` (list[(int,int)]): Sequence of row and column indices of tiles along the solution path
    """

    random_grid = generate_random_grid(grid_size)
    random_path, random_moves = generate_random_path(grid_size, max_path_length)
    
    for i,(r,c) in enumerate(random_path):
        start_end = None
        if i == 0:
            start_end = 2
        elif i == len(random_path) - 1:
            start_end = 3
        random_grid[r*CELL_SIZE:r*CELL_SIZE+CELL_SIZE, c*CELL_SIZE:c*CELL_SIZE+CELL_SIZE] = np.max(np.array([random_grid[r*CELL_SIZE:r*CELL_SIZE+CELL_SIZE, c*CELL_SIZE:c*CELL_SIZE+CELL_SIZE], get_tile(random_moves[i], random_moves[i+1], start_end)]), axis=0)
    
    if shortest:
        random_grid, random_moves, random_path = find_shortest_path(grid_size, random_grid, random_path[0][0], random_path[0][1], random_path[-1][0], random_path[-1][1])

    return random_grid, random_moves, random_path

def draw_moves_on_grid(grid:np.array, moves:list[int], sr:int, sc:int, er:int, ec:int):
    r"""
    Overlays the given sequence of moves on top of a given grid.

    Args:
        `grid` (np.array): Base maze grid
        `moves` (list[int]): Sequence of moves along the path
        `sr` (int): Row index of the starting tile
        `sc` (int): Column index of the starting tile
        `er` (int): Row index of the ending tile
        `ec` (int): Column index of the ending tile
    
    Return:
        Numpy array containing the overlayed grid.
    """

    gt_grid = grid.copy()
    cr, cc  = sr, sc
    in_move = moves[0]
    for out_move in moves[1:]:
        sol_block = get_tile(in_move, out_move) * 4
        gt_grid[cr*CELL_SIZE:cr*CELL_SIZE+CELL_SIZE, cc*CELL_SIZE:cc*CELL_SIZE+CELL_SIZE] = np.max(np.stack([sol_block, gt_grid[cr*CELL_SIZE:cr*CELL_SIZE+CELL_SIZE, cc*CELL_SIZE:cc*CELL_SIZE+CELL_SIZE]]),axis=0) * np.sign(gt_grid[cr*CELL_SIZE:cr*CELL_SIZE+CELL_SIZE, cc*CELL_SIZE:cc*CELL_SIZE+CELL_SIZE])
        if cr == sr and cc == sc:
            gt_grid[cr*CELL_SIZE+CELL_SIZE//2, cc*CELL_SIZE+CELL_SIZE//2] = 2
        elif cr == er and cc == ec:
            gt_grid[cr*CELL_SIZE+CELL_SIZE//2, cc*CELL_SIZE+CELL_SIZE//2] = 3
        in_move = out_move
        dm = GO[out_move]
        cr = cr + dm[0]
        cc = cc + dm[1]
    return gt_grid

def valid_move(grid:np.array, cr:int, cc:int, next_move:int):
    r"""
    Checks if a move is valid in a given grid.

    Args:
        `grid` (np.array): Maze grid
        `cr` (int): Current tile row index
        `cc` (int): Current tile column index
        `next_move` (int): ID of the move to check

    Return:
        `True` if the move is possible.
    """
    grid_size = grid.shape[0]//CELL_SIZE
    if next_move == MOVE_UP:
        if cr - 1 < 0:
            return False
        if np.any(grid[cr*CELL_SIZE:cr*CELL_SIZE+CELL_SIZE//2+1,cc*CELL_SIZE+CELL_SIZE//2] == 0) or np.any(grid[(cr-1)*CELL_SIZE+CELL_SIZE//2:(cr-1)*CELL_SIZE+CELL_SIZE,cc*CELL_SIZE+CELL_SIZE//2] == 0):
            return False
    if next_move == MOVE_DOWN:
        if cr + 1 >= grid_size:
            return False
        if np.any(grid[cr*CELL_SIZE+CELL_SIZE//2:cr*CELL_SIZE+CELL_SIZE,cc*CELL_SIZE+CELL_SIZE//2] == 0) or np.any(grid[(cr+1)*CELL_SIZE:(cr+1)*CELL_SIZE+CELL_SIZE//2+1,cc*CELL_SIZE+CELL_SIZE//2] == 0):
            return False
    if next_move == MOVE_LEFT:
        if cc - 1 < 0:
            return False
        if np.any(grid[cr*CELL_SIZE+CELL_SIZE//2,cc*CELL_SIZE:cc*CELL_SIZE+CELL_SIZE//2+1] == 0) or np.any(grid[cr*CELL_SIZE+CELL_SIZE//2,(cc-1)*CELL_SIZE+CELL_SIZE//2:(cc-1)*CELL_SIZE+CELL_SIZE] == 0):
            return False
    if next_move == MOVE_RIGHT:
        if cc + 1 >= grid_size:
            return False
        if np.any(grid[cr*CELL_SIZE+CELL_SIZE//2,cc*CELL_SIZE+CELL_SIZE//2:cc*CELL_SIZE+CELL_SIZE] == 0) or np.any(grid[cr*CELL_SIZE+CELL_SIZE//2,(cc+1)*CELL_SIZE:(cc+1)*CELL_SIZE+CELL_SIZE//2+1] == 0):
            return False
    return True