GLOBAL = {
    'DEVICE'          : 'cuda',
    'SEED'            : 128,
}

MAZE = {
    'GRID_SIZE'       : 5,
    'MAX_PATH_LENGTH' : 20,
    'SHORTEST_PATH'   : True,
}

DATASET = {
    'NUM_TRAIN'       : 10000,
    'NUM_EVAL'        : 1000,
}

TRANSFORMERS = {
    'BATCH_SIZE'      : 1024, 
    'MAX_EPOCH'       : 2000,
    'D_MODEL'         : 32,
    'N_HEAD'          : 8,
    'N_ENCODER'       : 4,
    'N_DECODER'       : 10,
    'FC_DIM'          : 512,
    'LR'              : 0.0001,
}

CNNS = {
    'BATCH_SIZE'      : 1024, 
    'MAX_EPOCH'       : 1000,
    'D_MODEL'         : 32,
    'FC_DIM'          : 2**13,
    'LR'              : 0.0001,
    'WD'              : 0.001,
}