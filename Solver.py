"""
Implements various solvers for the maze
"""
from torch import nn

class TransformerSolver(nn.Module):
    def __init__(self, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, batch_first) -> None:
        super().__init__()
        self.transformer = nn.Transformer(d_model=d_model, nhead=nhead, num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers,
                                          dim_feedforward=dim_feedforward, batch_first=batch_first)
        self.classifier  = nn.Sequential(nn.Linear(d_model, 5))
        self.d_model = d_model
    
    def forward(self, x, y, mask=True):
        if mask:
            net_out = self.transformer(x,y,tgt_mask=nn.Transformer.generate_square_subsequent_mask(y.shape[1], y.device))
        else:
            net_out = self.transformer(x,y)
        net_out = net_out.reshape(-1, self.d_model)
        pred    = self.classifier(net_out)
        pred    = pred.reshape(len(y), -1, 5)
        return pred
    
class CNNSolver(nn.Module):
    def __init__(self, d_model, dim_feedforward) -> None:
        super().__init__()
        self.d_model = d_model
        self.features = nn.Sequential(
            nn.Conv2d(1, d_model, 3, 1, 1), nn.ReLU(),
            nn.Conv2d(d_model, d_model, 5, 5, 0), nn.ReLU(),
            nn.Conv2d(d_model, d_model, 3, 1, 1), nn.ReLU(),
            nn.MaxPool2d(2,2,1),
            nn.Conv2d(d_model, d_model, 3, 1, 0), nn.ReLU(),    
        )
        self.classifier  = nn.Sequential(nn.Linear(d_model, dim_feedforward), nn.ReLU(), nn.Dropout(), nn.Linear(dim_feedforward, 5))
    
    def forward(self, x):
        f = self.features(x)
        f = f.reshape(-1, self.d_model)
        pred    = self.classifier(f)
        return pred