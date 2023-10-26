import torch.nn as nn

# LSTM 모델 정의
class StockPredictor(nn.Module):
    
    """
    input_dim: Number of Input Features
    hidden_dim: Number of features in hidden state
    num_layers: Number of Recurrent Layers
    """
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(StockPredictor, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.linear(out[:, -1, :])
        return out
