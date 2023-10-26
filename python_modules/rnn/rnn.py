import torch.nn as nn

class RnnModel(nn.Module):

    """
    RNN
    
    input_size: Number of Features
    hidden_size: number of features in hidden state
    num_layers: Number of Recurrent Layers
    """
    def __init__(self, input_size: int, hidden_size: int, num_layers: int):
        super(RnnModel, self).__init__()
        
        self.layer1 = nn.RNN(input_size, hidden_size, num_layers, batch_first=True, nonlinearity='relu')
        
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        out, _ = self.layer1(x)
        print("Layer Out: {}".format(out))
        
        result = self.fc(out)
        
        print("Full Connected Layer Result: {}".format(result))
        
        return result