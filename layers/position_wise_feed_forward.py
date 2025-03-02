from torch import nn

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, hidden_dim, drop_prob=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=drop_prob)
        )
        self.linear2 = nn.Linear(hidden_dim, d_model)

    def forward(self, x):
        x = self.layer1(x)
        x = self.linear2(x)
        return x