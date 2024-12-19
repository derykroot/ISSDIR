from torch import nn

class model(nn.Module):
    def __init__(self, in_dim = 1536):
        super().__init__()
       
        in_dim, hid_dim = in_dim, 12288 # 6144 # 3072 # 768

        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.BatchNorm1d(in_dim),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.BatchNorm1d(in_dim),
            nn.ReLU(inplace=True)
        )
        self.layer3 = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.BatchNorm1d(in_dim),
            nn.ReLU(inplace=True)
        )
        self.layerlast = nn.Sequential(
            nn.Linear(in_dim, 2),
            nn.BatchNorm1d(2)
        )        
        
    def forward(self, X):
        x = self.layer1(X)
        x = self.layer2(x)
        x = self.layer3(x)
        embds = self.layerlast(x)
        
        return embds