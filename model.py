
import torch.nn as nn

class netmodel(nn.Module):
    def __init__(self, out_dim, in_dim = 1536):
        super().__init__()
       
        in_dim, hid_dim = in_dim, 12288 # 6144 # 3072 # 768

        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hid_dim),
            nn.BatchNorm1d(hid_dim),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hid_dim, hid_dim),
            nn.BatchNorm1d(hid_dim),
            nn.ReLU(inplace=True)
        )
        self.layer3 = nn.Sequential(
            nn.Linear(hid_dim, 3072),
            nn.BatchNorm1d(3072),
            nn.ReLU(inplace=True)
        )
        self.layerlast = nn.Sequential(
            nn.Linear(3072, in_dim),
            nn.BatchNorm1d(in_dim)
        )
        self.layerlogits = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim)
        )

        self.encoder = nn.Sequential(
            nn.Linear(in_dim, 3072),
            nn.BatchNorm1d(3072),
            nn.GELU(),
            nn.Linear(3072, 1536),
            nn.BatchNorm1d(1536),
            nn.GELU(),
            nn.Linear(1536, 768),
            nn.BatchNorm1d(768),
            nn.GELU(),
            nn.Linear(768, 384),
            nn.BatchNorm1d(384),
            nn.GELU(),
            nn.Linear(384, 96),
            nn.BatchNorm1d(96),
            nn.GELU(),
            nn.Linear(96, 2),
            nn.BatchNorm1d(2),
        )
        
        
    def forward(self, X):
        x = self.layer1(X)
        x = self.layer2(x)
        x = self.layer3(x)
        embds = self.layerlast(x) 
        redembs = self.encoder(embds)
        logits = self.layerlogits(embds)
        
        return embds, logits, redembs