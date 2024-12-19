import torch
import torch.nn as nn
import torch.nn.functional as F


class HybridLoss(torch.nn.Module):
    def __init__(self, margin=2.0):
        super(HybridLoss, self).__init__()
        self.margin = margin
        self.loss_fn = nn.CrossEntropyLoss()

    def contrastive_calc(self, output1, output2, label, mgadd):
        mgadd = ((mgadd+1)**2)-1
        euclidean_distance = F.pairwise_distance(output1, output2, p=2)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp((self.margin+mgadd) - euclidean_distance, min=0.0), 2))
        return loss_contrastive

    def forward(self, label, outlbs, lbs, outred1, outred2, mgadd):
        
        loss_entropy = self.loss_fn(input=outlbs, target=lbs)
        loss_reduc = self.contrastive_calc(outred1, outred2, label, mgadd)

        loss_total = loss_entropy*0.5 + loss_reduc*0.5
        return loss_total #loss.mean()