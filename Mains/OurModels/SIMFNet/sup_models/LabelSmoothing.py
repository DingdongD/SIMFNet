import torch.nn as nn
import torch
import torch.nn.functional as F


class labelsmoothing(nn.Module):
    def __init__(self, smoothing=0.1):
        super(labelsmoothing, self).__init__()
        self.confidence = 1.0 - smoothing

    def forward(self, x, y):
        n_class = x.size(-1)
        logprobs = F.log_softmax(x, dim=1)
        y = y * self.confidence + (1 - y) * (1 - self.confidence) / (n_class - 1)
        loss = -torch.sum(y * logprobs, dim=1)

        return loss.mean()
        
## RMSELoss=torch.sqrt(self.criterion(prediction_data,label_data))
