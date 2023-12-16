import torch
from torch import nn
from torch.nn import Sequential

from hw_scuf.base import BaseModel


class MFM(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    
    def forward(self, x):
        N = x.shape[1] // 2
        return torch.maximum(x[:, :N], x[:, N:])
    
class TransposeChannels(nn.Module):
    def __init__(self, make_last=False, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.make_last = make_last
    def forward(self, x):
        if self.make_last:
            return torch.movedim(x, 1, -1)
        else:
            return torch.movedim(x, -1, 1)

class BatchNorm0d(nn.Module):
    def __init__(self, num_channels, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.bn = nn.BatchNorm1d(num_features=num_channels)
    def forward(self, x):
        return self.bn(x[..., None]).squeeze(dim=-1)
        
class LCNNNoDropout(BaseModel):
    def __init__(self, H, W, **batch):
        super().__init__(**batch)
        nH, nW = H, W
        for _ in range(4):
            nH //= 2
            nW //= 2
        self.net = Sequential(
            nn.Conv2d(1, 64, 5, 1, padding='same'), #1
            MFM(),                                  #2
            nn.MaxPool2d(2),                        #3

            nn.Conv2d(32, 64, 1, 1, padding='same'),#4
            MFM(),                                  #5
            nn.BatchNorm2d(32),                     #6
            nn.Conv2d(32, 96, 3, 1, padding='same'),#7
            MFM(),                                  #8

            nn.MaxPool2d(2),                        #9
            nn.BatchNorm2d(48),                     #10

            nn.Conv2d(48, 96, 1, 1, padding='same'),#11
            MFM(),                                  #12
            nn.BatchNorm2d(48),                     #13
            nn.Conv2d(48, 128, 3, 1, padding='same'),#14
            MFM(),                                  #15

            nn.MaxPool2d(2),                        #16

            nn.Conv2d(64, 64, 1, 1, padding='same'),#17
            MFM(),                                  #18
            nn.BatchNorm2d(32),                     #19
            nn.Conv2d(32, 64, 3, 1, padding='same'),#20
            MFM(),                                  #21

            nn.BatchNorm2d(32),                     #22

            nn.Conv2d(32, 64, 1, 1, padding='same'),#23
            MFM(),                                  #24
            nn.BatchNorm2d(32),                     #25
            nn.Conv2d(32, 64, 3, 1, padding='same'),#26
            MFM(),                                  #27

            nn.MaxPool2d(2),                        #28
            nn.Flatten(),
            nn.Linear(nH * nW * 32, 160),           #29
            MFM(),                                  #30
            BatchNorm0d(80),                        #31
            nn.Linear(80, 2)                        #32
        )

    def forward(self, spectrogram, **batch):
        out = self.net(spectrogram[:, None, :, :])
        return {"logits": out / torch.linalg.vector_norm(self.net[-1].weight, dim=-1)}

    def transform_input_lengths(self, input_lengths):
        return input_lengths  # we don't reduce time dimension here