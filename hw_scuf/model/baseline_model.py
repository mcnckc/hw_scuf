from torch import nn
from torch.nn import Sequential

from hw_scuf.base import BaseModel


class BaselineModel(BaseModel):
    def __init__(self, n_feats, fc_hidden=512, **batch):
        super().__init__(**batch)
        self.net = Sequential(
            # people say it can aproximate any function...
            nn.Linear(in_features=n_feats, out_features=fc_hidden),
            nn.ReLU(),
            nn.Linear(in_features=fc_hidden, out_features=fc_hidden),
            nn.ReLU(),
            nn.Linear(in_features=fc_hidden, out_features=2)
        )

    def forward(self, spectrogram, **batch):
        return {"logits": self.net(spectrogram.transpose(1, 2))}

    def transform_input_lengths(self, input_lengths):
        return input_lengths  # we don't reduce time dimension here
