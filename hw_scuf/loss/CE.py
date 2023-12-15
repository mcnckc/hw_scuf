import torch
from torch import Tensor
from torch.nn import CrossEntropyLoss


class CE(CrossEntropyLoss):
    def forward(self, logits, target, **batch) -> Tensor:
        return super().forward(
           logits, target
        )
