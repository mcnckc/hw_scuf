import torch
from torch import Tensor
from torch.nn import CrossEntropyLoss


class CE(CrossEntropyLoss):
    def forward(self, logits, target, **batch) -> Tensor:
        print(logits.shape, type(logits), logits.dtype)
        print(target.shape, type(target), target.dtype)
        return super().forward(
           logits, target
        )
