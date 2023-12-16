from typing import List

import torch
from torch import Tensor

from hw_scuf.base.base_metric import BaseMetric


class AccuracyMetric(BaseMetric):
    def __init__(self,  *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, log_probs: Tensor, target: Tensor, **kwargs):
        preds = log_probs.argmax(dim=-1)
        print("LOG PROBS", log_probs.shape)
        print("PREDS", preds.shape)
        print("TARGET", target.shape)
        print('Measure', torch.abs(preds-target).float().mean())
        return torch.sum(preds == target) / preds.numel()
