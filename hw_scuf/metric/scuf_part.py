from typing import List

import torch
from torch import Tensor

from hw_scuf.base.base_metric import BaseMetric


class ScufPart(BaseMetric):
    def __init__(self,  *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, log_probs: Tensor, target: Tensor, **kwargs):
        preds = log_probs.argmax(dim=-1)
        return preds.float().mean()
