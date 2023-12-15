from typing import List

import torch
from torch import Tensor

from hw_scuf.base.base_metric import BaseMetric
from hw_scuf.metric.calculate_eer import compute_eer


class EERMetric(BaseMetric):
    def __init__(self,  *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, log_probs: Tensor, target: Tensor, **kwargs):
        bon_probs, scuf_probs = log_probs[:, 0], log_probs[:, 1]
        return compute_eer(bon_probs[target == 0].cpu().detach().numpy(), bon_probs[target == 1].cpu().detach().numpy())[0]
