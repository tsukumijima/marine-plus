from collections.abc import Callable

import torch
from torch.nn.modules.loss import _Loss


class LogLikelhood(_Loss):
    def __init__(
        self,
        log_likehood_func: Callable[
            [torch.Tensor, torch.Tensor, torch.Tensor | None], torch.Tensor
        ],
    ) -> None:
        super().__init__()
        self.log_likehood_func = log_likehood_func

    def forward(
        self,
        classified: torch.Tensor,
        label: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size = label.size(0)
        log_likelihood = self.log_likehood_func(classified, label, mask)

        return -log_likelihood / batch_size
