# Copyright (c) 2021 Hongji Wang (jijijiang77@gmail.com)
#               2022 Chengdong Liang (liangchengdong@mail.nwpu.edu.cn)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations
from typing import List, Dict, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F


def make_pruning_param_groups(model: torch.nn.Module,
                              cls_lr: float = 2e-4,
                              reg_lr: float | None = 2e-2
) -> Tuple[List[Dict[str, Any]], Tuple[nn.Parameter, nn.Parameter]]:
    """
    Return param groups for AdamW supporting learnable λ / log_alpha.
    """
    main_params = [p for n, p in model.named_parameters() if "log_alpha" not in n]
    lambda1 = nn.Parameter(torch.tensor(0.0))
    lambda2 = nn.Parameter(torch.tensor(0.0))
    pg = [
        {"params": main_params, "lr": cls_lr, "weight_decay": 0.0, "name": "main"},
    ]
    if reg_lr is not None:
        pg += [
            {"params": [p for n, p in model.named_parameters() if "log_alpha" in n],
             "lr": reg_lr, "weight_decay": 0.0, "name": "log_alpha"},
            {"params": [lambda1, lambda2], "lr": -reg_lr, "weight_decay": 0.0, "name": "lambda"},
        ]
    return pg, (lambda1, lambda2)

def pruning_loss(cur_params, orig_params, target_sp, l1, l2):
    """Return pruning regularization term & expected sparsity."""
    expected_sp = 1.0 - cur_params / orig_params
    diff = expected_sp - target_sp
    return l1 * diff + l2 * diff * diff, expected_sp
