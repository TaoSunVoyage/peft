# coding=utf-8
# Copyright 2023-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from abc import ABC, abstractmethod

import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions.relaxed_bernoulli import RelaxedBernoulli

from .config import PolyConfig


EPS = 1e-12


def get_router(poly_config: PolyConfig) -> nn.Module:
    if poly_config.poly_type == "poly":
        return PolyRouter(poly_config)
    elif poly_config.poly_type == "cpoly":
        return CPolyRouter(poly_config)
    elif poly_config.poly_type == "cpoly_simple":
        return CPolySimpleRouter(poly_config)
    else:
        raise ValueError(
            f"Unsupported poly_type: {poly_config.poly_type}. "
            "Currently, only the following types are supported: "
            "`poly`, `cpoly`, `cpoly_simple`."
        )


class Router(nn.Module, ABC):
    @abstractmethod
    def reset(self):
        ...

    @abstractmethod
    def forward(self, task_ids: torch.Tensor, input_ids: torch.Tensor):
        ...


class PolyRouter(Router):
    # It's a simplified implementation of
    # https://github.com/microsoft/mttl/blob/ce4ca51dbca73be656feb9b3e5233633e3c5dec7/mttl/models/poly.py#L138
    def __init__(self, poly_config: PolyConfig):
        super().__init__()

        self.poly_type = poly_config.poly_type
        self.n_tasks = poly_config.n_tasks
        self.n_skills = poly_config.n_skills
        self.n_splits = poly_config.n_splits

        self.module_logits = nn.Parameter(torch.empty((self.n_tasks, self.n_splits * self.n_skills)))

    def reset(self):
        torch.nn.init.uniform_(self.module_logits, -1e-3, 1e-3)

    def forward(self, task_ids: torch.Tensor, input_ids: torch.Tensor):
        if task_ids is None:
            raise ValueError("task_ids should not be None.")
        if task_ids.max().item() >= self.n_tasks:
            raise ValueError(f"Only {self.n_tasks} tasks available. Found task id = {task_ids.max().item()}")

        # move task id to input's device
        task_ids = task_ids.to(self.module_logits.device)

        module_logits = self.module_logits[task_ids]
        module_logits = module_logits.view(-1, self.n_splits, self.n_skills)

        if self.training:
            module_logits = RelaxedBernoulli(temperature=1.0, logits=module_logits).rsample()
        else:
            module_logits = torch.sigmoid(module_logits)

        module_weights = module_logits / (module_logits.sum(dim=-1, keepdim=True) + EPS)

        return module_weights


class CPolySimpleRouter(PolyRouter):
    def forward(self, task_ids: torch.Tensor, input_ids: torch.Tensor):
        module_weights = super().forward(task_ids, input_ids)
        c_module_weights = F.one_hot(task_ids, num_classes=self.n_tasks).unsqueeze(1).expand((1, self.n_splits, 1))
        module_weights = torch.cat((module_weights, c_module_weights), dim=-1)
        return module_weights


class CPolyRouter(PolyRouter):
    def __init__(self, poly_config: PolyConfig):
        super().__init__(poly_config)
        self.c_module_logits = nn.Parameter(torch.empty((self.n_tasks, self.n_tasks)))

    def reset(self):
        super().reset()
        torch.nn.init.eye_(self.c_module_logits)

    def forward(self, task_ids: torch.Tensor, input_ids: torch.Tensor):
        module_weights = super().forward(task_ids, input_ids)
        c_module_weights = self.c_module_logits[task_ids].unsqueeze(1).expand((1, self.n_splits, 1))
        module_weights = torch.cat((module_weights, c_module_weights), dim=-1)
        return module_weights
