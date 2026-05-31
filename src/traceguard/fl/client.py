"""Federated client local training."""

from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy

import torch
from torch import nn
from torch.utils.data import DataLoader


@dataclass
class ClientResult:
    client_id: int
    update: dict[str, torch.Tensor]
    num_samples: int
    train_loss: float


class FLClient:
    def __init__(
        self,
        client_id: int,
        dataloader: DataLoader,
        device: torch.device | str,
        hardening=None,
    ) -> None:
        self.client_id = client_id
        self.dataloader = dataloader
        self.device = torch.device(device)
        self.hardening = hardening

    def train(
        self,
        global_model: nn.Module,
        *,
        local_epochs: int,
        lr: float,
        momentum: float,
    ) -> ClientResult:
        local_model = deepcopy(global_model).to(self.device)
        local_model.train()

        initial_state = {
            key: value.detach().cpu().clone()
            for key, value in global_model.state_dict().items()
        }
        optimizer = torch.optim.SGD(local_model.parameters(), lr=lr, momentum=momentum)
        criterion = nn.CrossEntropyLoss()

        total_loss = 0.0
        total_seen = 0
        for _ in range(local_epochs):
            for images, labels in self.dataloader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                if self.hardening is not None:
                    hard_images, hard_labels = self.hardening.make_hardening_batch(
                        images,
                        labels,
                    )
                    if hard_images.numel() > 0:
                        images = torch.cat([images, hard_images], dim=0)
                        labels = torch.cat([labels, hard_labels], dim=0)
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(local_model(images), labels)
                loss.backward()
                optimizer.step()
                batch_size = int(labels.numel())
                total_loss += float(loss.item()) * batch_size
                total_seen += batch_size

        local_state = {
            key: value.detach().cpu().clone()
            for key, value in local_model.state_dict().items()
        }
        update = {
            key: local_state[key] - initial_state[key]
            for key in initial_state
        }
        mean_loss = total_loss / max(total_seen, 1)
        return ClientResult(
            client_id=self.client_id,
            update=update,
            num_samples=len(self.dataloader.dataset),
            train_loss=mean_loss,
        )
