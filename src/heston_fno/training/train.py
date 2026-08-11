"""Training entry points and helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import torch
from torch import nn


def _move_to_device(batch, device: torch.device):
	x, y = batch
	return x.to(device), y.to(device)


def evaluate_model(model: nn.Module, data_loader, criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor], device: torch.device) -> float:
	model.eval()
	total_loss = 0.0
	num_batches = 0
	with torch.no_grad():
		for batch in data_loader:
			x, y = _move_to_device(batch, device)
			loss = criterion(model(x), y)
			total_loss += float(loss.item())
			num_batches += 1
	return total_loss / max(num_batches, 1)


def train_model(
	model: nn.Module,
	train_loader,
	val_loader,
	optimizer: torch.optim.Optimizer,
	criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
	device: torch.device,
	epochs: int,
	checkpoint_path: str | Path | None = None,
	scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
) -> dict[str, list[float]]:
	history = {"train_loss": [], "val_loss": []}
	best_val_loss = float("inf")
	best_state = None

	for epoch in range(epochs):
		model.train()
		running_loss = 0.0
		num_batches = 0

		for batch in train_loader:
			x, y = _move_to_device(batch, device)
			optimizer.zero_grad(set_to_none=True)
			prediction = model(x)
			loss = criterion(prediction, y)
			loss.backward()
			optimizer.step()

			running_loss += float(loss.item())
			num_batches += 1

		train_loss = running_loss / max(num_batches, 1)
		val_loss = evaluate_model(model, val_loader, criterion, device)

		if scheduler is not None:
			scheduler.step()

		history["train_loss"].append(train_loss)
		history["val_loss"].append(val_loss)

		if val_loss < best_val_loss:
			best_val_loss = val_loss
			best_state = {
				"model_state_dict": model.state_dict(),
				"optimizer_state_dict": optimizer.state_dict(),
				"epoch": epoch + 1,
				"best_val_loss": best_val_loss,
			}

		print(f"Epoch {epoch + 1}/{epochs}: train={train_loss:.6f}, val={val_loss:.6f}")

	if checkpoint_path is not None:
		checkpoint_path = Path(checkpoint_path)
		checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
		payload = best_state or {"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict()}
		torch.save(payload, checkpoint_path)

	return history
