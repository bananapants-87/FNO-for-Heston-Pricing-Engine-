"""Training entry points and helpers."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable, Optional

import torch
from torch import nn


def _move_to_device(batch, device: torch.device):
    x, y = batch
    return x.to(device), y.to(device)


def evaluate_model(
    model: nn.Module,
    data_loader,
    criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    device: torch.device,
) -> float:
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

    history = {
        "train_loss": [],
        "val_loss": [],
    }

    best_val_loss = float("inf")
    best_state = None

    # Total training timer
    total_start = time.perf_counter()

    # Store duration of every epoch
    epoch_times = []

    # Number of batches in one epoch
    total_batches = len(train_loader)

    print()
    print("=" * 70)
    print("STARTING TRAINING")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Epochs: {epochs}")
    print(f"Batches per epoch: {total_batches}")
    print("=" * 70)
    print()

    for epoch in range(epochs):

        # Start timer for this epoch
        epoch_start = time.perf_counter()

        model.train()

        running_loss = 0.0
        num_batches = 0

        print(f"--- Epoch {epoch + 1}/{epochs} ---")

        for batch_idx, batch in enumerate(train_loader, start=1):

            # Start timer for this batch
            batch_start = time.perf_counter()

            x, y = _move_to_device(batch, device)

            # Print tensor shape for the first batch
            if batch_idx == 1:
                print(f"Input shape : {tuple(x.shape)}")
                print(f"Target shape: {tuple(y.shape)}")

            optimizer.zero_grad(set_to_none=True)

            prediction = model(x)

            loss = criterion(prediction, y)

            loss.backward()

            optimizer.step()

            running_loss += float(loss.item())
            num_batches += 1

            # Batch timing
            batch_time = time.perf_counter() - batch_start

            # Print progress
            print(
                f"Epoch {epoch + 1}/{epochs} - "
                f"Batch {batch_idx}/{total_batches} - "
                f"Loss: {loss.item():.6f} - "
                f"Time: {batch_time:.2f}s"
            )

        # Average training loss for this epoch
        train_loss = running_loss / max(num_batches, 1)

        # Validation
        print("Running validation...")

        val_start = time.perf_counter()

        val_loss = evaluate_model(
            model,
            val_loader,
            criterion,
            device,
        )

        val_time = time.perf_counter() - val_start

        # Scheduler
        if scheduler is not None:
            scheduler.step()

        # Save history
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        # Save best model
        if val_loss < best_val_loss:

            best_val_loss = val_loss

            best_state = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch + 1,
                "best_val_loss": best_val_loss,
            }

        # Epoch timing
        epoch_time = time.perf_counter() - epoch_start

        epoch_times.append(epoch_time)

        print()
        print(
            f"Epoch {epoch + 1}/{epochs} COMPLETE - "
            f"train={train_loss:.6f}, "
            f"val={val_loss:.6f}, "
            f"epoch_time={epoch_time:.2f}s, "
            f"validation_time={val_time:.2f}s"
        )

        print()

    # Total time
    total_time = time.perf_counter() - total_start

    # Sum of epoch times
    sum_epoch_times = sum(epoch_times)

    print()
    print("=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)

    for i, epoch_time in enumerate(epoch_times, start=1):
        print(f"Epoch {i} time: {epoch_time:.2f}s")

    print("-" * 70)

    print(f"Sum of epoch times : {sum_epoch_times:.2f}s")
    print(f"Total wall time    : {total_time:.2f}s")

    print("=" * 70)
    print()

    # Save checkpoint
    if checkpoint_path is not None:

        checkpoint_path = Path(checkpoint_path)

        checkpoint_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        payload = best_state or {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }

        torch.save(
            payload,
            checkpoint_path,
        )

        print(f"Checkpoint saved to: {checkpoint_path}")

    return history