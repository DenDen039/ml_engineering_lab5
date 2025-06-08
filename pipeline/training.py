import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader

from .metrics import evaluate_metrics


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    loss_function: nn.Module,
    optimizer: optim.Optimizer,
    num_epochs: int,
    device: torch.device,
    save_path: Path = Path("best_model.pth"),
    mlflow_logger=None,
) -> Path:
    model.to(device)
    best_val_loss: float = float("inf")
    best_model_path: Path = Path()

    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0
        for batch_idx, (inputs, targets) in enumerate(train_loader, start=1):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if mlflow_logger is not None:
                mlflow_logger.log_metric("train_loss", loss.item(), step=(
                    epoch - 1) * len(train_loader) + batch_idx)

        avg_train_loss = running_loss / len(train_loader)
        logging.info(
            f"Epoch {epoch}/{num_epochs} - Avg Train Loss: {avg_train_loss:.4f}")

        val_loss, acc, precision, recall, f1 = evaluate_metrics(
            model, val_loader, loss_function, device
        )
        logging.info(
            f"Epoch {epoch}/{num_epochs} - Val Loss: {val_loss:.4f}, "
            f"Acc: {acc:.4f}, Prec: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}"
        )

        if mlflow_logger is not None:
            mlflow_logger.log_metrics(
                {
                    "val_loss": val_loss,
                    "val_accuracy": acc,
                    "val_precision": precision,
                    "val_recall": recall,
                    "val_f1": f1,
                },
                step=epoch
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = save_path
            torch.save(model.state_dict(), best_model_path)
            logging.info(
                f"Best model saved to {best_model_path} with Val Loss: {best_val_loss:.4f}")

            if mlflow_logger is not None:
                mlflow_logger.log_artifact(
                    str(best_model_path), artifact_path="model")

    logging.info("Training complete.")
    return best_model_path
