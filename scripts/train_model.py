#!/usr/bin/env python3
import logging
import os
import yaml
import pandas as pd
import torch
from torch import nn, optim

import wandb

from pipeline.models import ResNetClassifier
from pipeline.loader import create_data_loader
from pipeline.training import train_model as train_loop

if __name__ == "__main__":
    cfg = yaml.safe_load(open("params.yaml", "r"))
    prepare_cfg = cfg["prepare"]
    train_cfg = cfg["train"]
    wandb_cfg = cfg.get("wandb", {})

    wandb.init(
        project=wandb_cfg["project"],
        config={
            "lr": train_cfg["lr"],
            "num_epochs": train_cfg["num_epochs"],
            "batch_size": train_cfg["batch_size"],
            "num_workers": train_cfg["num_workers"],
            "device": train_cfg["device"],
        },
        reinit=True
    )
    logging.basicConfig(level=logging.INFO)

    train_df = pd.read_pickle(prepare_cfg["train_split_path"])
    val_df = pd.read_pickle(prepare_cfg["val_split_path"])
    loader_cfg = {
        "batch_size": train_cfg["batch_size"],
        "num_workers": train_cfg["num_workers"],
        "transform": None,
    }
    train_loader = create_data_loader(train_df, loader_cfg)
    val_loader = create_data_loader(val_df, loader_cfg)

    device = torch.device(train_cfg["device"]
                          if torch.cuda.is_available() else "cpu")
    model = ResNetClassifier(n_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=train_cfg["lr"])

    os.makedirs(os.path.dirname(train_cfg["model_save_path"]), exist_ok=True)

    best_val_loss = float("inf")
    best_model_path = train_cfg["model_save_path"]
    global_step = 0

    for epoch in range(1, train_cfg["num_epochs"] + 1):
        model.train()
        epoch_loss = 0.0
        for batch_idx, (x, y) in enumerate(train_loader, start=1):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            preds = model(x)
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            global_step += 1
            wandb.log({"train/loss": loss.item()}, step=global_step)

        val_loss, acc, prec, rec, f1 = train_loop.__globals__["evaluate_metrics"](
            model, val_loader, criterion, device
        )
        wandb.log({
            "val/loss":      val_loss,
            "val/accuracy":  acc,
            "val/precision": prec,
            "val/recall":    rec,
            "val/f1":        f1,
        }, step=global_step)

        logging.info(
            f"Epoch {epoch}/{train_cfg['num_epochs']} - "
            f"Avg Train Loss: {epoch_loss/len(train_loader):.4f}, Val Loss: {val_loss:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            artifact = wandb.Artifact(
                name="resnet_cifar10", type="model",
                metadata={"epoch": epoch, "val_loss": val_loss}
            )
            artifact.add_file(best_model_path)
            wandb.log_artifact(artifact)

    wandb.finish()
