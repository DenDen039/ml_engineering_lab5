#!/usr/bin/env python3
import logging
import sys
import os
import yaml
import pandas as pd
import torch
from torch import nn, optim

import mlflow

from pipeline.models import ResNetClassifier
from pipeline.loader import create_data_loader
from pipeline.training import train_model

if __name__ == "__main__":
    params = yaml.safe_load(open("params.yaml", "r"))
    prepare_cfg = params["prepare"]
    train_cfg = params["train"]
    mlflow_cfg = params["mlflow"]

    mlflow.set_tracking_uri(mlflow_cfg["tracking_uri"])
    mlflow.set_experiment(mlflow_cfg["experiment_name"])

    train_pkl = prepare_cfg["train_split_path"]
    val_pkl = prepare_cfg["val_split_path"]
    model_out = train_cfg["model_save_path"]
    lr = train_cfg["lr"]
    num_epochs = train_cfg["num_epochs"]
    batch_size = train_cfg["batch_size"]
    num_workers = train_cfg["num_workers"]
    device_str = train_cfg["device"]

    logging.basicConfig(level=logging.INFO)

    try:
        train_df = pd.read_pickle(train_pkl)
        val_df = pd.read_pickle(val_pkl)
        loader_cfg = {
            "batch_size":   batch_size,
            "num_workers":  num_workers,
            "transform":    None,
        }
        train_loader = create_data_loader(train_df, loader_cfg)
        val_loader = create_data_loader(val_df, loader_cfg)

        device = torch.device(
            device_str if torch.cuda.is_available() else "cpu")
        model = ResNetClassifier(n_classes=10).to(device)
        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=lr)

        os.makedirs(os.path.dirname(model_out), exist_ok=True)

        with mlflow.start_run():
            mlflow.log_params({
                "lr":           lr,
                "num_epochs":   num_epochs,
                "batch_size":   batch_size,
                "num_workers":  num_workers,
                "device":       device_str,
            })

            best_model_path = train_model(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                loss_function=loss_fn,
                optimizer=optimizer,
                num_epochs=num_epochs,
                device=device,
                save_path=model_out,
                mlflow_logger=mlflow
            )

            mlflow.pytorch.log_model(
                pytorch_model=model,
                artifact_path="model",
                registered_model_name="ResNet_CIFAR10"
            )
        logging.info(
            f"Training complete. Best model saved at: {best_model_path}")
    except Exception as e:
        logging.error(f"Training failed: {e}")
        sys.exit(1)
