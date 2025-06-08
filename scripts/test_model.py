#!/usr/bin/env python3
import logging
import sys
import os
import yaml
import json

import pandas as pd
import torch
from torch import nn

import mlflow

from pipeline.models import ResNetClassifier
from pipeline.loader import create_data_loader
from pipeline.testing import test_model

if __name__ == "__main__":
    params = yaml.safe_load(open("params.yaml", "r"))
    prepare_cfg = params["prepare"]
    train_cfg = params["train"]
    test_cfg = params["test"]
    mlflow_cfg = params.get("mlflow", {})

    mlflow.set_tracking_uri(mlflow_cfg.get(
        "tracking_uri", "http://localhost:5000"))
    mlflow.set_experiment(mlflow_cfg.get("experiment_name", "Default"))

    test_pkl = prepare_cfg["test_split_path"]
    model_path = train_cfg["model_save_path"]
    metrics_out = test_cfg["metrics_out"]

    logging.basicConfig(level=logging.INFO)

    try:
        test_df = pd.read_pickle(test_pkl)
        loader_cfg = {
            "batch_size":  test_cfg.get("batch_size", 32),
            "num_workers": test_cfg.get("num_workers", 2),
            "transform":   None,
        }
        test_loader = create_data_loader(test_df, loader_cfg)

        device = torch.device(train_cfg.get("device", "cpu")
                              if torch.cuda.is_available() else "cpu")
        model = ResNetClassifier(n_classes=10).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))

        os.makedirs(os.path.dirname(metrics_out), exist_ok=True)

        with mlflow.start_run(run_name="evaluation"):
            mlflow.log_param("eval_batch_size", test_loader.batch_size)

            val_loss, acc, precision, recall, f1 = test_model(
                model=model,
                test_loader=test_loader,
                device=device,
                loss_function=nn.CrossEntropyLoss()
            )

            metrics_dict = {
                "eval_loss": val_loss,
                "eval_accuracy": acc,
                "eval_precision": precision,
                "eval_recall": recall,
                "eval_f1": f1,
            }

            mlflow.log_metrics(metrics_dict)

            with open(metrics_out, "w") as f:
                json.dump(metrics_dict, f, indent=2)

        logging.info(f"Evaluation complete. Metrics saved to: {metrics_out}")
    except Exception as e:
        logging.error(f"Evaluation failed: {e}")
        sys.exit(1)
