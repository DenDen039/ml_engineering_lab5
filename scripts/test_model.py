#!/usr/bin/env python3
import logging
import os
import yaml
import json
import pandas as pd
import torch
from torch import nn

import wandb

from pipeline.models import ResNetClassifier
from pipeline.loader import create_data_loader
from pipeline.testing import test_model

if __name__ == "__main__":
    cfg = yaml.safe_load(open("params.yaml", "r"))
    prepare = cfg["prepare"]
    train_cfg = cfg["train"]
    test_cfg = cfg["test"]
    wandb_cfg = cfg.get("wandb", {})

    wandb.init(
        project=wandb_cfg["project"],
        name="evaluation",
        config={
            "eval_batch_size": test_cfg.get("batch_size", 32)
        },
        reinit=True
    )

    logging.basicConfig(level=logging.INFO)

    test_df = pd.read_pickle(prepare["test_split_path"])
    loader_cfg = {
        "batch_size":  test_cfg.get("batch_size", 32),
        "num_workers": test_cfg.get("num_workers", 2),
        "transform":   None,
    }
    test_loader = create_data_loader(test_df, loader_cfg)

    device = torch.device(
        train_cfg["device"] if torch.cuda.is_available() else "cpu"
    )
    model = ResNetClassifier(n_classes=10).to(device)
    model.load_state_dict(torch.load(
        train_cfg["model_save_path"], map_location=device))

    val_loss, acc, prec, rec, f1 = test_model(
        model=model,
        test_loader=test_loader,
        device=device,
        loss_function=nn.CrossEntropyLoss()
    )

    metrics = {
        "eval/loss":     val_loss,
        "eval/accuracy": acc,
        "eval/precision": prec,
        "eval/recall":    rec,
        "eval/f1":        f1,
    }
    wandb.log(metrics)
    wandb.finish()

    os.makedirs(os.path.dirname(test_cfg["metrics_out"]), exist_ok=True)
    with open(test_cfg["metrics_out"], "w") as f:
        json.dump(metrics, f, indent=2)

    logging.info(f"Evaluation complete. Metrics at {test_cfg['metrics_out']}")
