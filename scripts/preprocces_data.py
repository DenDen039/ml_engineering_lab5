#!/usr/bin/env python3
import logging
import sys
import os
import yaml

from pipeline.ingestion import process_data

if __name__ == "__main__":
    params = yaml.safe_load(open("params.yaml", "r"))
    download_cfg = params["download"]
    prepare_cfg = params["prepare"]

    cifar_dir = download_cfg["cifar_dir"]
    train_out = prepare_cfg["train_split_path"]
    val_out = prepare_cfg["val_split_path"]
    test_out = prepare_cfg["test_split_path"]

    ingestion_config = {
        "n_batches": prepare_cfg["n_batches"],
        "batch_indices": list(range(1, prepare_cfg["n_batches"] + 1)),
        "val_size": prepare_cfg["val_size"],
        "random_state": prepare_cfg["random_state"],
    }

    logging.basicConfig(level=logging.INFO)
    try:
        train_df, val_df, test_df = process_data(
            data_dir=cifar_dir, config=ingestion_config)

        os.makedirs(os.path.dirname(train_out), exist_ok=True)
        os.makedirs(os.path.dirname(val_out), exist_ok=True)
        os.makedirs(os.path.dirname(test_out), exist_ok=True)

        train_df.to_pickle(train_out)
        val_df.to_pickle(val_out)
        test_df.to_pickle(test_out)

        logging.info(f"Saved train split pickle → {train_out}")
        logging.info(f"Saved val   split pickle → {val_out}")
        logging.info(f"Saved test  split pickle → {test_out}")
        sys.exit(0)
    except Exception as e:
        logging.error(f"Ingestion failed: {e}")
        sys.exit(1)
