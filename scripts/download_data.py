#!/usr/bin/env python3
import logging
import sys
import yaml
from pathlib import Path
from pipeline.download import download_and_extract

if __name__ == "__main__":
    params = yaml.safe_load(open("params.yaml", "r"))
    download_cfg = params.get("download", {})
    cifar_dir = download_cfg.get("cifar_dir", "data/raw/cifar-10-batches-py")

    save_parent = str(Path(cifar_dir).parent)
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"

    logging.basicConfig(level=logging.INFO)
    try:
        extracted_path = download_and_extract(url=url, save_dir=save_parent)
        if extracted_path.endswith(".tar.gz"):
            extracted_path = str(Path(save_parent) / "cifar-10-batches-py")
        logging.info(f"Raw CIFAR-10 extracted to: {extracted_path}")
        sys.exit(0)
    except Exception as e:
        logging.error(f"Download failed: {e}")
        sys.exit(1)
