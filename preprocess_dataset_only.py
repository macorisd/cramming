#!/usr/bin/env python3
"""Preprocess a cramming dataset without constructing a model."""

import logging
import os

import hydra
import torch

import cramming

log = logging.getLogger(__name__)


@hydra.main(config_path="cramming/config", config_name="cfg_pretrain", version_base="1.1")
def launch(cfg):
    torch.set_num_threads(min(torch.get_num_threads(), int(cfg.impl.threads)))
    dataset, tokenizer = cramming.load_pretraining_corpus(cfg.data, cfg.impl)
    log.info(
        "Dataset %s is ready at %s with %s blocks and tokenizer vocab size %s.",
        cfg.data.name,
        os.path.abspath(cfg.impl.path),
        f"{len(dataset):,}",
        tokenizer.vocab_size,
    )
    print(
        f"Dataset {cfg.data.name} ready: "
        f"{len(dataset):,} blocks, tokenizer vocab size {tokenizer.vocab_size}, cache {os.path.abspath(cfg.impl.path)}"
    )


if __name__ == "__main__":
    launch()
