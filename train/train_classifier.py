#!/usr/bin/env python3
""" A script which will show how to train an efficientnet classifier on the COCO 2017
dataset. """

import argparse
import pathlib

import torch 

from src import efficientnet
from train import datasets

_DATA_DIR = pathlib.Path("~/datasets/melanoma").expanduser()


def train(
    model: torch.nn.Module, 
    train_loader: torch.utils.data.DataLoader,
    eval_loader: torch.utils.data.DataLoader
) -> None:
    """ Main trainer loop. """
    
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()

    for epoch in range(30):

        for imgs, labels in train_loader:

            if torch.cuda.is_available():
                imgs = imgs.cuda()
                labels = labels.cuda()

            preds = model(imgs)

            loss = loss_fn(preds, labels)
            loss.backward()
            optimizer.step()


if __name__ == "__main__":
    model = efficientnet.EfficientNet("efficientnet-lite0", num_classes=2)
    train_dataset = datasets.ClfDataset(
        _DATA_DIR / "train.csv",
        _DATA_DIR / "train"
    )
    eval_dataset = datasets.ClfDataset(
        _DATA_DIR / "eval.csv",
        _DATA_DIR / "eval"
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=torch.get_num_threads()
    )

    eval_loader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=torch.get_num_threads()
    )

    if torch.cuda.is_available():
        model.cuda()

    train(model, train_loader, eval_loader)