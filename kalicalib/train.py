#!/usr/bin/python3

import sys
import argparse
import os
import random
from datetime import datetime

import torch
from tqdm import tqdm
import numpy as np

sys.path.append(".")
from model_resnet import makeModel
from losses import KeypointsCrossEntropyLoss
from config import cfg
from data import make_data_loader

random.seed(4212)
np.random.seed(4212)
torch.manual_seed(4212)
torch.use_deterministic_algorithms(True)


def train(cfg):
    nEpoch = 500
    outputDir = os.path.join("output", datetime.now().strftime("%d_%m_%Y %H:%M:%S"))
    os.makedirs(outputDir, exist_ok=True)

    device = torch.device("cuda")
    model = makeModel().to(device)
    modelData = torch.load(os.path.join("models", "resnet18-f37072fd.pth"), map_location=device)
    model.load_state_dict(modelData, strict=False)

    weights = torch.ones((94,
        cfg.INPUT.MULTIPLICATIVE_FACTOR * cfg.INPUT.GENERATED_VIEW_SIZE[1] // 4,
        cfg.INPUT.MULTIPLICATIVE_FACTOR * cfg.INPUT.GENERATED_VIEW_SIZE[0] // 4)).to(device)
    weights[:-1] *= 3750
    lossFunc = KeypointsCrossEntropyLoss(weights)

    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 2 * nEpoch // 3)

    dataLoader = make_data_loader(cfg, is_train=True)

    for e in range(nEpoch):
        print("Epoch: %d" % (e + 1))

        pbar = tqdm(dataLoader)
        nbIt = 0
        totalLoss = 0
        for imgs, data in pbar:

            imgs = imgs.to(device)
            heatmaps = data["heatmaps"].to(device)

            out = model(imgs)

            loss = lossFunc(out, heatmaps)
            loss_val = loss.item()
            pbar.set_description("loss: %f" % loss_val)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            totalLoss += loss_val
            nbIt += 1

        lr_scheduler.step()

        pbar.write("Avg loss: %.10e" % (totalLoss / nbIt,))

        if (e + 1) % 10 == 0:
            torch.save(model.state_dict(), os.path.join(outputDir, "model_%d.pth" % (e + 1)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Field calibration training script")
    parser.add_argument("--config_file", default="", help="path to config file", type=str)
    args = parser.parse_args()

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    #cfg.merge_from_list(args.opts)
    cfg.freeze()

    train(cfg)
