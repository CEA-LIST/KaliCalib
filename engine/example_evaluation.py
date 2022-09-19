# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""
"""
@author:  davide zambrano
@contact: d.zambrano@sportradar.com

"""
from typing import List, Dict
import json
import logging
import sys
import os

from ignite.engine import Events
from ignite.engine import create_supervised_evaluator
from ignite.metrics import confusion_matrix, MeanAbsoluteError
import torch.nn.functional as F
import numpy as np
import cv2

import torch
from deepsport_utilities.calib import Point2D, Point3D, Calib

sys.path.append(".")
from modeling.example_camera_model import compute_camera_model
from kalicalib.estimateHomography import estimateCalibHM, getFieldPoints2d
from data.datasets.viewds import getFieldPoints


EVAL_GT = {
    "val": "groundtruth/ground_truth_val.json",
    "test": "groundtruth/ground_truth_test.json",
}

TEST_2D_POINTS = Point2D(
    [
        [1, 1, 1 / 2, 1 / 2, 0, 0],
        [1, 1 / 2, 1, 1 / 2, 1, 1 / 2],
    ]
)


def save_predictions_to_json(
    results_list: List[Dict[str, List[np.ndarray]]],
    json_file: str = "predictions.json",
) -> None:
    """Create JSON format results

    Args:
        results_list (List[np.ndarray]): prediction results
        json_file (str, optional): JSON file. Defaults to "predictions.json".
    """
    with open(json_file, "w") as f:
        json.dump(results_list, f, indent=4)


def run_metrics(
    json_file: str = "predictions.json",
    ground_truth: str = "groundtruth/ground_truth_test.json",
) -> None:
    """Compute metrics from JSON. In case of empty dictionary, a default P is provided.

    Args:
        json_file (str): Results saved in JSON file. Defaults to "predictions.json".
        ground_truth (str): The ground truth camera model in JSON format.
    """
    with open(json_file, "r") as f:
        data = f.read()
    with open(ground_truth, "r") as f:
        ground_truth = f.read()

    default_h = np.eye(3, 4)
    default_h[2, 3] = 1.0

    obj = json.loads(data)
    obj_gt = json.loads(ground_truth)

    wid_, hei_ = (obj_gt[0]["width"], obj_gt[0]["height"])
    test_2d_ponts = TEST_2D_POINTS * [[wid_], [hei_]]

    accuracy = 0
    mse_ = []
    len_set = 0

    for dat, gt in zip(obj, obj_gt[1:]):
        len_set += 1
        predicted_P = dat.get("P", None)
        if not predicted_P:
            calib = Calib.from_P(
                np.array(default_h),
                width=wid_,
                height=hei_,
            )
        else:
            predicted_P = np.array(predicted_P).reshape(3, 4)
            accuracy += 1
            calib = Calib.from_P(
                predicted_P,
                width=wid_,
                height=hei_,
            )
        calib_gt = Calib.from_P(
            np.array(gt["P"]).reshape(3, 4),
            width=wid_,
            height=hei_,
        )
        y_pred = calib.project_2D_to_3D(test_2d_ponts, Z=0)
        y = calib_gt.project_2D_to_3D(test_2d_ponts, Z=0)

        val = np.sqrt(np.square(np.subtract(y_pred, y)).mean())
        print(len_set-1, val)
        mse_.append(val)
    accuracy /= len_set
    print(f"Accuracy: {accuracy}, discounted MSE: {np.mean(mse_)} cm")


def json_serialisable(array: np.ndarray) -> List[float]:
    """Takes a np array slice and makes it JSON serialisable.

    Args:
        array (np.ndarray): N-dim np.ndarray.

    Returns:
        List[float]: reformatted 1-dim list of floats.
    """
    if not isinstance(array, np.ndarray) or isinstance(array, Point3D):
        array = np.array(array)

    array_slice = array.reshape(
        -1,
    )
    return list(map(float, array_slice))


class CameraTransform:
    """Callable to output transform."""

    def __init__(self, cfg) -> None:
        self.width, self.height = (
            cfg.INPUT.MULTIPLICATIVE_FACTOR * cfg.INPUT.GENERATED_VIEW_SIZE[0],
            cfg.INPUT.MULTIPLICATIVE_FACTOR * cfg.INPUT.GENERATED_VIEW_SIZE[1],
        )
        self.test_2d_ponts = TEST_2D_POINTS * [[self.width], [self.height]]
        self.dumpable_list = []
        self.is_challenge = cfg.DATASETS.TEST == "challenge"

        self.outputImgDir = os.path.join("output", "evaluation_images")
        os.makedirs(self.outputImgDir , exist_ok=True)

    def __call__(self, x, y, y_pred):
        npImg = y["img"].cpu().numpy()[0]
        npImg = cv2.cvtColor(npImg, cv2.COLOR_RGB2BGR)

        # here use actual prediction
        visualization = False
        calib = estimateCalibHM(y_pred, npImg,
            getFieldPoints2d(), getFieldPoints(),
            self.height, self.width, visualization)

        imgPath = os.path.join(self.outputImgDir, "%03d.png" % y["index"])
        cv2.imwrite(imgPath, npImg)

        data = {
            #"numper of points2d": len(points2d),
        }
        for key, value in calib.dict.items():
            data.update({key: json_serialisable(value)})
        self.dumpable_list.append(data)

        y_pred = calib.project_2D_to_3D(self.test_2d_ponts, Z=0)
        if self.is_challenge:
            return torch.as_tensor(y_pred)

        calib_gt = y["calib"]
        calib_gt = Calib.from_P(
            np.squeeze(calib_gt.cpu().numpy().astype(np.float32)),
            width=self.width,
            height=self.height,
        )
        y = calib_gt.project_2D_to_3D(self.test_2d_ponts, Z=0)

        return (torch.as_tensor(y_pred), torch.as_tensor(y))


def evaluation(cfg, model, val_loader):
    """Evaluation script.

    Args:
        cfg (_type_): config file
        model (_type_): model used for preditctions
        val_loader (_type_): dataset loader
    """
    device = cfg.MODEL.DEVICE

    logger = logging.getLogger("template_model.evaluation")
    logger.info("Start evaluation")
    camera_transform = CameraTransform(cfg)
    metrics = {"mse": MeanAbsoluteError()}
    if cfg.DATASETS.TEST == "challenge":
        metrics = None

    evaluator = create_supervised_evaluator(
        model,
        metrics=metrics,
        device=device,
        output_transform=camera_transform,
    )

    # adding handlers using `evaluator.on` decorator API

    @evaluator.on(Events.EPOCH_COMPLETED)
    def print_validation_results(engine):
        metrics = evaluator.state.metrics
        mse = metrics["mse"] if cfg.DATASETS.TEST == "sviewds" else 0.0
        logger.info(
            "Camera Evaluation Overall Results - MSE: {:.3f}".format(mse)
        )

    @evaluator.on(Events.ITERATION_COMPLETED)
    def print_it(engine):
        print("it", engine.state.iteration)

    evaluator.run(val_loader)

    save_predictions_to_json(camera_transform.dumpable_list)
    if cfg.DATASETS.RUN_METRICS:
        run_metrics("predictions.json", EVAL_GT[cfg.DATASETS.EVAL_ON])
