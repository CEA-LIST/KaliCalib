# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""
"""
@author:  davide zambrano
@contact: d.zambrano@sportradar.com

"""
import logging

from ignite.engine import Events
from ignite.engine import create_supervised_evaluator
from ignite.metrics import Loss, mIoU, confusion_matrix
import torch.nn.functional as F


def inference(cfg, model, val_loader):
    """Inference of the segmentation model and metrics.

    Args:
        cfg (_type_): _description_
        model (_type_): _description_
        val_loader (_type_): _description_
    """
    device = cfg.MODEL.DEVICE

    logger = logging.getLogger("template_model.inference")
    logger.info("Start inferencing")
    cm = confusion_matrix.ConfusionMatrix(num_classes=21)

    evaluator = create_supervised_evaluator(
        model,
        metrics={
            "loss": Loss(F.cross_entropy),
            "mIoU": mIoU(cm, ignore_index=0),
            "cm": cm,
        },
        device=device,
        output_transform=lambda x, y, y_pred: (y_pred["out"], y),
    )

    # adding handlers using `evaluator.on` decorator API
    @evaluator.on(Events.EPOCH_COMPLETED)
    def print_validation_results(engine):
        metrics = evaluator.state.metrics
        loss = metrics["loss"]
        miou = metrics["mIoU"]
        logger.info(
            "Validation Results - Cross Entropy Loss: {:.3f} - mIoU: {:.3f}".format(
                loss, miou
            )
        )

    evaluator.run(val_loader)
