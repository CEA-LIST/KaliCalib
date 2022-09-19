# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com

and

@author:  davide zambrano
@contact: d.zambrano@sportradar.com

"""

import logging
from torch.utils.tensorboard.writer import SummaryWriter
import torch.nn.functional as F


from ignite.engine import (
    Events,
    create_supervised_trainer,
    create_supervised_evaluator,
)
from ignite.handlers import ModelCheckpoint, Timer
from ignite.metrics import Loss, RunningAverage, mIoU, confusion_matrix


def do_train(
    cfg,
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    loss_fn,
):
    """_summary_

    Args:
        cfg (_type_): config object
        model (_type_): _description_
        train_loader (_type_): _description_
        val_loader (_type_): _description_
        optimizer (_type_): _description_
        scheduler (_type_): _description_
        loss_fn (_type_): _description_
    """
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    output_dir = cfg.OUTPUT_DIR
    device = cfg.MODEL.DEVICE
    epochs = cfg.SOLVER.MAX_EPOCHS

    logger = logging.getLogger("template_model.train")
    logger.info("Start training")
    tensor_board_writer = SummaryWriter()

    cm = confusion_matrix.ConfusionMatrix(num_classes=cfg.MODEL.NUM_CLASSES)
    trainer = create_supervised_trainer(
        model, optimizer, loss_fn, device=device
    )
    evaluator = create_supervised_evaluator(
        model,
        metrics={"loss": Loss(F.cross_entropy), "mIoU": mIoU(cm)},
        device=device,
        output_transform=lambda x, y, y_pred: (y_pred["out"], y),
    )

    checkpointer = ModelCheckpoint(
        output_dir,
        "model",
        save_interval=None,
        n_saved=10,
        require_empty=False,
    )
    timer = Timer(average=True)

    trainer.add_event_handler(
        Events.EPOCH_COMPLETED,
        checkpointer,
        {"model": model, "optimizer": optimizer},
    )
    timer.attach(
        trainer,
        start=Events.EPOCH_STARTED,
        resume=Events.ITERATION_STARTED,
        pause=Events.ITERATION_COMPLETED,
        step=Events.ITERATION_COMPLETED,
    )

    RunningAverage(output_transform=lambda x: x).attach(trainer, "avg_loss")

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(engine):
        iter = (engine.state.iteration - 1) % len(train_loader) + 1

        if iter % log_period == 0:
            logger.info(
                "Epoch[{}] Iteration[{}/{}] Loss: {:.2f}".format(
                    engine.state.epoch,
                    iter,
                    len(train_loader),
                    engine.state.metrics["avg_loss"],
                )
            )
        tensor_board_writer.add_scalar(
            "Loss/train",
            engine.state.metrics["avg_loss"],
            engine.state.iteration,
        )

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(engine):
        evaluator.run(train_loader)
        metrics = evaluator.state.metrics
        avg_loss = metrics["loss"]
        miou = metrics["mIoU"]
        logger.info(
            "Training Results - Epoch: {} Avg Loss: {:.3f} mIoU: {:.3f}".format(
                engine.state.epoch, avg_loss, miou
            )
        )
        tensor_board_writer.add_scalar(
            "Avg Loss/train", avg_loss, engine.state.epoch
        )

    if val_loader is not None:

        @trainer.on(Events.EPOCH_COMPLETED)
        def log_validation_results(engine):
            evaluator.run(val_loader)
            metrics = evaluator.state.metrics
            avg_loss = metrics["loss"]
            miou = metrics["mIoU"]
            logger.info(
                "Validation Results - Epoch: {} Avg Loss: {:.3f} mIoU: {:.3f}".format(
                    engine.state.epoch, avg_loss, miou
                )
            )
            tensor_board_writer.add_scalar(
                "Loss/val", avg_loss, engine.state.epoch
            )
            tensor_board_writer.add_scalar("mIoU", miou, engine.state.epoch)
            tensor_board_writer.flush()

    # adding handlers using `trainer.on` decorator API
    @trainer.on(Events.EPOCH_COMPLETED)
    def print_times(engine):
        logger.info(
            "Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]".format(
                engine.state.epoch,
                timer.value() * timer.step_count,
                train_loader.batch_size / timer.value(),
            )
        )
        timer.reset()

    trainer.run(train_loader, max_epochs=epochs)
