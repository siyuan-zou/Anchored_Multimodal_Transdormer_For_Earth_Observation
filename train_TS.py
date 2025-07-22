"""Train using TreeSAT-AI dataset."""

import os
import logging
from copy import deepcopy
from datetime import datetime

import hydra
import torch
import wandb
import pytorch_lightning as pl
from omegaconf import DictConfig
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning import Trainer


from src.models.adapt import ADAPT
from src.utils.training.sizedatamodule import SizeDatamodule
# Datamodule structure
from src.datamodules.multimodal_datamodule import MultimodaDataModule, TSDataModule

def hook_fn(module, input, output):
    # 处理输入
    if isinstance(input, (tuple, list)):
        input_shapes = [i.shape for i in input if hasattr(i, "shape")]
    elif isinstance(input, dict):
        input_shapes = {k: v.shape for k, v in input.items() if hasattr(v, "shape")}
    else:
        input_shapes = input.shape if hasattr(input, "shape") else str(type(input))

    # 处理输出
    if isinstance(output, (tuple, list)):
        output_shapes = [o.shape for o in output if hasattr(o, "shape")]
    elif isinstance(output, dict):
        output_shapes = {k: v.shape for k, v in output.items() if hasattr(v, "shape")}
    else:
        output_shapes = output.shape if hasattr(output, "shape") else str(type(output))

    print(f"{module.__class__.__name__} | Input shape: {input_shapes} -> Output shape: {output_shapes}")


def training(cfg: DictConfig, idx_cv: int, date=""):
    """Train multimodal model for classification."""
    if cfg.get("seed"):
        pl.seed_everything(cfg.seed, workers=True)

    if cfg.logger.mode == "offline":
        os.environ["WANDB_MODE"] = "offline"
        os.environ["WANDB_DIR"] = cfg.paths.logs

    # callbacks
    callbacks = []
    name = f"{date}_{cfg.logger.name}_cv_{idx_cv}"
    model = ADAPT(cfg=cfg, name=name, stage="anchoring")

    ### Resume Training
    if cfg.load_checkpoint:
        checkpoint = torch.load(cfg.path_cpt) # 
        model.load_state_dict(checkpoint["state_dict"], strict=False)

    ######### Add hook #########

    # for layer in model.children():
    #     layer.register_forward_hook(hook_fn)
    
    # def register_hooks(model, hook_fn):
    #     for _, layer in model.named_children():
    #         layer.register_forward_hook(hook_fn)
    #         # 递归注册子模块的 hook
    #         register_hooks(layer, hook_fn)
    # register_hooks(model, hook_fn)

    ######### logger #########
    if cfg.log:
        logger = hydra.utils.instantiate(cfg.logger, id=name)
        callbacks.extend(
            [LearningRateMonitor(logging_interval="epoch"), SizeDatamodule(cfg.log)]
        )
    else:
        logger = None
    # ######### checkpoint #########
    # if cfg.checkpoints:
    #     callbacks.append(
    #         ModelCheckpoint(
    #             save_last=True,
    #             dirpath=f"{cfg.paths.misc}/checkpoints/adapt_TS",
    #             filename=name,
    #             every_n_epochs=5,
    #             verbose=True
    #         )
    #     )

    ######### Anchoring #########
    if cfg.anchoring:
        logging.info("Anchoring training")
        cfg.multimodal.keep_missing = False
        ######### checkpoint #########
        if cfg.checkpoints:
            callbacks1 = deepcopy(callbacks)
            callbacks1.append(
                ModelCheckpoint(
                    save_last=True,
                    dirpath=f"{cfg.paths.misc}/checkpoints/adapt_TS/anchoring",
                    filename=name,
                    every_n_epochs=1,
                    verbose=True,
                    monitor="val_anchoring_loss",
                    mode="min",
                )
            )

        # Datamodule structure
        datamodule = TSDataModule(cfg=cfg, cv=idx_cv)
        trainer = hydra.utils.instantiate(
            cfg.trainer,
            logger=logger,
            max_epochs=cfg.model.anchoring_loss.max_epochs,
            callbacks=callbacks1,
        )

        trainer.fit(model, datamodule)

    ### contrastive training ###
    if cfg.contrastive:
        logging.info("Contrastive training")
        cfg.multimodal.keep_missing = True
        ######### checkpoint #########
        if cfg.checkpoints:
            callbacks2 = deepcopy(callbacks)
            callbacks2.append(
                ModelCheckpoint(
                    save_last=True,
                    dirpath=f"{cfg.paths.misc}/checkpoints/adapt_TS/contrastive",
                    filename=name,
                    every_n_epochs=1,
                    verbose=True,
                    monitor="val_contrastive_loss",
                    mode="min",
                )
            )

        # Datamodule structure
        datamodule = TSDataModule(cfg=cfg, cv=idx_cv)
        datamodule.stage = "contrastive"
        model.stage = "contrastive"
        trainer = hydra.utils.instantiate(
            cfg.trainer,
            logger=logger,
            max_epochs=cfg.model.contrastive_loss.max_epochs,
            # max_epochs=0,
            callbacks=callbacks2,
        )
        trainer.fit(model, datamodule)

    ### classification training ###
    if cfg.clf:
        cfg.multimodal.keep_missing = True
        ######### checkpoint #########
        if cfg.checkpoints:
            callbacks3 = deepcopy(callbacks)
            callbacks3.append(
                ModelCheckpoint(
                    save_last=True,
                    dirpath=f"{cfg.paths.misc}/checkpoints/adapt_TS/classification",
                    filename=name,
                    every_n_epochs=1,
                    verbose=True,
                    monitor="val_classification_loss",
                    mode="min",
                )
            )

        # Datamodule structure
        datamodule_clf = TSDataModule(cfg=cfg, cv=idx_cv, stage="classification")

        model.stage = "classification"
        trainer = hydra.utils.instantiate(
            cfg.trainer,
            logger=logger,
            max_epochs=cfg.model.supervised_loss.max_epochs,
            callbacks=callbacks3,
        )
        trainer.fit(model, datamodule_clf)
        ######### test #########
        if cfg.test:
            cfg.classification_partition = 1 # test_size
            datamodule_clf.cfg.multimodal.hyperparams.batch_size = (
                1  # for test batch = 1 -- different size of samples.
            )
            trainer.test(model, datamodule=datamodule_clf)

    ### END ###
    if cfg.log:
        wandb.finish()


@hydra.main(version_base="1.2", config_path="configs", config_name="TS_ADAPT.yaml")
def main(cfg: DictConfig):
    """Train function for multimodal model."""
    ### set-up ###
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )

    torch.set_float32_matmul_precision("high")
    logger_console = logging.getLogger(__name__)
    logger_console.info("start")
    date_time = datetime.now().strftime("%Y-%m-%d_%Hh%M")

    ### cv = training times ###
    for idx_cv in range(cfg.cv):
        cfg_xp = deepcopy(cfg)
        training(cfg=cfg_xp, idx_cv=idx_cv, date=date_time)
    logger_console.info("finished")


if __name__ == "__main__":
    main()
