import os
import torch
import hydra
import json
import logging
from copy import deepcopy
import wandb
from datetime import datetime
import pytorch_lightning as pl

from omegaconf import DictConfig
from src.datamodules.datasets.stressid import StressID
from src.datamodules.multimodal_datamodule import MultimodaDataModule, TSDataModule
from src.datamodules.datasets.TreeSAT import TreeSAT
from src.datamodules.datasets.transforms.transform import Transform
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from src.utils.training.sizedatamodule import SizeDatamodule

from src.models.encoders.ltae import LTAE2d
from src.models.encoders.patch_embeddings import PatchEmbed
from src.models.adapt import ADAPT
from src.models.modules.backbone import Encoders_TS

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


def print_dict_sample_data(sample, stage="anchoring"):
    if stage == "contrastive":
        for index, item in enumerate(sample):
            print("Output", index, ": ")
            for key, value in item.items():
                if isinstance(value, torch.Tensor):
                    print(f"Key: {key}, Shape: {value.shape}")
                else:
                    print(f"Key: {key}, Value: {value}")
            print("label", item["label"])
    else:
        for key, value in sample.items():
            if isinstance(value, torch.Tensor):
                print(f"Key: {key}, Shape: {value.shape}")
            else:
                print(f"Key: {key}, Value: {value}")
        print("label", sample["label"])

@hydra.main(version_base="1.2", config_path="configs", config_name="TS_ADAPT.yaml")
def main1(cfg: DictConfig, stage: str = "contrastive"):
    # print(os.listdir("misc/StressID/StressID_Dataset/Videos"))
    # folder_path = "misc/StressID/StressID_Dataset/Videos"
    # directories_with_files = []

    # for root, dirs, files in os.walk(folder_path):
    #     for file in files:
    #         if file.endswith(".mp4"):
    #             directories_with_files.append(root)

    # print(directories_with_files)

    # print(torch.cuda.is_available())  # Should return True
    # print(torch.cuda.device_count())  # Should be > 0
    # print(torch.cuda.get_device_name(0))  # Verify GPU name

    dataset1 = TSDataModule(cfg).setup()
    sample1 = dataset1[0]

    print(f"Dataset1 sample type: {type(sample1)}")

    print_dict_sample_data(sample1)


@hydra.main(version_base="1.2", config_path="/Data/zou.siyuan/ADAPT/configs", config_name="config.yaml")
def main2(cfg: DictConfig):
    dataset2 = MultimodaDataModule(cfg).setup()
    print("stressid dataset length ", len(dataset2))
    sample2 = dataset2[0]

    print(f"Dataset2 sample type: {type(sample2)}")

    for key, value in sample2.items():
        if isinstance(value, torch.Tensor):
            print(f"Key: {key}, Shape: {value.shape}")
        else:
            print(f"Key: {key}, Value: {value}")
    
    print("label", sample2["label"])


def training(path_cpt, cfg: DictConfig, idx_cv: int, date=""):

    if cfg.get("seed"):
        pl.seed_everything(cfg.seed, workers=True)

    if cfg.logger.mode == "offline":
        os.environ["WANDB_MODE"] = "offline"
        os.environ["WANDB_DIR"] = cfg.paths.logs

    # callbacks
    callbacks = []
    name = f"{date}_{cfg.logger.name}_cv_{idx_cv}"
    model = ADAPT(cfg=cfg, name=name, stage="anchoring")

    checkpoint = torch.load(path_cpt) # 100 epochs pre-trained model before classification
    print(checkpoint.keys())
    model.load_state_dict(checkpoint["state_dict"], strict=False)

    ######### logger #########
    if cfg.log:
        logger = hydra.utils.instantiate(cfg.logger, id=name)
        callbacks.extend(
            [LearningRateMonitor(logging_interval="epoch"), SizeDatamodule(cfg.log)]
        )
    else:
        logger = None

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
            max_epochs=cfg.model.classification_loss.max_epochs,
            callbacks=callbacks3,
        )
        trainer.fit(model, datamodule_clf)
        ######### test #########
        if cfg.test:
            datamodule_clf.cfg.multimodal.hyperparams.batch_size = (
                1  # for test batch = 1 -- different size of samples.
            )
            trainer.test(model, datamodule=datamodule_clf)

    ### END ###
    if cfg.log:
        wandb.finish()

# continue with cpt
@hydra.main(version_base="1.2", config_path="configs", config_name="TS_ADAPT.yaml")
def main_continue_with_cpt(path_cpt, cfg: DictConfig):
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
        training(path_cpt, cfg=cfg_xp, idx_cv=idx_cv, date=date_time)
    logger_console.info("finished")



def model_dim():
    model = PatchEmbed(res=True)
    print(model)

def load_probas():
    checkpoint = torch.load("./misc/checkpoints/adapt_TS/probas/2025-02-08_19h59_adapt-TreeSat_cv_0_probas_targets.pth")
    probas = checkpoint["probas"]
    targets = checkpoint["targets"]
    
    threshold = 0.5

    # 将概率值转换为二元预测标签
    preds = (probas >= threshold).float()

    # 计算多标签准确率
    correct = (preds == targets).float()
    accuracy = correct.mean().item()

    print(f'Multilabel Accuracy: {accuracy:.4f}')

@hydra.main(version_base="1.2", config_path="configs", config_name="TS_ADAPT.yaml")
def test_cpt(cfg: DictConfig):

    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )

    torch.set_float32_matmul_precision("high")
    logger_console = logging.getLogger(__name__)
    logger_console.info("start")
    date = datetime.now().strftime("%Y-%m-%d_%Hh%M")

    for idx_cv in range(cfg.cv):
    
        if cfg.get("seed"):
            pl.seed_everything(cfg.seed, workers=True)

        if cfg.logger.mode == "offline":
            os.environ["WANDB_MODE"] = "offline"
            os.environ["WANDB_DIR"] = cfg.paths.logs

        # callbacks
        callbacks = []
        name = f"{date}_{cfg.logger.name}_cv_{idx_cv}"
        model = ADAPT(cfg=cfg, name=name, stage="classification")

        print(os.getcwd())
        checkpoint = torch.load(cfg.path_cpt) # 100 epochs pre-trained model before classification
        print(checkpoint.keys())
        model.load_state_dict(checkpoint["state_dict"])

        ######### logger #########
        if cfg.log:
            logger = hydra.utils.instantiate(cfg.logger, id=name)
            callbacks.extend(
                [LearningRateMonitor(logging_interval="epoch"), SizeDatamodule(cfg.log)]
            )
        else:
            logger = None

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
                max_epochs=80,
                callbacks=callbacks3,
            )

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
    
    logger_console.info("finished")
    

if __name__ == "__main__":
    # test_transform_on_1d_tensor()
    path_cpt = "./misc/checkpoints/adapt_TS/probas/2025-03-09_13h00_100E_0.1_avg_anchor-s2_cv_0_probas_targets.pth" #100E 0.1 avg
    test_cpt()