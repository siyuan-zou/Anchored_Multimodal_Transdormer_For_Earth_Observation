"""ADAPT model implementation."""

# pylint:disable=C0303

import logging
from typing import Any
from collections import OrderedDict
import os

import hydra
import torch
import torchmetrics
import numpy as np
from torch import nn
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT


from src.utils.training.losses import InfoNCEAnchor, InfoNCE
from src.utils.training.cos_sch import CosineWarmupScheduler
from src.utils.evaluation.torch_evaluation import Evaluation
from src.models.modules.backbone import clone, Encoders, ProjectionHead, Encoders_TS, Encoders_PT


class ADAPT(pl.LightningModule):
    """
    AnchoreD multimodAl Transformer.
    """

    def __init__(
        self,
        cfg,
        name: str = "",
        stage: str = "classification",
    ) -> None:
        """ADAPT model.

        Parameters
        ----------
        cfg : Dict
            config for the model.
        name: str.
            name of the xp
        stage: str.
            stage in the training.
        """
        super().__init__()
        self.save_hyperparameters()

        self.num_classes = 15 if "TreeSAT" in cfg["model"]["encoders"]["type"] else 2
        self.stage = stage
        self.name = name
        self.cfg = cfg
        self.modalities = cfg["multimodal"]["modalities"]
        self.criterion = None

        self.test_save_path = cfg["paths"]["misc"] + "/checkpoints/adapt_TS/probas" # 存储路径

        ### model
        
        ###### Encoders ######
        if "TreeSAT" in cfg["model"]["encoders"]["type"]:
            self.encoders = Encoders_TS(cfg=cfg)
        elif "PASTIS" in cfg["model"]["encoders"]["type"]:
            self.encoders = Encoders_PT(cfg=cfg)
        else:
            self.encoders = Encoders(cfg=cfg)
        ###### Transformer ######

        d_model = cfg["model"]["transformer"]["d_model"]
        scale = d_model**-0.5
        attention = hydra.utils.instantiate(cfg.model.transformer)
        self.blocks = clone(attention, self.cfg["model"]["transformer"]["n_blocks"])
        self.cls_embedding = nn.Parameter(scale * torch.randn(1, d_model))

        ###### Classifier ######

        self.classifier = nn.Sequential(
            OrderedDict(
                [
                    ("linear", nn.Linear(d_model, self.num_classes)),
                    (
                        "dropout",
                        nn.Dropout(p=self.cfg["model"]["transformer"]["dropout"]),
                    ),
                ]
            )
        )


        ###### outputs & metrics ######

        self.evaluation = Evaluation(num_classes=self.num_classes, logger=self.logger)
        self.test_epoch_outputs = []

        ### Metrics ######
        metric_params = (
            {
                "task": "multilabel",
                "num_labels": self.num_classes,  # 多标签任务
                "top_k": 1
            } if self.num_classes > 2 else {
                "task": "binary",
                "num_classes": self.num_classes,
                "top_k": 1
            }
        )

        ### Metric objects for calculating and averaging accuracy across batches ######
        self.accuracy = torchmetrics.Accuracy(**metric_params)
        self.f1 = torchmetrics.F1Score(**metric_params)

    def setup(self, stage: str) -> None:
        """Setup the model for training/validation/testing."""
        ## model - anchoring
        if self.stage == "anchoring":
            self.criterion = InfoNCEAnchor(self.cfg)
            ### freeze the transformer
            for param in self.blocks.parameters():
                param.requires_grad = False
            self.blocks.eval()
            self.cls_embedding.requires_grad = False
            # if self.cfg["model"]["encoders"]["freeze"]:
            #     logging.info("Freeze the encoders")
            #     for block in self.encoders.encoders.keys():
            #         for sub_block in self.encoders.encoders[block]:
            #             ### The ecnoders are frozen IF the net is considered frozen
            #             if not isinstance(sub_block, ProjectionHead):
            #                 for param in sub_block.parameters():
            #                     param.requires_grad = False
            #                 sub_block.eval()

        ## criterion - contrastive
        if self.stage == "contrastive":
            self.criterion = InfoNCE(self.cfg)
            ### unfreeze the transformer
            for param in self.blocks.parameters():
                param.requires_grad = True
            self.blocks.train()
            self.cls_embedding.requires_grad = True

            # ### freeze the backbone
            # if self.cfg["model"]["contrastive_loss"]["freeze"]:
            for param in self.encoders.parameters():
                param.requires_grad = False
            self.encoders.eval()
            # if self.cfg["model"]["encoders"]["freeze"]:
            #     logging.info("Freeze the encoders")
            #     for block in self.encoders.encoders.keys():
            #         for sub_block in self.encoders.encoders[block]:
            #             ### The ecnoders are frozen IF the net is considered frozen
            #             if not isinstance(sub_block, ProjectionHead):
            #                 for param in sub_block.parameters():
            #                     param.requires_grad = False
            #                 sub_block.eval()

        ## criterion - classification
        elif self.stage == "classification" and stage == "fit":
            ## Criterion
            if any(enc in self.cfg["model"]["encoders"]["type"] for enc in ["TreeSAT", "PASTIS"]):
                self.criterion = nn.BCEWithLogitsLoss()
            else:
                self.criterion = nn.CrossEntropyLoss()

            if "TreeSAT" in self.cfg["model"]["encoders"]["type"]:
                for block in self.encoders.encoders.keys():
                    for param in self.encoders.encoders[block].parameters():
                        param.requires_grad = True
                    self.encoders.encoders[block].eval()
            else:
                for block in self.encoders.encoders.keys():
                    for sub_block in self.encoders.encoders[block]:
                        for param in sub_block.parameters():
                            param.requires_grad = True
                        sub_block.eval()
        return super().setup(stage)

    def _shared_log_step(
        self,
        mode: str,
        metrics: dict,
        on_step: bool = True,
        on_epoch: bool = False,
    ):
        """Shared log step."""
        for key, value in metrics.items():
            self.log(f"{mode}_{key}", value, on_step=on_step, on_epoch=on_epoch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = {}
        # the mask just need to be in the x
        B = x[self.modalities[0]].shape[0]
        x_representation = self.encoders(x)
        device = x_representation[self.modalities[0]].device

        # print("The device is: ", device)

        ### stage classif or contrastive
        if self.stage in ["contrastive", "classification"]:
            mask = np.zeros(
                (B, len(self.modalities) + 1), dtype=bool
            )  # + 1 for the CLS TOKEN
            cls_batch = torch.zeros(
                x_representation[self.modalities[0]].shape[0],
                1,
                x_representation[self.modalities[0]].shape[-1],
            ).to(device)
            x_u = [self.cls_embedding.to(device) + cls_batch]

            for idx, modality in enumerate(self.cfg["multimodal"]["modalities"]):
                x_u.append(x_representation[modality].unsqueeze(1))
                mask[:, idx + 1] = (
                    np.char.find(np.array(x[f"id_{modality}"]), "missing") != -1
                )
            mask = torch.Tensor(mask).bool().to(device)
            x_u = torch.cat(x_u, dim=1)
            # 过 transformer
            for block in self.blocks:
                x_u = block(x_u, mask=mask, softmax_dim=-1)
            cls = x_u[:, 0, :]
            outputs["cls"] = cls
        ## stage : anchoring
        else:
            for idx, modality in enumerate(self.cfg["multimodal"]["modalities"]):
                outputs[modality] = x_representation[modality]
        return outputs

    def step(self, batch):
        """Perform a step in the train/test/val loop."""
        x = batch
        outputs = {}
        modality_0 = self.modalities[0]
        if self.stage == "anchoring":
            outputs = self.forward(x)
        elif self.stage == "contrastive":
            x_1, x_2 = x[0], x[1]
            outputs["view_1"] = self.forward(x_1)["cls"]
            outputs["view_2"] = self.forward(x_2)["cls"]
            outputs["targets"] = x_1["label"]
            outputs[f"id_{modality_0}"] = x_1[f"id_{modality_0}"]
        elif self.stage == "classification":
            outputs = self.forward(x)
            outputs["logits"] = self.classifier(outputs["cls"])
            outputs["targets"] = x["label"]
            outputs[f"id_{modality_0}"] = x[f"id_{modality_0}"]
        return outputs

    def training_step(self, batch, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        outputs = self.step(batch)
        if self.stage == "anchoring":
            loss = self.criterion(outputs=outputs, current_epoch=self.current_epoch)
            # with torch.autograd.detect_anomaly():
            #     loss.backward()
            if hasattr(self.criterion, "temperature"):
                self.log(
                    "temperature_epoch",
                    self.criterion.temperature,
                    on_step=False,
                    on_epoch=True,
                )
            metrics = res = {"loss": loss}
            train_name = "train_anchoring"
        elif self.stage == "contrastive":
            loss = self.criterion(
                outputs["view_1"], outputs["view_2"], current_epoch=self.current_epoch
            )
            # with torch.autograd.detect_anomaly():
            #     loss.backward()
            metrics = {"loss": loss}
            res = {"loss": loss, "targets": outputs["targets"]}
            if hasattr(self.criterion, "temperature"):
                self.log(
                    "temperature_epoch",
                    self.criterion.temperature,
                    on_step=False,
                    on_epoch=True,
                )
            train_name = "train_contrastive"
        else:
            if any(enc in self.cfg["model"]["encoders"]["type"] for enc in ["TreeSAT", "PASTIS"]):
                y_true = outputs["targets"].float()
                # print("type y_true", type(y_true))
                # print("type y_true element", type(y_true[0]))  # targets 是多热编码
                loss = self.criterion(outputs["logits"], y_true)  # 使用 BCEWithLogitsLoss

                # 计算概率
                probas = torch.sigmoid(outputs["logits"])  # 用 sigmoid 获取每类的概率

                # 二值化预测：选择概率大于 0.5 的类作为预测
                y_pred = (probas > 0.5).float()

                # 计算 TPR 和 TNR
                tpr = self.accuracy(y_pred, y_true)
                tnr = self.f1(y_pred, y_true)
                
                # 日志记录和返回结果
                metrics = {"loss": loss, "tpr": tpr, "tnr": tnr}
                res = {"loss": loss, "targets": y_true, "probas": probas}
                train_name = "train_classification"
            else:
                y_true = outputs["targets"]
                loss = self.criterion(outputs["logits"], outputs["targets"])
                probas = torch.nn.functional.softmax(outputs["logits"], dim=1)
                y_pred = torch.argmax(probas, dim=1)
                y_true = torch.argmax(y_true, dim=1)

                tpr = self.accuracy(y_pred, y_true)
                tnr = self.f1(y_pred, y_true)
                metrics = {"loss": loss, "tpr": tpr, "tnr": tnr}
                res = {"loss": loss, "targets": outputs["targets"], "probas": probas}
                train_name = "train_classification"
        self._shared_log_step(
            train_name,
            metrics=metrics,
            on_step=False,
            on_epoch=True,
        )
        return res

    def validation_step(self, batch, batch_idx):
        """validation step."""
        outputs = self.step(batch)
        if self.stage == "contrastive":
            loss = self.criterion(
                outputs["view_1"], outputs["view_2"], current_epoch=self.current_epoch
            )
            metrics = {"loss": loss}
            res = {"loss": loss, "targets": outputs["targets"]}
            val_name = "val_contrastive"
        elif self.stage == "anchoring":
            loss = self.criterion(outputs=outputs, current_epoch=self.current_epoch)
            metrics = {"loss": loss}
            res = {"loss": loss}
            val_name = "val_anchoring"
        elif self.stage == "classification":
            if any(enc in self.cfg["model"]["encoders"]["type"] for enc in ["TreeSAT", "PASTIS"]):
                y_true = outputs["targets"].float()  # targets 是多热编码
                # print("type y_true", type(y_true))
                # print("type y_true element", type(y_true[0]))
                loss = self.criterion(outputs["logits"], y_true)  # 使用 BCEWithLogitsLoss

                # 计算概率
                probas = torch.sigmoid(outputs["logits"])  # 用 sigmoid 获取每类的概率

                # 二值化预测：选择概率大于 0.5 的类作为预测
                y_pred = (probas > 0.5).float()

                # 计算 TPR 和 TNR
                tpr = self.accuracy(y_pred, y_true)
                tnr = self.f1(y_pred, y_true)
                
                # 日志记录和返回结果
                metrics = {"loss": loss, "tpr": tpr, "tnr": tnr}
                res = {"loss": loss, "targets": y_true, "probas": probas}
                val_name = "val_classification"
            else:
                y_true = outputs["targets"]
                loss = self.criterion(outputs["logits"], outputs["targets"])
                probas = torch.nn.functional.softmax(outputs["logits"], dim=1)
                y_pred = torch.argmax(probas, dim=1)
                y_true = torch.argmax(y_true, dim=1)

                tpr = self.accuracy(y_pred, y_true)
                tnr = self.f1(y_pred, y_true)

                metrics = {"loss": loss, "tpr": tpr, "tnr": tnr}
                res = {"loss": loss, "targets": y_true, "probas": probas}
                val_name = "val_classification"
        self._shared_log_step(val_name, metrics=metrics, on_step=True, on_epoch=True)
        return res

    def test_step(self, batch, batch_idx):
        """test step."""
        if any(enc in self.cfg["model"]["encoders"]["type"] for enc in ["TreeSAT", "PASTIS"]):
            outputs = self.step(batch)

            logits = outputs["logits"]
            
            # For BCEWithLogitsLoss, use sigmoid instead of softmax
            probas = torch.sigmoid(logits)  # Get probabilities in [0, 1]

            # Prepare results for logging or evaluation
            res = {"targets": outputs["targets"], "probas": probas}

            # print("Targets shape: ", outputs["targets"].shape)
            # print("Probas shape: ", probas.shape)
            # print("Probas: ", probas)
            # print("Targets: ", outputs["targets"])

            # Save the results for later aggregation in `on_test_epoch_end`
            self.test_epoch_outputs.append(res)

            # print("Test step done.")
        else:
            outputs = self.step(batch)
            logits = outputs["logits"]
            ## add the emb ?
            probas = torch.nn.functional.softmax(logits, dim=1)
            res = {"targets": outputs["targets"], "probas": probas}
            self.test_epoch_outputs.append(res)


    def on_test_epoch_end(self):

        outputs = self.test_epoch_outputs
        
        keys = outputs[0].keys()
        final_outputs = {}
        for x in outputs:
            for key in keys:
                if key not in final_outputs:
                    final_outputs[key] = []
                final_outputs[key].append(x[key])
        for key in keys:
            final_outputs[key] = torch.cat(final_outputs[key], dim=0)
        
        for key, value in final_outputs.items():
            print(key, value.shape)

        # Save the results to disk
        print("Saving test results to disk...")
        if os.path.exists(self.test_save_path):
            self.test_save_path = os.path.join(self.test_save_path, f"{self.name}_probas_targets.pth")
        torch.save(final_outputs, self.test_save_path)
        print(f"Test results saved to {self.test_save_path}")
            
        
        res = self.evaluation(
            final_outputs["probas"], final_outputs["targets"], prefix=""
        )
        self._shared_log_step("test", metrics=res, on_step=False, on_epoch=True)


    def configure_optimizers(self):
        """Configure optimizers."""
        if self.stage == "anchoring":
            lr = self.cfg["model"]["anchoring_loss"]["lr"]
            wd = self.cfg["model"]["anchoring_loss"]["weight_decay"]
            epoch_warmup = self.cfg["model"]["anchoring_loss"]["warmup"]
            max_epochs = self.cfg["model"]["anchoring_loss"]["max_epochs"]
            min_lr = self.cfg["model"]["anchoring_loss"]["min_lr"]
            sch = self.cfg["model"]["anchoring_loss"]["sch"]
        elif self.stage == "contrastive":
            lr = self.cfg["model"]["contrastive_loss"]["lr"]
            wd = self.cfg["model"]["contrastive_loss"]["weight_decay"]
            epoch_warmup = self.cfg["model"]["contrastive_loss"]["warmup"]
            max_epochs = self.cfg["model"]["contrastive_loss"]["max_epochs"]
            min_lr = self.cfg["model"]["contrastive_loss"]["min_lr"]
            sch = self.cfg["model"]["contrastive_loss"]["sch"]
        else:
            lr = self.cfg["model"]["supervised_loss"]["lr"]
            wd = self.cfg["model"]["supervised_loss"]["weight_decay"]
            epoch_warmup = self.cfg["model"]["supervised_loss"]["warmup"]
            max_epochs = self.cfg["model"]["supervised_loss"]["max_epochs"]
            min_lr = self.cfg["model"]["supervised_loss"]["min_lr"]
            sch = self.cfg["model"]["supervised_loss"]["sch"]
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=wd)
        configuration = {"optimizer": optimizer}
        if sch:
            scheduler = CosineWarmupScheduler(
                optimizer,
                epoch_warmup=epoch_warmup,
                max_epoch=max_epochs,
                min_lr=min_lr,
            )
            configuration = {
                "optimizer": optimizer,
                "lr_scheduler": scheduler,
            }
        return configuration
