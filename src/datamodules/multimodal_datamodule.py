"""Multimodal datamodule."""

import os
import pandas as pd
from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule

from src.datamodules.datasets.stressid import StressID
from src.datamodules.datasets.TreeSAT import TreeSAT
from src.datamodules.datasets.Pastis import PASTIS
from src.utils.training.splits import create_splits
import time


class MultimodaDataModule(LightningDataModule):
    """Multimodal datamodule class"""

    def __init__(self, cfg: dict, stage: str = "anchoring", cv: int = 0) -> None:
        """Initialization

        Parameters
        ----------
        cfg : dict
            cfg dict.
        """
        super().__init__()
        self.cfg = cfg
        self.cv = cv
        self.stage = stage
        self.train_set = None
        self.val_set = None
        self.test_set = None

    def setup(self, stage=None):
        """Setup

        Parameterss
        ----------
        stage : str, optional
            Stage, by default None
        """

        all_launches = pd.read_csv(
            os.path.join(
                self.cfg["paths"]["data"], self.cfg["multimodal"]["path_labels"]
            ),
            index_col=0,
            sep=",",
        )
        folds = create_splits(
            ids=all_launches.index, cv=self.cfg.cv, seed=self.cfg.seed
        )
        if stage == "fit" or stage is None:
            self.train_set = StressID(
                cfg=self.cfg, stage=self.stage, ids=folds[0][self.cv]
            )
            self.val_set = StressID(
                cfg=self.cfg, stage=self.stage, ids=folds[1][self.cv]
            )

        if stage == "test":
            self.test_set = StressID(
                cfg=self.cfg, stage=self.stage, ids=folds[2][self.cv]
            )

    def train_dataloader(self) -> DataLoader:
        collate = self.train_set.collate if hasattr(self.train_set, "collate") else None
        return DataLoader(
            shuffle=True,
            dataset=self.train_set,
            batch_size=self.cfg["multimodal"]["hyperparams"]["batch_size"],
            num_workers=self.cfg["multimodal"]["hyperparams"]["num_workers"],
            collate_fn=collate,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        collate = self.train_set.collate if hasattr(self.train_set, "collate") else None
        return DataLoader(
            shuffle=False,
            dataset=self.val_set,
            batch_size=self.cfg["multimodal"]["hyperparams"]["batch_size"],
            num_workers=self.cfg["multimodal"]["hyperparams"]["num_workers"],
            collate_fn=collate,
            drop_last=False,
        )

    def test_dataloader(self) -> DataLoader:
        collate = self.train_set.collate if hasattr(self.train_set, "collate") else None
        return DataLoader(
            shuffle=False,
            dataset=self.test_set,
            collate_fn=collate,
            batch_size=self.cfg["multimodal"]["hyperparams"]["batch_size"],
            num_workers=self.cfg["multimodal"]["hyperparams"]["num_workers"],
        )
    
    def get_train_set(self):
        return self.train_set

class TSDataModule(LightningDataModule):
    """TreeSAT AI datamodule class"""

    def __init__(self, cfg: dict, stage: str = "anchoring", cv: int = 0) -> None:
        """Initialization

        Parameters
        ----------
        cfg : dict
            cfg dict.
        """
        super().__init__()
        self.cfg = cfg
        self.cv = cv
        self.stage = stage
        self.train_set = None
        self.val_set = None
        self.test_set = None

    def setup(self, stage=None):
        """Setup the datamodule.
        Args:
            stage (str): stage of the datamodule
                Is be one of "fit" or "test" or None
        """
        print("Stage", stage)
        start_time = time.time()
        if stage == "fit" or stage is None:
            self.train_set = TreeSAT(
                cfg=self.cfg, stage=self.stage, split="train"
            )
            self.val_set = TreeSAT(
                cfg=self.cfg, stage=self.stage, split="val"
            )
            print(f"Train dataset size: {len(self.train_set)}")
            print(f"Val dataset size: {len(self.val_set)}")

            # for testing purposes
            return self.train_set
        if stage == "test":
            self.test_set = TreeSAT(
                cfg=self.cfg, stage=self.stage, split="test"
            )
            print(f"Test dataset size: {len(self.test_set)}")
        end_time = time.time()
        print(f"Setup took {(end_time - start_time):.2f} seconds")

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.cfg["multimodal"]["hyperparams"]["batch_size"],
            num_workers=self.cfg["multimodal"]["hyperparams"]["num_workers"],
            shuffle=True,
            pin_memory=False,
            drop_last=True,
            collate_fn=self.train_set.collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set,
            batch_size=self.cfg["multimodal"]["hyperparams"]["batch_size"],
            num_workers=self.cfg["multimodal"]["hyperparams"]["num_workers"],
            shuffle=False,
            pin_memory=False,
            drop_last=True,
            collate_fn=self.val_set.collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_set,
            batch_size=self.cfg["multimodal"]["hyperparams"]["batch_size"],
            num_workers=self.cfg["multimodal"]["hyperparams"]["num_workers"],
            shuffle=False,
            pin_memory=False,
            collate_fn=self.test_set.collate_fn,
        )



class PTDataModule(LightningDataModule):
    """TreeSAT AI datamodule class"""

    def __init__(self, cfg: dict, stage: str = "anchoring", cv: int = 0) -> None:
        """Initialization

        Parameters
        ----------
        cfg : dict
            cfg dict.
        """
        super().__init__()
        self.cfg = cfg
        self.cv = cv
        self.stage = stage
        self.train_set = None
        self.val_set = None
        self.test_set = None

    def setup(self, stage=None):
        """Setup the datamodule.
        Args:
            stage (str): stage of the datamodule
                Is be one of "fit" or "test" or None
        """
        print("Stage", stage)
        start_time = time.time()
        if stage == "fit" or stage is None:
            self.train_set = PASTIS(
                cfg=self.cfg, 
                stage=self.stage, 
                split="train", 
                folds=self.cfg["multimodal"]["train_dataset"]["folds"],
                reference_date="2018-09-01",
                nb_split = self.cfg["multimodal"]["nb_split"],
                num_classes = self.cfg["multimodal"]["num_classes"],
            )
            self.val_set = PASTIS(
                cfg=self.cfg, 
                stage=self.stage, 
                split="val", 
                folds=self.cfg["multimodal"]["val_dataset"]["folds"],
                reference_date="2018-09-01",
                nb_split = self.cfg["multimodal"]["nb_split"],
                num_classes = self.cfg["multimodal"]["num_classes"],
            )
            print(f"Train dataset size: {len(self.train_set)}")
            print(f"Val dataset size: {len(self.val_set)}")

            # for testing purposes
            return self.train_set
        if stage == "test":
            self.test_set = PASTIS(
                cfg=self.cfg, 
                stage=self.stage, 
                split="test", 
                folds=self.cfg["multimodal"]["test_dataset"]["folds"],
                reference_date="2018-09-01",
                nb_split = self.cfg["multimodal"]["nb_split"],
                num_classes = self.cfg["multimodal"]["num_classes"],
            )
            print(f"Test dataset size: {len(self.test_set)}")
        end_time = time.time()
        print(f"Setup took {(end_time - start_time):.2f} seconds")

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.cfg["multimodal"]["hyperparams"]["batch_size"],
            num_workers=self.cfg["multimodal"]["hyperparams"]["num_workers"],
            shuffle=True,
            pin_memory=False,
            drop_last=True,
            collate_fn=self.train_set.collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set,
            batch_size=self.cfg["multimodal"]["hyperparams"]["batch_size"],
            num_workers=self.cfg["multimodal"]["hyperparams"]["num_workers"],
            shuffle=False,
            pin_memory=False,
            drop_last=True,
            collate_fn=self.val_set.collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_set,
            batch_size=self.cfg["multimodal"]["hyperparams"]["batch_size"],
            num_workers=self.cfg["multimodal"]["hyperparams"]["num_workers"],
            shuffle=False,
            pin_memory=False,
            collate_fn=self.test_set.collate_fn,
        )
