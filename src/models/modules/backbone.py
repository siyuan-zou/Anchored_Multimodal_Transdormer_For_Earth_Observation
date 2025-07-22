"""Modality-specific encoders."""

# pylint:disable=C0303

import logging
from collections import OrderedDict


import os
import copy
import torch
from torch import nn

from assets.byol_a.models import AudioNTT2020
from src.models.modules.make_ts import ResnetTS, Inception
from src.models.modules.make_vision import make_hiera, make_hiera_image
from src.models.encoders.patch_embeddings import PatchEmbed, PatchEmbedPastis
from src.models.encoders.ltae import LTAE2d, LTAE2dPastis
from src.models.modules.Fine_tuning import Fine

def clone(module, number_of_copies):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(number_of_copies)])


def default(val, default):
    """Get default."""
    return default if val is None else val


class Encoders(nn.Module):
    """Projection module."""

    def __init__(self, cfg) -> None:
        """Intialization of the projection module."""
        super().__init__()
        self.cfg = cfg
        self.encoders = nn.ModuleDict()
        self.modalities = self.cfg["multimodal"]["modalities"]
        for modality in self.modalities:
            self.encoders[modality] = self.make_encoder(modality)

    def forward(self, x) -> dict:
        """Forward pass of the projection."""
        features = {}
        for modality in self.modalities:
            features[modality] = self.encoders[modality](x[modality])
        return features

    def make_encoder(self, modality) -> nn.Module:
        """Make encoder for a given modality

        Parameters
        ----------
        modality : str
            Modality of the encoder.

        Returns
        -------
        nn.Module
            Encoder module.
        """
        out_features = self.cfg["model"]["transformer"]["d_model"]
        encoder = []
        ### VIDEO ###
        if modality == "VIDEO":
            if "hiera" in self.cfg["model"]["encoders"]["type"]:
                logging.info("Using hiera")
                directory = (
                    "/gpfswork/rech/oyr/urt67oj/misc/ext/hiera"
                    if self.cfg["paths"]["___config_name___"] == "jz"
                    else None
                )
                model, in_features = make_hiera(
                    freeze=self.cfg["model"]["encoders"]["freeze"], directory=directory
                )
                encoder.append(("encoder", model))
            else:
                model = nn.Identity()  # featurization is already done
                in_features = 768
        ### AUDIO ###
        elif modality == "AUDIO":
            if "byola" in self.cfg["model"]["encoders"]["type"]:
                logging.info("Using byola")
                model = AudioNTT2020(d=self.cfg["model"]["encoders"]["n_dims_audio"])
                device = (
                    torch.device("cuda")
                    if torch.cuda.is_available()
                    else torch.device("cpu")
                )
                if self.cfg["model"]["encoders"]["pretrained"]:
                    model.load_weight(
                        os.path.join(
                            self.cfg["paths"]["misc"],
                            self.cfg["model"]["weight_path_byola"],
                        ),
                        device=device,
                    )
                in_features = model.fc[0].in_features
                model.fc = nn.Identity()
                encoder.append(("encoder", model))
            else:
                raise ValueError(f"Unknown encoder for modality {modality}")
        ### IMAGE ###
        elif modality == "IMAGE":
            if "hiera-image" in self.cfg["model"]["encoders"]["type"]:
                logging.info("Using hiera image")
                directory = (
                    "/gpfswork/rech/oyr/urt67oj/misc/ext/hiera"
                    if self.cfg["paths"]["___config_name___"] == "jz"
                    else None
                )
                model, in_features = make_hiera_image(freeze=False, directory=directory)
                encoder.append(("encoder", model))
            else:
                raise ValueError(f"Unknown encoder for modality {modality}")
        ### EDA, ECG, RR ###
        elif modality in ["EDA", "ECG", "RR"]:
            if "resnet-ts" in self.cfg["model"]["encoders"]["type"]:
                logging.info("Using resnet-ts")
                model = ResnetTS(
                    hidden_channels=self.cfg["model"]["encoders"]["ts_setting"][
                        "hidden"
                    ],
                    kernel_size=self.cfg["model"]["encoders"]["ts_setting"]["kernel"],
                )
                model.classifier = nn.Identity()
                in_features = model.output_features
                encoder.append(("encoder", model))
            elif "inception-ts" in self.cfg["model"]["encoders"]["type"]:
                model = Inception(
                    hidden_channels=self.cfg["model"]["encoders"]["ts_setting"][
                        "hidden"
                    ],
                    kernel_size=self.cfg["model"]["encoders"]["ts_setting"]["kernel"],
                    bottleneck=self.cfg["model"]["encoders"]["ts_setting"][
                        "bottleneck"
                    ],
                    depth=self.cfg["model"]["encoders"]["ts_setting"]["depth"],
                    rezero=self.cfg["model"]["encoders"]["ts_setting"]["rezero"],
                )
                model.classifier = nn.Identity()
                in_features = model.output_features
                encoder.append(("encoder", model))
            else:
                raise ValueError(f"Unknown encoder for modality {modality}")
        else:
            raise ValueError(f"Unknown modality: {modality}")
        if self.cfg["model"]["encoders"]["projection"]:
            encoder.append(
                (
                    "projection",
                    ProjectionHead(
                        in_features=in_features,
                        out_features=out_features,
                    ),
                )
            )
        encoder = nn.Sequential(OrderedDict(encoder))
        return encoder

class Encoders_TS(nn.Module):
    """Projection module for TreeSAT."""

    def __init__(self, cfg) -> None:
        """Intialization of the projection module."""
        super().__init__()
        self.cfg = cfg
        self.encoders = nn.ModuleDict()
        self.modalities = self.cfg["multimodal"]["modalities"]

        self.num_patches = {
            modality: self.cfg["model"]["encoders"]["num_patches"] for modality in self.modalities if modality != "aerial"
            } | {
                "aerial": int((300 / self.cfg["model"]["encoders"]["aerial"]["patch_size"]) ** 2)
            }

        self.embed_dim = self.cfg["model"]["encoders"]["embed_dim"]
        
        self.pooling_method = self.cfg["model"]["encoders"]["projection_method"]

        for modality in self.modalities:
            self.encoders[modality] = self.make_encoder(modality)
        
        
        
        # self.out_features = self.cfg["model"]["transformer"]["d_model"]
        # # 创建一个占位符 Linear 层
        # self.projection = None
        # self.projection_TS = None


    def forward(self, x) -> dict:
        """Forward pass of the projection."""
        features = {}
        for modality in self.modalities:
            if modality == "aerial":
                features[modality] = self.encoders[modality](x[modality])
            elif modality == "s1-asc" or modality == "s2":
                for name, module in self.encoders[modality]._modules.items():
                    # print(name)
                    if name == "encoder":
                        features[modality] = module(x[modality], batch_positions=x[modality + "_dates"])
                    else:
                        features[modality] = module(features[modality])
                # features[modality] = self.encoders[modality]._modules["projection"](self.encoders[modality]._modules["encoder"](x[modality], batch_positions=x[modality + "_dates"]))
        return features
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # features = {}
        # for modality in self.modalities:
        #     if modality == "aerial":
        #         if self.cfg["model"]["encoders"]["projection"]:
                    
        #             output = self.encoders[modality](x[modality])
        #             # print(f"Encoder output type: {type(output)}")
        #             if isinstance(output, tuple):
        #                 feature = output[0]  # 只提取第一个 Tensor
        #             else:
        #                 feature = output
        #             # 只取第一个输出,剩下的输出是pooling的indexes
        #             # print(f"Aerial feature shape before flatten: {feature.shape}") #正常情况下是 [batch_size, num_patches, embed_dim]
                    
        #             # batch_size = feature.size(0)
        #             if feature.dim() == 2:  # [batch_size, num_patches, embed_dim] test mode下batch_size=1
        #                 feature = feature.unsqueeze(0)

        #             feature = feature.flatten(start_dim=1) # Flatten all but batch dimension
        #             # print(f"Aerial feature shape: {feature.shape}")
                    
        #             # 动态计算 in_features
        #             in_features = feature.size(1)

        #             # if self.projection is None:  # 只创建一次
        #             #     self.projection = ProjectionHead(
        #             #         in_features=in_features,
        #             #         out_features=self.out_features,
        #             #     ).to(device)

        #             features[modality] = self.projection(feature.to(device))
        #         else:
        #             features[modality] = self.encoders[modality](x[modality][0].to(device))
        #     elif modality == "s1-asc" or modality == "s2":
        #         # print(self.encoders[modality])
        #         output = self.encoders[modality](x[modality].to(device), batch_positions=x[modality + "_dates"].to(device))
        #         # print(f"{modality} output: {output}")
        #         output = torch.mean(output, dim=(2, 3), keepdim=True).squeeze() #听取chatgpt的建议，土地覆盖分类任务

        #         if output.dim() == 1:
        #             output = output.unsqueeze(0) # [batch_size, embed_dim] test mode下batch_size=1
        #         # output.view(output.shape[0], output.shape[1], -1).permute(0, 2, 1) # 

        #         # print(f"Output shape: {output.shape}")
        #         # print(f"in features", output.size(1))
        #         # print(f"{modality} output: {output}")

        #         if self.cfg["model"]["encoders"]["projection"]:
        #             if self.projection_TS is None:
        #                 self.projection_TS = ProjectionHead(
        #                     in_features=output.size(1),
        #                     out_features=self.out_features,
        #                 ).to(device)
        #             features[modality] = self.projection_TS(output)
        #             # print(f"{modality} feature after projection: {features[modality]}")
        #         else:
        #             features[modality] = output
        # return features

    def make_encoder(self, modality) -> nn.Module:
        """Make encoder for a given modality

        Parameters
        ----------
        modality : str
            Modality of the encoder.

        Returns
        -------
        nn.Module
            Encoder module.
        """
        out_features = self.cfg["model"]["transformer"]["d_model"]
        encoder = []
        ### aerial ###
        if modality == "aerial":
            model = PatchEmbed(
                patch_size=self.cfg["model"]["encoders"]["aerial"]["patch_size"], 
                in_chans=self.cfg["model"]["encoders"]["aerial"]["in_chans"],
                embed_dim=self.cfg["model"]["encoders"]["aerial"]["embed_dim"],
                bias=self.cfg["model"]["encoders"]["aerial"]["bias"],
                res=self.cfg["model"]["encoders"]["aerial"]["res"],
                gp_norm=self.cfg["model"]["encoders"]["aerial"]["gp_norm"])
            encoder.append(("encoder", model))
        ### s1-asc ###
        elif modality == "s1-asc" or modality == "s2":
            model = LTAE2d(
                in_channels=self.cfg["model"]["encoders"][modality]["in_channels"],
                n_head=self.cfg["model"]["encoders"][modality]["n_head"],
                d_k=self.cfg["model"]["encoders"][modality]["d_k"],
                mlp=self.cfg["model"]["encoders"][modality]["mlp"],
                mlp_in=self.cfg["model"]["encoders"][modality]["mlp_in"],
                dropout=self.cfg["model"]["encoders"][modality]["dropout"],
                T=self.cfg["model"]["encoders"][modality]["T"],
                in_norm=self.cfg["model"]["encoders"][modality]["in_norm"],
                positional_encoding=self.cfg["model"]["encoders"][modality]["positional_encoding"]
            )

            encoder.append(("encoder", model))
        else:
            raise ValueError(f"Unknown modality: {modality}")

        if self.cfg["model"]["encoders"]["projection"]:
            if self.cfg["model"]["encoders"]["projection_method"] == "pooling":
                print(modality, self.num_patches[modality])
                encoder.append(
                    (
                        "Pooling", 
                        Fine(
                            self.embed_dim,
                            self.num_patches[modality],
                            [],
                            0.2,
                            0, # No classification head
                            self.pooling_method
                            )
                    )
                )

                encoder.append(
                    (
                        "projection",
                        ProjectionHead(
                            in_features=self.embed_dim if self.pooling_method not in ["avg_f", "max_f"] else self.num_patches[modality],
                            out_features=out_features,
                        )
                    )
                )
            
            if self.cfg["model"]["encoders"]["projection_method"] == "full":

                encoder.append(
                    (
                        "projection",
                        ProjectionHead(
                            in_features=self.num_patches[modality] * self.embed_dim,
                            out_features=out_features,
                        ),
                    )
                )

        encoder = nn.Sequential(OrderedDict(encoder))
        return encoder
    

class Encoders_PT(nn.Module):
    """Projection module for PASTIS-HD."""

    def __init__(self, cfg) -> None:
        """Intialization of the projection module."""
        super().__init__()
        self.cfg = cfg
        self.encoders = nn.ModuleDict()
        self.modalities = self.cfg["multimodal"]["modalities"]
        self.num_patches = self.cfg["model"]["num_patches"] # 默认1024

        self.embed_dim = self.cfg["model"]["encoders"]["embed_dim"]
        
        self.pooling_method = self.cfg["model"]["encoders"]["projection_method"]

        for modality in self.modalities:
            self.encoders[modality] = self.make_encoder(modality)

    def forward(self, x) -> dict:
        """Forward pass of the projection."""
        features = {}
        for modality in self.modalities:
            if modality == "aerial":
           
                features[modality] = self.encoders[modality](x[modality])
            elif modality == "s1-asc" or modality == "s2":
               
                for name, module in self.encoders[modality]._modules.items():
                    # print(name)
                    if name == "encoder":
                        features[modality] = module(x[modality], batch_positions=x[modality + "_dates"])
                    else:
                        features[modality] = module(features[modality])

        return features
    
    def make_encoder(self, modality) -> nn.Module:
        """Make encoder for a given modality

        Parameters
        ----------
        modality : str
            Modality of the encoder.

        Returns
        -------
        nn.Module
            Encoder module.
        """
        out_features = self.cfg["model"]["transformer"]["d_model"]
        encoder = []
        ### aerial ###
        if modality == "aerial":
            model = PatchEmbedPastis(
                patch_size=self.cfg["model"]["encoders"]["aerial"]["patch_size"], 
                in_chans=self.cfg["model"]["encoders"]["aerial"]["in_chans"],
                embed_dim=self.cfg["model"]["encoders"]["aerial"]["embed_dim"],
                bias=self.cfg["model"]["encoders"]["aerial"]["bias"],
                res=self.cfg["model"]["encoders"]["aerial"]["res"],
                gp_norm=self.cfg["model"]["encoders"]["aerial"]["gp_norm"])
            encoder.append(("encoder", model))
        ### s1-asc ###
        elif modality == "s1-asc" or modality == "s2":
            model = LTAE2dPastis(
                in_channels=self.cfg["model"]["encoders"][modality]["in_channels"],
                n_head=self.cfg["model"]["encoders"][modality]["n_head"],
                d_k=self.cfg["model"]["encoders"][modality]["d_k"],
                mlp=self.cfg["model"]["encoders"][modality]["mlp"],
                mlp_in=self.cfg["model"]["encoders"][modality]["mlp_in"],
                dropout=self.cfg["model"]["encoders"][modality]["dropout"],
                T=self.cfg["model"]["encoders"][modality]["T"],
                in_norm=self.cfg["model"]["encoders"][modality]["in_norm"],
                positional_encoding=self.cfg["model"]["encoders"][modality]["positional_encoding"],
                patch_size=self.cfg["model"]["encoders"][modality]["patch_size"]
            )

            encoder.append(("encoder", model))
        else:
            raise ValueError(f"Unknown modality: {modality}")

        if self.cfg["model"]["encoders"]["projection"]:
            if self.cfg["model"]["encoders"]["projection_method"] == "pooling":
                encoder.append(
                    (
                        "Pooling", 
                        Fine(
                            self.embed_dim,
                            self.num_patches,
                            [],
                            0.2,
                            0, # No classification head
                            self.pooling_method
                            )
                    )
                )

                encoder.append(
                    (
                        "projection",
                        ProjectionHead(
                            in_features=self.embed_dim if self.pooling_method not in ["avg_f", "max_f"] else self.num_patches,
                            out_features=out_features,
                        )
                    )
                )
            
            if self.cfg["model"]["encoders"]["projection_method"] == "full":

                encoder.append(
                    (
                        "projection",
                        ProjectionHead(
                            in_features=self.num_patches * self.embed_dim,
                            out_features=out_features,
                        ),
                    )
                )

        encoder = nn.Sequential(OrderedDict(encoder))
        return encoder



class ProjectionHead(nn.Module):
    """Projection module."""

    def __init__(self, in_features, out_features, *args, **kwargs) -> None:
        """Initialize the projection module."""
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=out_features),
            nn.BatchNorm1d(out_features),
            nn.ReLU(),
        )

    def forward(self, x) -> torch.Tensor:
        """forward."""
        x = self.projection(x)
        return x

class SequentialWithKwargs(nn.Sequential):
    def forward(self, *inputs, **kwargs):
        for module in self:
            if isinstance(inputs, tuple):
                inputs = module(*inputs, **kwargs)
            else:
                inputs = module(inputs, **kwargs)
        return inputs