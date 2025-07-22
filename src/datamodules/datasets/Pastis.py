from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import os
import geopandas as gpd
import torch
import rasterio
from datetime import datetime

import h5py
import json
from src.utils.utils import subset_dict_by_filename, filter_labels_by_threshold
from skmultilearn.model_selection import iterative_train_test_split
from src.datamodules.datasets.transforms.transform import Transform

from src.utils.training.augmentations import MultiModalAugmentations


def collate_fn(batch):
    """
    Collate function for the dataloader.
    Args:
        batch (list): list of dictionaries with keys "label", "name" and the other corresponding to the modalities used
    Returns:
        dict: dictionary with keys "label", "name"  and the other corresponding to the modalities used
    """
    if isinstance(batch[0], tuple):
        data1, data2 = zip(*batch)
        return (collate_fn(data1), collate_fn(data2))

    keys = list(batch[0].keys())
    output = {}
    for key in ["s2", "s1-asc", "s1-des", "s1"]:
        if key in keys:
            idx = [x[key] for x in batch]
            max_size_0 = max(tensor.size(0) for tensor in idx)
            stacked_tensor = torch.stack([
                    torch.nn.functional.pad(tensor, (0, 0, 0, 0, 0, 0, 0, max_size_0 - tensor.size(0)))
                    for tensor in idx
                ], dim=0)
            output[key] = stacked_tensor.float()
            keys.remove(key)
            key = '_'.join([key, "dates"])
            idx = [x[key] for x in batch]
            max_size_0 = max(tensor.size(0) for tensor in idx)
            stacked_tensor = torch.stack([
                    torch.nn.functional.pad(tensor, (0, max_size_0 - tensor.size(0)))
                    for tensor in idx
                ], dim=0)
            output[key] = stacked_tensor.long()
            keys.remove(key)
    if 'name' in keys:
        output['name'] = [x['name'] for x in batch]
        keys.remove('name')
    for key in keys:
        output[key] = torch.stack([torch.tensor(x[key]) for x in batch]).float()
    return output

def day_number_in_year(date_arr, place=4):
    day_number = []
    for date_string in date_arr:
        date_object = datetime.strptime(str(date_string).split('_')[place][:8], '%Y%m%d')
        day_number.append(date_object.timetuple().tm_yday) # Get the day of the year
    return torch.tensor(day_number)

def prepare_dates(date_dict, reference_date):
    """Date formating."""
    # print(f"date_dict type: {type(date_dict)}, value: {date_dict}")
    # 如果 date_dict 是字符串，转换成字典
    if isinstance(date_dict, str):
        try:
            date_dict = json.loads(date_dict)  # 解析 JSON 字符串
        except json.JSONDecodeError:
            raise ValueError(f"Expected dict but got string: {date_dict}")

    d = pd.DataFrame().from_dict(date_dict, orient="index")
    d = d[0].apply(
        lambda x: (
            datetime(int(str(x)[:4]), int(str(x)[4:6]), int(str(x)[6:]))
            - reference_date
        ).days
    )
    return torch.tensor(d.values)

def split_image(image_tensor, nb_split, id):
    """
    Split the input image tensor into four quadrants based on the integer i.
    To use if Pastis data does not fit in your GPU memory.
    Returns the corresponding quadrant based on the value of i
    """
    if nb_split == 1:
        return image_tensor
    i1 = id // nb_split
    i2 = id % nb_split
    height, width = image_tensor.shape[-2:]
    half_height = height // nb_split
    half_width = width // nb_split
    if image_tensor.dim() == 4:
        return image_tensor[:, :, i1*half_height:(i1+1)*half_height, i2*half_width:(i2+1)*half_width].float()
    if image_tensor.dim() == 3:
        return image_tensor[:, i1*half_height:(i1+1)*half_height, i2*half_width:(i2+1)*half_width].float()
    if image_tensor.dim() == 2:
        return image_tensor[i1*half_height:(i1+1)*half_height, i2*half_width:(i2+1)*half_width].float()

class PASTIS(Dataset):
    def __init__(
        self,
        cfg,
        split: str = "train",
        stage="anchor",
        folds=None,
        reference_date="2018-09-01",
        nb_split = 1,
        num_classes = 20
        ):
        """
        Initializes the dataset.
        Args:
            path (str): path to the dataset
            modalities (list): list of modalities to use
            folds (list): list of folds to use
            reference_date (str date): reference date for the data
            nb_split (int): number of splits from one observation
            num_classes (int): number of classes
        """
        super(PASTIS, self).__init__()
        self.cfg = cfg
        self.stage = stage
        self.path = self.cfg["paths"]["data"]+"/"
        # transform = Identity

        self.modalities = self.cfg["multimodal"]["modalities"]
        self.mono_strict = self.cfg["multimodal"]["mono_strict"]

        if stage == "classification":
            self.partition = self.cfg["multimodal"]["classification_partition"]
        else:
            self.partition = self.cfg["multimodal"]["partition"]

        self.nb_split = nb_split

        self.reference_date = datetime(*map(int, reference_date.split("-")))

        self.meta_patch = gpd.read_file(os.path.join(self.path, "metadata.geojson"))

        self.num_classes = self.cfg["multimodal"]["num_classes"] + 2

        if folds is not None:
            self.meta_patch = pd.concat(
                [self.meta_patch[self.meta_patch["Fold"] == f] for f in folds]
            )
        self.collate_fn = collate_fn

        if self.stage == "contrastive":
            self.augmentations = MultiModalAugmentations(
                modalities=self.cfg["multimodal"]["modalities"],
                modality_dropout=self.cfg["model"]["contrastive_loss"][
                    "modality_dropout"
                ],
                noise_sigma=self.cfg["model"]["contrastive_loss"]["noise_sigma"],
            )

    def __getitem__(self, i):
        """
        Returns an item from the dataset.
        Args:
            i (int): index of the item
        Returns:
            dict: dictionary with keys "label", "name" and the other corresponding to the modalities used
        """
        line = self.meta_patch.iloc[i // (self.nb_split * self.nb_split)]
        name = line['ID_PATCH']
        part = i % (self.nb_split * self.nb_split)
        label = torch.from_numpy(np.load(os.path.join(self.path, 'ANNOTATIONS/TARGET_' + str(name) + '.npy'))[0].astype(np.int32))
        label = torch.unique(split_image(label, self.nb_split, part)).long()
        label = torch.sum(torch.nn.functional.one_hot(label, num_classes=self.num_classes), dim = 0)
        label = label[1:-1] #remove Background and Void classes
        output = {'label': label, 'name': name}

        for modality in self.modalities:
            if modality == "aerial":
                with rasterio.open(os.path.join(self.path, 'DATA_SPOT/PASTIS_SPOT6_RVB_1M00_2019/SPOT6_RVB_1M00_2019_' + str(name) + '.tif')) as f:
                    output["aerial"] = split_image(torch.FloatTensor(f.read()), self.nb_split, part)
                    output["id_aerial"] = name
            else:
                # if len(modality) > 3:
                #     modality_name = modality[:2] + modality[3]
                #     output[modality] = split_image(torch.from_numpy(np.load(os.path.join(
                #             self.path,
                #             "DATA_{}".format(modality_name.upper()),
                #             "{}_{}.npy".format(modality_name.upper(), name),
                #         ))), self.nb_split, part)
                #     output['_'.join([modality, 'dates'])] = prepare_dates(line['-'.join(['dates', modality_name.upper()])], self.reference_date)
                # else:
                correspondence = {"s2": "S2", "s1-asc": "S1A"}

                output[modality] = split_image(torch.from_numpy(np.load(os.path.join(
                        self.path,
                        "DATA_{}".format(correspondence[modality]),
                        "{}_{}.npy".format(correspondence[modality], name),
                    ))), self.nb_split, part)

                output['_'.join([modality, 'dates'])] = prepare_dates(line['-'.join(['dates', correspondence[modality]])], self.reference_date)
                # print(output.keys())
                N = len(output[modality])
                if N > 50:
                    random_indices = torch.randperm(N)[:50]
                    output[modality] = output[modality][random_indices]
                    output['_'.join([modality, 'dates'])] = output['_'.join([modality, 'dates'])][random_indices]
                
                output["id_" + modality] = name
                output["id_" + modality + "_dates"] = name
        
        if self.stage == "contrastive":
            output = self.augmentations(output)
            return output
        else:
            return output

    def __len__(self):
        return len(self.meta_patch) * self.nb_split * self.nb_split