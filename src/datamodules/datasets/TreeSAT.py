from torch.utils.data import Dataset
import h5py
import numpy as np
import json
from src.utils.utils import subset_dict_by_filename, filter_labels_by_threshold
from skmultilearn.model_selection import iterative_train_test_split
import torch
import rasterio
from datetime import datetime
from src.datamodules.datasets.transforms.transform import Transform

from src.utils.training.augmentations import MultiModalAugmentations

def collate_fn(batch):
    """
    Collate function for the dataloader.
    Args:
        batch (list): list of dictionaries with keys "label", "name"  and the other corresponding to the modalities used
    Returns:
        dict: dictionary with keys "label", "name"  and the other corresponding to the modalities used
    """
    # print(type(batch[0]))
    # 处理contrastive的batch
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
            output[key] = stacked_tensor
            keys.remove(key)
            key = '_'.join([key, "dates"])
            idx = [x[key] for x in batch]
            max_size_0 = max(tensor.size(0) for tensor in idx)
            stacked_tensor = torch.stack([
                    torch.nn.functional.pad(tensor, (0, max_size_0 - tensor.size(0)))
                    for tensor in idx
                ], dim=0)
            output[key] = stacked_tensor
            keys.remove(key)
    if 'name' in keys:
        output['name'] = [x['name'] for x in batch]
        keys.remove('name')
    keys_with_id = [key for key in keys if 'id' in key]
    for key in keys_with_id:
        output[key] = [x[key] for x in batch]
        keys.remove(key)
    for key in keys:
        output[key] = torch.stack([x[key] for x in batch])
    
    return output

def day_number_in_year(date_arr, place=4):
    day_number = []
    for date_string in date_arr:
        date_object = datetime.strptime(str(date_string).split('_')[place][:8], '%Y%m%d')
        day_number.append(date_object.timetuple().tm_yday) # Get the day of the year
    return torch.tensor(day_number)

def replace_nans_with_mean(batch_of_images):
    image_means = torch.nanmean(batch_of_images, dim=(3, 4), keepdim=True)
    image_means[torch.isnan(image_means)] = 0.
    nan_mask = torch.isnan(batch_of_images)
    batch_of_images[nan_mask] = image_means.expand_as(batch_of_images)[nan_mask]
    return batch_of_images

class TreeSAT(Dataset):
    def __init__(
        self,
        cfg,
        split: str = "train",
        stage="anchor"
        ):
        """
        Initializes the dataset.
        Args:
            path (str): path to the dataset
            modalities (list): list of modalities to use
            transform (torch module): transform to apply to the data
            split (str): split to use (train, val, test)
            classes (list): name of the differrent classes
            partition (float): proportion of the dataset to keep
            mono_strict (bool): if True, puts monodate in same condition as multitemporal
        """
        self.cfg = cfg
        self.stage = stage
        self.path = cfg["paths"]["data"]+"/"
        self.transform = Transform(cfg["multimodal"][split + "_transform"][0]["p"], cfg["multimodal"][split + "_transform"][1]["size"])
        if stage == "classification":
            self.partition = cfg["multimodal"]["classification_partition"]
        else:
            self.partition = cfg["multimodal"]["partition"]
        self.modalities = cfg["multimodal"]["modalities"]
        self.mono_strict = cfg["multimodal"]["mono_strict"]
        data_path = self.path + split + "_filenames.lst"
        with open(data_path, 'r') as file:
            self.data_list = [line.strip() for line in file.readlines()]
        self.load_labels(cfg["multimodal"]["classes"])
        self.collate_fn = collate_fn

        if self.stage == "contrastive":
            self.augmentations = MultiModalAugmentations(
                modalities=self.cfg["multimodal"]["modalities"],
                modality_dropout=self.cfg["model"]["contrastive_loss"][
                    "modality_dropout"
                ],
                noise_sigma=self.cfg["model"]["contrastive_loss"]["noise_sigma"],
            )
            
    def load_labels(self, classes):
        """
        Load the labels from the json file and filter them according to the threshold.
        Args:
            classes (list): list of the classes
        """
        with open(self.path + "labels/TreeSatBA_v9_60m_multi_labels.json") as file:
            jfile = json.load(file)
            subsetted_dict = subset_dict_by_filename(self.data_list, jfile) # Filter out labels that are not in the dataset, return form of {filename: labels}
            labels = filter_labels_by_threshold(subsetted_dict, 0.07) # Filter out labels with area less than 0.07, return form of {filename: [label, label, ...]}
            lines = list(labels.keys()) # Get the filenames

        # Create the y matrix
        y = [[0 for i in range(len(classes))] for line in lines]
        for i, line in enumerate(lines):
            for u in labels[line]:
                y[i][classes.index(u)] = 1


        self.data_list, self.labels, _, _ = iterative_train_test_split(np.expand_dims(np.array(lines), axis=1), np.array(y), test_size = 1. - self.partition) # Split the data into train and test sets, but get only the train set
        self.data_list = list(np.concatenate(self.data_list).flat)

    def __getitem__(self, i):
        """
        Returns an item from the dataset.
        Args:
            i (int): index of the item
        Returns:
            dict: dictionary with keys "label", "name" and the other corresponding to the modalities used
        """
        name = self.data_list[i]
        output = {'label': torch.tensor(self.labels[i]), 'name': name}
    
        ########## Load Aerial Data ##########
        if 'aerial' in self.modalities:
            with rasterio.open(self.path + "aerial/" + name) as f:
                output["aerial"] = torch.FloatTensor(f.read())
                output["id_aerial"] = name
        
        ########## Load Time Series Data ##########
        # # 文件名处理
        # with open(self.path + "sentinel_filenames.lst", 'r') as f:
        #     filenames = f.read().splitlines()
        
        # # 过滤出以 prefix 开头的文件名
        # matching_files = [file for file in filenames if file.startswith('.'.join(name.split('.')[:-1]))]
        # # print(len(matching_files))

        with h5py.File(self.path + "sentinel/" + '.'.join(name.split('.')[:-1]) + ".h5", 'r') as file:
        # with h5py.File(self.path+ "sentinel/" + matching_files[0], 'r') as file:
            ########## Load s1-asc Data ##########
            if 's1-asc' in self.modalities:
                output["s1-asc_dates"] = torch.load(self.path + "s1-asc_dates/" + '.'.join(name.split('.')[:-1]) + ".pth")[:]
                output["s1-asc"] = torch.load(self.path + "s1-asc/" + '.'.join(name.split('.')[:-1]) + ".pth")[:, :2, : ,:]
                N = len(output["s1-asc_dates"])
                # print("s1-asc_dates shape ", output["s1-asc_dates"].shape)
                # print("s1-asc shape ", output["s1-asc"].shape)
                # print("N ", N)
                if N > 50:
                    random_indices = torch.randperm(N)[:50]
                    # print("Random indices:", random_indices)  # 检查是否有超出 60 的索引
                    # print("Max index:", random_indices.max().item())  # 查看最大索引值
                    output["s1-asc"] = output["s1-asc"][random_indices]
                    output["s1-asc_dates"] = output["s1-asc_dates"][random_indices]

                output["id_s1-asc"] = name
                output["id_s1-asc_dates"] = name

                # output["s1-asc_dates"] = day_number_in_year(file["sen-1-asc-products"][:])
                # load s1-asc directly from the h5 file according to huggingface
                # output["s1-asc"] = torch.tensor(file['sen-1-asc-data'][:]) 没有预处理！！！
                
                # output["s1-asc"] = torch.load(self.path + "s1-asc/" + '.'.join(name.split('.')[:-1]) + ".pth")[:, :2, : ,:]
                # # print(output["s1-asc"].shape)
                # N = len(output["s1-asc_dates"])
                # if N > 50:
                #     random_indices = torch.randperm(N)[:50]
                #     output["s1-asc"] = output["s1-asc"][random_indices]
                #     output["s1-asc_dates"] = output["s1-asc_dates"][random_indices]
                # print("s1-asc_dates shape ", output["s1-asc_dates"].shape)
                # print("s1-asc shape ", output["s1-asc"].shape)

                

            ########## Load s1-des Data ##########
            if 's1-des' in self.modalities:
                output["s1-des_dates"] = day_number_in_year(file["sen-1-des-products"][:])
            
            ########## Load s1 Data ##########
            if 's1' in self.modalities:
                s1_asc_dates = day_number_in_year(file["sen-1-asc-products"][:])
                s1_des_dates = day_number_in_year(file["sen-1-des-products"][:])
            
            ########## Load s2 Data ##########
            if 's2' in self.modalities:
                output["s2"]= torch.tensor(file["sen-2-data"][:])
                # print(output["s2"].shape)
                output["s2_dates"] = day_number_in_year(file["sen-2-products"][:], place=2)
                N = len(output["s2_dates"])
                if N > 50:
                    random_indices = torch.randperm(N)[:50]
                    output["s2"] = output["s2"][random_indices]
                    output["s2_dates"] = output["s2_dates"][random_indices]
                
                output["id_s2"] = name
                output["id_s2_dates"] = name


        if 's1-des' in self.modalities:
            output["s1-des"] = torch.load(self.path + "s1-des/" + '.'.join(name.split('.')[:-1]) + ".pth")[:, :2, : ,:]
            N = len(output["s1-des_dates"])
            if N > 50:
                random_indices = torch.randperm(N)[:50]
                output["s1-des"] = output["s1-des"][random_indices]
                output["s1-des_dates"] = output["s1-des_dates"][random_indices]

        if 's1' in self.modalities:
            s1_asc = torch.load(self.path + "s1-asc/" + '.'.join(name.split('.')[:-1]) + ".pth")[:, :2, : ,:]
            s1_des = torch.load(self.path + "s1-des/" + '.'.join(name.split('.')[:-1]) + ".pth")[:, :2, : ,:]
            output["s1"] = torch.cat([s1_asc, s1_des], dim=0)
            output["s1_dates"] = torch.cat([s1_asc_dates, s1_des_dates], dim=0)
            N = len(output["s1_dates"])
            if N > 50:
                random_indices = torch.randperm(N)[:50]
                output["s1"] = output["s1"][random_indices]
                output["s1_dates"] = output["s1_dates"][random_indices]

        if "s1-mono" in self.modalities:
            with rasterio.open(self.path + "s1/60m/" + name) as f:
                numpy_array = f.read()
            numpy_array = numpy_array.astype(np.float32)
            output["s1-mono"] = torch.FloatTensor(numpy_array)
            if self.mono_strict:
                output["s1-mono"] = output["s1-mono"][:2, :, :]

        if "s2-mono" in self.modalities:
            with rasterio.open(self.path + "s2/60m/" + name) as f:
                numpy_array = f.read()
            numpy_array = numpy_array.astype(np.float32)
            output["s2-mono"] = torch.FloatTensor(numpy_array)
            if self.mono_strict:
                output["s2-mono"] = output["s2-mono"][:10, :, :]

        if "s2-4season-median" in self.modalities:
            with h5py.File(self.path + "sentinel/" + '.'.join(name.split('.')[:-1]) + ".h5", 'r') as file:
                output_inter = torch.tensor(file["sen-2-data"][:])
                dates = day_number_in_year(file["sen-2-products"][:], place=2)
            l = []
            for i in range (4):
                mask = ((dates >= 92 * i) & (dates < 92 * (i+1)))
                if sum(mask) > 0:
                    r, _ = torch.median(output_inter[mask], dim = 0)
                    l.append(r)
                else:
                    l.append(torch.zeros((output_inter.shape[1], output_inter.shape[-2], output_inter.shape[-1])))
            output["s2-4season-median"] = torch.cat(l)

        if "s2-median" in self.modalities:
            with h5py.File(self.path + "sentinel/" + '.'.join(name.split('.')[:-1]) + ".h5", 'r') as file:
                output["s2-median"], _ = torch.median(torch.tensor(file["sen-2-data"][:]), dim = 0)

        if "s1-4season-median" in self.modalities:
            with h5py.File(self.path + "sentinel/" + '.'.join(name.split('.')[:-1]) + ".h5", 'r') as file:
                dates = day_number_in_year(file["sen-1-asc-products"][:])
            output_inter = torch.load(self.path + "s1-asc/" + '.'.join(name.split('.')[:-1]) + ".pth")[:, :2, : ,:]
            l = []
            for i in range (4):
                mask = ((dates >= 92 * i) & (dates < 92 * (i+1)))
                if sum(mask) > 0:
                    r, _ = torch.median(output_inter[mask], dim = 0)
                    l.append(r)
                else:
                    l.append(torch.zeros((output_inter.shape[1], output_inter.shape[-2], output_inter.shape[-1])))
            output["s1-4season-median"] = torch.cat(l)

        if "s1-median" in self.modalities:
            output["s1-median"], _ = torch.median(torch.load(self.path + "s1-asc/" + '.'.join(name.split('.')[:-1]) + ".pth")[:, :2, : ,:], dim = 0)


        if self.stage == "contrastive":
            output = self.augmentations(output)
            return output
        else:
            return self.transform(output)

    def __len__(self):
        return len(self.data_list)