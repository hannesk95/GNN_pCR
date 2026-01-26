from importlib_metadata import files
import torch
import nibabel as nib
from torch.utils.data import Dataset
from monai.transforms import Compose, ScaleIntensity, Resize
from tqdm import tqdm
import monai
from typing import List, Dict, Optional
from glob import glob
from monai.transforms import (
    Compose,
    MapTransform,
    RandFlipD,
    RandRotateD,
    RandGaussianNoiseD,
    EnsureTypeD,
)
import numpy as np
import itertools
from datetime import datetime
from sklearn.preprocessing import StandardScaler

class SplitMRIAndMask(MapTransform):
    def __init__(self, keys):
        super().__init__(keys)
    def __call__(self, data):
        d = dict(data)
        x = d["image"]                   # (3, H, W, D)

        d["mri"]  = x[:2]               # (2, H, W, D)
        d["mask"] = x[2:3]              # keep channel dim
        return d

class Recombine(MapTransform):
    def __init__(self, keys):
        super().__init__(keys)
    def __call__(self, data):
        d = dict(data)
        d["image"] = torch.cat([d["mri"], d["mask"]], dim=0)
        return d

class ISPY2MIPParametric(Dataset):

    def __init__(
            self, 
            data_list: Dict, 
            timepoints:Optional[List[str]] =['T0','T1','T2','T3'], 
            augmentation: monai.transforms = None, 
            res: tuple = (256,256),
            two_view_transform: bool = False
            ):
        
        self.timepoints = timepoints
        self.augmentation = augmentation
        self.preprocessing_transform = Compose([
            ScaleIntensity(minv=0.0, maxv=1.0),
            Resize(spatial_size=res),
        ])
        self.data_list = self.load_data(data_list)
        self.two_view_transform = two_view_transform

    def get_labels(self):
        labels = [d.get('pcr') for d in self.data_list]
        return labels
    
    def load_three_channel_data(self, timepoint_directory_path):
        pe_early = self.preprocessing_transform(
            torch.FloatTensor(
                nib.load(timepoint_directory_path + 'pe_early.nii.gz').get_fdata()).unsqueeze(0))
        
        pe_late = self.preprocessing_transform(
            torch.FloatTensor(
                nib.load(timepoint_directory_path + 'pe_late.nii.gz').get_fdata()).unsqueeze(0))
        
        ser = self.preprocessing_transform(
            torch.FloatTensor(
                nib.load(timepoint_directory_path + 'ser.nii.gz').get_fdata()).unsqueeze(0))
        
        
        return torch.stack([pe_early, pe_late, ser], dim=0).squeeze()

    def load_data(self, data_dict:Dict):
        print('loading the data ...', flush=True)
        data_list = [] # list of data dictionaries for all of the patients
        for patient in tqdm(data_dict.keys()):
            patient_dict = data_dict[patient]
            loaded_data_dict = {} # dictionary of a single patient data e.g. {'T0' : torch.FloatTensor, 'T1' ..etc}
            loaded_data_dict['id'] = patient
            loaded_data_dict['pcr'] = patient_dict.get('pcr')
            for t in self.timepoints:
                loaded_data_dict[t] = self.load_three_channel_data(patient_dict.get(t))
                loaded_data_dict[f'target_{t}'] = self.load_three_channel_data(patient_dict.get(t))
            data_list.append(loaded_data_dict)
        return data_list

    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        data_dict = self.data_list[idx].copy()
        if self.augmentation is not None:
            for t in self.timepoints:
                if self.two_view_transform:
                    #HACK to make it compatible with monai
                    view1, view2 = self.augmentation(data_dict.get(t))
                    #HACK to make it compatible with monai
                    if type(view1) == monai.data.meta_tensor.MetaTensor and type(view2) == monai.data.meta_tensor.MetaTensor:
                        data_dict[t] = [view1.as_tensor(), view2.as_tensor()]
                    else:
                        data_dict[t] = [view1[0].as_tensor(), view2[0].as_tensor()]
                else:
                    data_dict[t] = self.augmentation(data_dict.get(t))
        return  data_dict

class ISPY2(Dataset):
    def __init__(self, split: str, fold: int, timepoints: int, output_2D: bool = False, output_time_dists: bool = False):

        self.split = split
        self.fold = fold
        self.timepoints = timepoints
        self.patient_ids = torch.load(f"./data/breast_cancer/data_splits_{timepoints}_timepoints.pt")[fold][split]        
        self.transforms = None
        self.output_2D = output_2D
        self.output_time_dists = output_time_dists

        if output_time_dists:

            train_patient_ids = torch.load(f"./data/breast_cancer/data_splits_{timepoints}_timepoints.pt")[fold]['train']
            
            time_diffs = []
            for patient_id in train_patient_ids:
                files = sorted(glob(f"./data/breast_cancer/data_processed/{patient_id}/*.pt"))
                dates_str = []
                for file in files:
                    dates_str.append(file.split("/")[-1].split("_")[1])
                
                dates = [datetime.strptime(d, "%Y-%m-%d") for d in dates_str]
        
                ref = dates[0]
                days_since_ref = [(d - ref).days for d in dates]
                days_since_ref = days_since_ref[:self.timepoints]
                time_diffs.extend(days_since_ref)
            time_diffs = np.array(time_diffs, dtype=np.float32).reshape(-1, 1)  # (N*T, 1)
            self.scaler = StandardScaler()
            self.scaler.fit(time_diffs)
        
        if split in ["train"]:
            self.transforms = Compose([

                SplitMRIAndMask(keys=["image"]),

                # -------- spatial transforms (shared random params) --------
                RandFlipD(
                    keys=["mri", "mask"],
                    prob=0.5,
                    spatial_axis=0,
                ),

                RandRotateD(
                    keys=["mri", "mask"],
                    range_x=np.deg2rad(0),
                    range_y=np.deg2rad(0),
                    range_z=np.deg2rad(175),
                    prob=1.0,
                    keep_size=True,
                    mode=["bilinear", "nearest"],
                ),

                # -------- Gaussian noise ONLY on MR images --------
                RandGaussianNoiseD(
                    keys=["mri"],
                    prob=1.0,
                    mean=0.0,
                    std=0.3,
                ),

                Recombine(keys=["mri", "mask"]),
                EnsureTypeD(keys=["image"]),
            ])            

            self.transforms.set_random_state(seed=42)  # for reproducibility

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):
        
        patient_id = self.patient_ids[idx]
        files = sorted(glob(f"./data/breast_cancer/data_processed/{patient_id}/*.pt"))

        if len(files) < self.timepoints:
            raise RuntimeError(f"Patient {patient_id} has only {len(files)} timepoints")

        data_list = []
        data_list_transforms = []

        for t in range(self.timepoints):
            data = torch.load(files[t])

            if self.output_2D:
                # select middle slice in axial plane
                data = data[:, :, :, :, data.shape[-1]//2]  # (C, D, H, W) -> (C, D, H)

            data_list.append(data)

            if self.transforms:
                data_transformed = self.transforms({"image": data.squeeze(0)})["image"]
                data_transformed = data_transformed.unsqueeze(0)  # add batch dim back            
                data_list_transforms.append(data_transformed)       
            else:
                data_list_transforms.append(data)  # no augmentation, just keep original     
        
        data = torch.cat(data_list+data_list_transforms, dim=0)  # (T, C, D, H, W)

        label = files[0].split('/')[-1].split('_')[-1].replace('.pt', '')
        label = torch.tensor(int(label))        

        if not self.output_time_dists:
            return data, label
        
        else:

            dates_str = []
            for file in files:
                dates_str.append(file.split("/")[-1].split("_")[1])
            
            dates = [datetime.strptime(d, "%Y-%m-%d") for d in dates_str]
            
            ref = dates[0]
            days_since_ref = [(d - ref).days for d in dates]
            # days_since_ref = np.array(days_since_ref, dtype=np.float32)  # (T,)
            # time_dists = torch.from_numpy(days_since_ref).clone()

            days_since_ref = days_since_ref[:self.timepoints]
            days_since_ref = np.array(days_since_ref, dtype=np.float32).reshape(-1, 1)  # (T, 1)
            days_since_ref = self.scaler.transform(days_since_ref).flatten()  # (T,)
            time_dists = torch.from_numpy(days_since_ref).clone()

            return data, label, time_dists 