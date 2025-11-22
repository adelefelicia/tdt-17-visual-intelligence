import os

import nibabel as nib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from config import DATA_ROOT, SEQUENCES
from utils.dataset_utils import load_all_annotations


class OdeliaDataset(Dataset):
    """
    Loads MRI volumes for breast lesion classification.
    Each sample corresponds to one breast (left or right) with 5 sequences.
    """
    def __init__(self, data_root = DATA_ROOT, split = "train", transform = None, cache_data = False, split_mode = 'csv'):
        self.data_root = data_root
        self.split = split
        self.transform = transform
        self.cache_data = cache_data
        self.cache = {}
        self.split_mode = split_mode
        
        self.split_df = self._load_splits()
        self.annotations = load_all_annotations(data_root)
        
        print(f"Loaded {len(self.split_df)} samples for {split} split (mode: {split_mode})")
    
    def _load_splits(self):
        """Load and combine institution-specific split files based on split mode."""
        if self.split_mode == 'csv':
            return self._load_csv_splits()
        elif self.split_mode == 'institution':
            return self._load_institution_based_splits()
        else:
            raise ValueError(f"Invalid split_mode: {self.split_mode}. Must be 'csv' or 'institution'")
    
    def _load_csv_splits(self):
        """Load splits using the Split column from CSV files."""
        institutions = ['CAM', 'UKA', 'MHA', 'RUMC']
        split_dfs = []
        
        for inst in institutions:
            split_file = os.path.join(self.data_root, "data", inst, "metadata_unilateral", "split.csv")
            if os.path.exists(split_file):
                df = pd.read_csv(split_file)
                # Filter by split
                df = df[df['Split'] == self.split]
                df['Institution'] = inst
                split_dfs.append(df)
            else:
                print(f"Warning: Split file not found for {inst}: {split_file}")
        
        combined_df = pd.concat(split_dfs, ignore_index=True)
        return combined_df.reset_index(drop=True)
    
    def _load_institution_based_splits(self):
        """Load splits based on institution: Train on CAM+RUMC, Val on MHA+UKA."""
        if self.split == 'train':
            institutions = ['CAM', 'RUMC']
        elif self.split == 'val':
            institutions = ['MHA', 'UKA']
        else:
            raise ValueError(f"Invalid split: {self.split}. Must be 'train' or 'val' for institution-based splitting")
        
        split_dfs = []
        for inst in institutions:
            split_file = os.path.join(self.data_root, "data", inst, "metadata_unilateral", "split.csv")
            if os.path.exists(split_file):
                df = pd.read_csv(split_file)
                # Load all samples from this institution (ignore Split column)
                df['Institution'] = inst
                split_dfs.append(df)
            else:
                print(f"Warning: Split file not found for {inst}: {split_file}")
        
        combined_df = pd.concat(split_dfs, ignore_index=True)
        return combined_df.reset_index(drop=True)
    
    def _get_file_paths(self, uid, institution):
        """Get file paths for all sequences of a given UID."""
        data_folder = os.path.join(self.data_root, "data", institution, "data_unilateral", uid)
        
        file_paths = {}
        for seq in SEQUENCES:
            file_path = os.path.join(data_folder, f"{seq}.nii.gz")
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Missing sequence file: {file_path}")
            file_paths[seq] = str(file_path)
        
        return file_paths
    
    def _get_lesion_label(self, uid):
        annotation = self.annotations[self.annotations['UID'] == uid]
        if len(annotation) == 0:
            raise ValueError(f"No annotation found for UID: {uid}")
        
        lesion = annotation.iloc[0]['Lesion']
        return int(lesion)
    
    def _get_patient_info(self, uid):
        annotation = self.annotations[self.annotations['UID'] == uid]
        if len(annotation) == 0:
            raise ValueError(f"No annotation found for UID: {uid}")
        
        row = annotation.iloc[0]
        return {
            'patient_id': row['PatientID'],
            'age': int(row['Age']),
            'breast_side': 'left' if '_left' in uid else 'right'
        }
    
    def __len__(self):
        return len(self.split_df)
    
    def __getitem__(self, idx):
        """
        Returns:
            Dictionary containing:
                - image: Tensor of shape (5, H, W, D)
                - label: Lesion label (0=no lesion, 1=benign, 2=malignant)
                - uid: Sample identifier
                - patient_id
                - age: Patient age in days
                - breast_side: 'left' or 'right'
                - institution
        """
        if self.cache_data and idx in self.cache:
            return self.cache[idx]
        
        row = self.split_df.iloc[idx]
        uid = row['UID']
        institution = row['Institution']
        
        seq_file_paths = self._get_file_paths(uid, institution)
        
        label = self._get_lesion_label(uid)
        patient_info = self._get_patient_info(uid)
        
        images = []
        for seq in SEQUENCES:
            nii = nib.load(seq_file_paths[seq])
            img_data = nii.get_fdata()
            img_data = img_data[np.newaxis, ...]    # Add channel dimension
            images.append(torch.from_numpy(img_data).float())
        image = torch.cat(images, dim=0)
        
        sample = {
            'image': image,
            'label': label,
            'uid': uid,
            'patient_id': patient_info['patient_id'],
            'age': patient_info['age'],
            'breast_side': patient_info['breast_side'],
            'institution': institution
        }

        if self.transform is not None:
            sample = self.transform(sample)
        
        # Cache if enabled
        if self.cache_data:
            self.cache[idx] = sample
        
        return sample
