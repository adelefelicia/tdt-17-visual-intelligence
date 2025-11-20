import os
from pathlib import Path

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
    def __init__(self, data_root = DATA_ROOT, split = "train", transform = None, cache_data = False):
        self.data_root = Path(data_root)
        self.split = split
        self.transform = transform
        self.cache_data = cache_data
        self.cache = {}
        
        # Load and filter dataset info
        self.split_df = pd.read_csv(self.data_root / "split_unilateral.csv")
        self.split_df = self.split_df[self.split_df['Split'] == split]
        # Filter to only institutions with available annotations
        self.split_df = self.split_df[self.split_df['Institution'].isin(['CAM', 'UKA', 'MHA', 'RUMC'])]
        self.split_df = self.split_df.reset_index(drop=True)
        self.annotations = load_all_annotations(data_root)
        
        print(f"Loaded {len(self.split_df)} samples for {split} split")
    
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
            'age': patient_info['age'],     # TODO should age be included in the model? It can correlate with higher risk...
            'breast_side': patient_info['breast_side'],
            'institution': institution
        }

        if self.transform is not None:
            sample = self.transform(sample)
        
        # Cache if enabled
        if self.cache_data:
            self.cache[idx] = sample
        
        return sample
