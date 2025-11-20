import os

import numpy as np
import pandas as pd
import torch

from config import NUM_CLASSES


def get_class_weights(split, data_root, split_mode='csv'):
    """
    Find class weights for handling class imbalance for a given split.
    
    Args:
        split: 'train' or 'val'
        data_root: Path to data root
        split_mode: 'csv' or 'institution'
    """
    if split_mode == 'csv':
        institutions = ['CAM', 'UKA', 'MHA', 'RUMC']
        split_dfs = []
        
        for inst in institutions:
            split_file = os.path.join(data_root, "data", inst, "metadata_unilateral", "split.csv")
            if os.path.exists(split_file):
                df = pd.read_csv(split_file)
                df = df[df['Split'] == split]
                df['Institution'] = inst
                split_dfs.append(df)
        
        split_df = pd.concat(split_dfs, ignore_index=True)
    
    elif split_mode == 'institution':
        if split == 'train':
            institutions = ['CAM', 'RUMC']
        elif split == 'val':
            institutions = ['MHA', 'UKA']
        else:
            raise ValueError(f"Invalid split: {split}. Must be 'train' or 'val' for institution-based splitting")
        
        split_dfs = []
        for inst in institutions:
            split_file = os.path.join(data_root, "data", inst, "metadata_unilateral", "split.csv")
            if os.path.exists(split_file):
                df = pd.read_csv(split_file)
                df['Institution'] = inst
                split_dfs.append(df)
        
        split_df = pd.concat(split_dfs, ignore_index=True)
    
    else:
        raise ValueError(f"Invalid split_mode: {split_mode}. Must be 'csv' or 'institution'")
    
    annotations_df = load_all_annotations(data_root)
    
    # Use inner join to only keep samples with annotations
    merged = split_df.merge(annotations_df, on='UID', how='inner')
    
    labels = merged['Lesion'].astype(int).values
    label_counts = np.bincount(labels, minlength=NUM_CLASSES)
    total = len(labels)
    weights = total / (NUM_CLASSES * label_counts + 1e-6) # Calculate inverse frequency weights
    
    return torch.FloatTensor(weights)

def load_all_annotations(data_root):
    """
    Load all annotations from all institutions to a dataframe.
    """
    annotations = []
    
    institutions = ["CAM", "UKA", "MHA", "RUMC"]
    for inst in institutions:
        annotation_path = os.path.join(data_root, "data", inst, "metadata_unilateral", "annotation.csv")
        if os.path.exists(annotation_path):
            df = pd.read_csv(annotation_path)
            df['Institution'] = inst
            annotations.append(df)
        else:
            print(f"Warning: Annotation file not found for {inst}: {annotation_path}")
    
    annotations_df = pd.concat(annotations, ignore_index=True)
    return annotations_df
