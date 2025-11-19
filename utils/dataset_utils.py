import torch
import os
import pandas as pd
import numpy as np

def get_class_weights(split, data_root):
    """
    Find class weights for handling class imbalance for a given split.
    """
    split_df = pd.read_csv(os.path.join(data_root, "split_unilateral.csv"))
    split_df = split_df[split_df['split'] == split]
    
    annotations_df = load_all_annotations(data_root)
    
    merged = split_df.merge(annotations_df, on='UID', how='left') # Merge to get labels for the split
    labels = merged['Lesion'].values
    label_counts = np.bincount(labels.astype(int), minlength=3)
    total = len(labels)
    weights = total / (3 * label_counts + 1e-6) # Calculate inverse frequency weights
    
    return torch.FloatTensor(weights)

def load_all_annotations(data_root):
    """
    Load all annotations from all institutions to a dataframe.
    """
    annotations = []
    
    institutions = ["CAM", "RSH", "UKA", "MHA", "RUMC"]
    for inst in institutions:
        annotation_path = os.path.join(data_root, inst, "metadata_unilateral", "annotation.csv")
        if os.path.exists(annotation_path):
            df = pd.read_csv(annotation_path)
            df['institution'] = inst
            annotations.append(df)
    
    annotations_df = pd.concat(annotations, ignore_index=True)
    return annotations_df
