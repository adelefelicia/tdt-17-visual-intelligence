import csv
import os
import time

import nibabel as nib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from config import NUM_CLASSES, NUM_SEQUENCES, SEQUENCES, CLASS_NAMES
from model import BreastMRIClassifier
from utils.predict_utils import load_checkpoint
from utils.transform_utils import get_val_transforms


class RSHDataset(Dataset):
    """
    Dataset for RSH data without gt labels.
    Mimics OdeliaDataset structure but without gt.
    """
    def __init__(self, data_root, split_csv_path, transform=None):
        self.data_root = data_root
        self.transform = transform
        
        # Load UIDs from split.csv
        self.split_df = pd.read_csv(split_csv_path)
        print(f"Loaded {len(self.split_df)} samples from {split_csv_path}")
    
    def _get_file_paths(self, uid):
        """Get file paths for all sequences of a given UID."""
        data_folder = os.path.join(self.data_root, uid)
        
        file_paths = {}
        for seq in SEQUENCES:
            file_path = os.path.join(data_folder, f"{seq}.nii.gz")
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Missing sequence file: {file_path}")
            file_paths[seq] = str(file_path)
        
        return file_paths
    
    def __len__(self):
        return len(self.split_df)
    
    def __getitem__(self, idx):
        row = self.split_df.iloc[idx]
        uid = row['UID']
        
        seq_file_paths = self._get_file_paths(uid)
        
        images = []
        for seq in SEQUENCES:
            nii = nib.load(seq_file_paths[seq])
            img_data = nii.get_fdata()
            img_data = img_data[np.newaxis, ...]    # Add channel dimension
            images.append(torch.from_numpy(img_data).float())
        image = torch.cat(images, dim=0)
        
        sample = {
            'image': image,
            'uid': uid
        }

        if self.transform is not None:
            sample = self.transform(sample)
        
        return sample


def predict(model, dataloader, device):
    model.eval()
    
    predictions = {}
    
    start_time = time.time()
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Predicting"):
            images = batch['image'].to(device)
            uids = batch['uid']
            
            logits = model(images)
            probs = torch.softmax(logits, dim=1)
            
            # Store predictions by UID
            for uid, prob in zip(uids, probs.cpu().numpy()):
                predictions[uid] = prob
    
    inference_time = time.time() - start_time

    print(f"\nInference completed in {inference_time:.2f} sec ({inference_time/60:.2f} min)")
    
    return predictions


def save_predictions_to_csv(predictions, output_path):
    
    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        writer.writerow(['ID'] + CLASS_NAMES)
        
        for uid in sorted(predictions.keys()): # Sort bc batching will give a non-deterministic order
            probs = predictions[uid]
            row = [uid]
            for i in range(len(CLASS_NAMES)):
                row.append(f"{probs[i]:.4f}")
            writer.writerow(row)
    
    print(f"Saved predictions to {output_path}")


def load_rsh_data(data_root, split_csv_path, batch_size):
    transforms = get_val_transforms()  # Same transforms as validation

    dataset = RSHDataset(
        data_root=data_root,
        split_csv_path=split_csv_path,
        transform=transforms
    )

    data_loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=1
    )

    return data_loader


def main():
    data_root = "/path/to/dataset/ODELIA2025/data/RSH/data_unilateral"
    split_csv_path = "/path/to/dataset/ODELIA2025/data/RSH/metadata_unilateral/split.csv"
    
    checkpoint_path = "/path/to/your/best_model.pth"
    output_csv_path = "predictions.csv"
    batch_size = 67  # All samples in one batch
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    rsh_loader = load_rsh_data(data_root, split_csv_path, batch_size)
    print(f"RSH batches: {len(rsh_loader)}")
    
    model = BreastMRIClassifier(NUM_SEQUENCES, NUM_CLASSES, dropout_prob=0.0)
    model = model.to(device)
    
    load_checkpoint(checkpoint_path, model, device)
    print(f"Loaded checkpoint from {checkpoint_path}")
    
    predictions = predict(model, rsh_loader, device)
    
    save_predictions_to_csv(predictions, output_csv_path)
    
    print(f"\nPredictions saved to {output_csv_path}")


if __name__ == "__main__":
    main()
