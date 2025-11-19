import datetime as dt
import os
import time

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import (BATCH_SIZE, DATA_ROOT, IMAGE_SHAPE, NUM_CLASSES,
                    NUM_WORKERS, NUM_SEQUENCES)
from model import BreastMRIClassifier
from odelia_dataset import OdeliaDataset
from utils.predict_utils import format_predictions_for_evaluation
from utils.train_utils import load_checkpoint
from utils.transform_utils import get_val_transforms


def predict(model, dataloader, device):
    """
    Generate predictions on a dataset.
    
    Returns:
        Dictionary mapping UID to predicted probabilities
    """
    model.eval()
    
    predictions = {}
    
    start_time = time.time()
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Predicting"):
            images = batch['image'].to(device)
            uids = batch['uid']
            
            logits = model(images)
            probs = torch.softmax(logits, dim=1)
            
            # Store predictions
            for uid, prob in zip(uids, probs.cpu().numpy()):
                predictions[uid] = prob
    
    inference_time = time.time() - start_time

    print(f"\nInference completed in {inference_time:.2f} s ({inference_time/60:.2f} min)")
    
    return predictions

def load_test_data():
    transforms = get_val_transforms() # Same transforms as validation

    dataset = OdeliaDataset(
        DATA_ROOT, 
        split='test', 
        dims=IMAGE_SHAPE, 
        transform=transforms,
        cache_dataset=True
    )

    data_loader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=NUM_WORKERS
    )

    return data_loader

def main(checkpoint_path, prediction_output_path):
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create dataloader
    test_loader = load_test_data()
    print(f"Test batches: {len(test_loader)}")
    
    # Create model
    model = BreastMRIClassifier(NUM_SEQUENCES, NUM_CLASSES)
    model = model.to(device)
    
    # Load checkpoint
    load_checkpoint(checkpoint_path, model)
    
    # Generate predictions
    predictions = predict(model, test_loader, device)
    
    # Save predictions
    output_dir = os.path.join(prediction_output_path)
    os.makedirs(output_dir, exist_ok=True)
    formatted_output_path = os.path.join(output_dir, 'predictions.json')
    format_predictions_for_evaluation(predictions, formatted_output_path)
    
    print(f"\nDone! Predictions saved to {formatted_output_path}.")


if __name__ == "__main__":
    timestamp = dt.datetime.now().strftime("%Y-%m-%d-%H-%M")
    checkpoint_path = "logs/train/odelia_2025_11_dd_hh_mm.pth"  # TODO Path to the trained model checkpoint
    prediction_output_path = f"logs/prediction/{timestamp}"
    
    main(checkpoint_path, prediction_output_path)
