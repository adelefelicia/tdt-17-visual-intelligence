# Old predict for when I thought I would have ground truth for RSH dataset

import argparse
import datetime as dt
import os
import time

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import (BATCH_SIZE, DATA_ROOT, NUM_CLASSES,
                    NUM_WORKERS, NUM_SEQUENCES, SPLIT_MODE)
from model import BreastMRIClassifier
from odelia_dataset import OdeliaDataset
from utils.predict_utils import format_predictions_for_evaluation, load_checkpoint
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

    print(f"\nInference completed in {inference_time:.2f} sec ({inference_time/60:.2f} min)")
    
    return predictions

def load_test_data():
    transforms = get_val_transforms() # Same transforms as validation

    dataset = OdeliaDataset(
        DATA_ROOT, 
        split='test',
        transform=transforms,
        cache_data=False,
        split_mode=SPLIT_MODE
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
    model = BreastMRIClassifier(NUM_SEQUENCES, NUM_CLASSES, dropout_prob=0.0)
    model = model.to(device)
    
    # Load checkpoint
    load_checkpoint(checkpoint_path, model, device)
    
    # Generate predictions
    predictions = predict(model, test_loader, device)
    
    # Save predictions
    output_dir = os.path.join(prediction_output_path)
    os.makedirs(output_dir, exist_ok=True)
    formatted_output_path = os.path.join(output_dir, 'predictions.json')
    format_predictions_for_evaluation(predictions, formatted_output_path)
    
    print(f"\nDone! Predictions saved to {formatted_output_path}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate predictions using a trained model')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to the trained model checkpoint (.pth file)')
    
    args = parser.parse_args()
    checkpoint_path = args.checkpoint
    
    timestamp = dt.datetime.now().strftime("%Y-%m-%d-%H-%M")
    prediction_output_path = f"logs/prediction/{timestamp}"
    
    main(checkpoint_path, prediction_output_path)
