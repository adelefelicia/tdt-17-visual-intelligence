import datetime as dt
import json
import os
import time

import monai
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from config import (BATCH_SIZE, DATA_ROOT, EARLY_STOPPING_PATIENCE,
                    IMAGE_SHAPE, LEARNING_RATE, MIN_LR, NUM_CLASSES,
                    NUM_EPOCHS, NUM_WORKERS, RANDOM_SEED, USE_CLASS_WEIGHTS,
                    WEIGHT_DECAY, NUM_SEQUENCES, CACHE_DATA)
from model import BreastMRIClassifier
from odelia_dataset import OdeliaDataset
from utils.dataset_utils import get_class_weights
from utils.early_stopping import EarlyStopping
from utils.train_utils import compute_metrics, save_checkpoint, set_seed
from utils.transform_utils import get_train_transforms, get_val_transforms

torch.set_float32_matmul_precision("medium")

def create_datasets(data_path, train_transform=None, val_transform=None):
    dataset_train = OdeliaDataset(
        data_path, 
        split='train',
        transform=train_transform, 
        cache_data=CACHE_DATA
    )
    dataset_val = OdeliaDataset(
        data_path, 
        split='val',
        transform=val_transform, 
        cache_data=CACHE_DATA
    )
    return dataset_train, dataset_val

def create_dataloaders(dataset_train, dataset_val, batch_size, num_workers):
    train_loader = DataLoader(
        dataset_train, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers
    )
    val_loader = DataLoader(
        dataset_val, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers
    )
    return train_loader, val_loader

def train_epoch(model, dataloader, criterion, optimizer, device, epoch, writer=None):
    model.train()
    
    running_loss = 0.0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]")
    for batch_idx, batch in enumerate(pbar):
        images = batch['image'].to(device)
        labels = batch['label'].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        # Update progress bar
        avg_batch_loss = running_loss / (batch_idx + 1)
        pbar.set_postfix({'loss': avg_batch_loss})
        
        # Log per-batch loss
        if writer:
            global_step = epoch * len(dataloader) + batch_idx
            writer.add_scalar('Train/BatchLoss', loss.item(), global_step)
    
    # Log epoch metrics
    avg_loss = running_loss / len(dataloader)
    if writer:
        writer.add_scalar('Train/Loss', avg_loss, epoch)
    
    return {"loss": avg_loss}

def validate_epoch(model, dataloader, criterion, device, epoch, writer = None):
    model.eval()
    
    running_loss = 0.0
    all_labels = []
    all_probs = []
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Val]")
    with torch.no_grad():
        for batch_idx, batch in enumerate(pbar):
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            
            # Forward pass
            logits = model(images)
            loss = criterion(logits, labels)
            
            # Track metrics
            running_loss += loss.item()
            
            probs = torch.softmax(logits, dim=1)
            
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

            # Update progress bar
            avg_batch_loss = running_loss / (batch_idx + 1)
            pbar.set_postfix({'loss': avg_batch_loss})
    
    # Compute epoch metrics
    metrics = compute_metrics(np.array(all_labels), np.array(all_probs))
    avg_loss = running_loss / len(dataloader)
    metrics['loss'] = avg_loss
    
    # Log epoch metrics
    if writer:
        for key, value in metrics.items():
            writer.add_scalar(f'Val/{key}', value, epoch)
    
    return metrics


def train(model, train_loader, val_loader, criterion, optimizer, scheduler, device,
          num_epochs, log_dir, early_stopping=None, writer=None):
    best_val_score = 0.0
    
    for epoch in range(1, num_epochs + 1):
        # Train
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, writer
        )
        
        # Validate
        val_metrics = validate_epoch(
            model, val_loader, criterion, device, epoch, writer
        )
        
        # Display challenge metrics
        print(f"\nChallenge Metrics (Validation):")
        print(f"AUC:                           {val_metrics['AUC']:.4f}")
        print(f"Specificity @ 90% Sensitivity: {val_metrics['Specificity']:.4f}")
        print(f"Sensitivity @ 90% Specificity: {val_metrics['Sensitivity']:.4f}")
        print(f"Composite Score:               {val_metrics['Score']:.4f}")

        
        # Learning rate scheduling
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        if writer:
            writer.add_scalar("Train/LearningRate", current_lr, epoch)
        
        # Save checkpoint
        if val_metrics['Score'] > best_val_score:
            best_val_score = val_metrics['Score']
        
            save_checkpoint(
                {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'train_metrics': train_metrics,
                    'val_metrics': val_metrics,
                    'best_val_score': best_val_score
                },
                filename=f'best_model.pth',
                checkpoint_dir=log_dir
            )
        
        # Early stopping (monitor composite score)
        if early_stopping:
            early_stopping(val_metrics['Score'])
            if early_stopping.early_stop:
                print(f"\nEarly stopping triggered after {epoch} epochs")
                break
    
    print(f"\nTraining completed!")
    print(f"Best validation (composite) score: {best_val_score:.4f}")

def save_training_config(log_dir):
    config_dict = {
        "IMAGE_SHAPE": IMAGE_SHAPE,
        "NUM_CLASSES": NUM_CLASSES,
        "NUM_EPOCHS": NUM_EPOCHS,
        "BATCH_SIZE": BATCH_SIZE,
        "LEARNING_RATE": LEARNING_RATE,
        "WEIGHT_DECAY": WEIGHT_DECAY,
        "MIN_LR": MIN_LR,
        "USE_CLASS_WEIGHTS": USE_CLASS_WEIGHTS,
        "RANDOM_SEED": RANDOM_SEED,
    }

    with open(os.path.join(log_dir, "config.json"), "w") as f:
        json.dump(config_dict, f, indent=2)

def main(log_dir):
    set_seed(RANDOM_SEED)
    
    save_training_config(log_dir)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print("\n[1/5] Creating datasets...")
    train_transforms = get_train_transforms()
    val_transforms = get_val_transforms()
    dataset_train, dataset_val = create_datasets(DATA_ROOT, train_transform=train_transforms, val_transform=val_transforms)

    print(f"[2/5] Creating dataloaders...")
    train_loader, val_loader = create_dataloaders(
        dataset_train, dataset_val, BATCH_SIZE, NUM_WORKERS
    )
    
    print(f"[3/5] Initializing model...")
    model = BreastMRIClassifier(NUM_SEQUENCES, NUM_CLASSES)
    model = model.to(device)
    
    print(f"[4/5] Setting up loss fn, optimizer, lr scheduler, early stopping, tensorboard...")
    
    # Setup loss function
    if USE_CLASS_WEIGHTS:
        class_weights = get_class_weights('train', DATA_ROOT)
        class_weights = class_weights.to(device)
        print(f"Using class weights: {class_weights}")
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=NUM_EPOCHS,
        eta_min=MIN_LR,
    )
    
    # Setup early stopping (monitor composite score)
    early_stopping = EarlyStopping(patience=EARLY_STOPPING_PATIENCE)
    
    # Setup tensorboard
    writer = SummaryWriter(log_dir=os.path.join(log_dir, 'tensorboard'))
    
    print(f"[5/5] Starting training...\n")
    train(model, train_loader, val_loader, criterion, optimizer, scheduler,
          device, NUM_EPOCHS, log_dir, early_stopping, writer)
    
    if writer:
        writer.close()

if __name__ == "__main__":
    start_time = time.time()

    timestamp = dt.datetime.now().strftime("%Y-%m-%d-%H-%M")
    log_dir = os.path.join('logs', 'train', f'odelia_{timestamp}')
    os.makedirs(log_dir, exist_ok=True) 

    main(log_dir)

    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time:.2f} sec ({training_time/60:.2f} min or {training_time/3600:.2f} h)")

    print(f"\nOpen Tensorboard: tensorboard --logdir {log_dir}/tensorboard")
    print(f"\nBest model: {log_dir}/best_model.pth")
