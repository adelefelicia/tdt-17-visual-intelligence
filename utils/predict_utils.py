import json
import torch


def format_predictions_for_evaluation(predictions, output_path):
    """
    Format predictions in the required evaluation format.
    
    Expected format:
    {
        "right": {"normal": 0.001, "benign": 0.01, "malignant": 0.988},
        "left": {"normal": 0.987, "benign": 0.02, "malignant": 0.001}
    }
    """
    
    formatted_predictions = {}
    class_names = ['normal', 'benign', 'malignant']
    
    for uid, probs in predictions.items():
        breast_side = 'left' if '_left' in uid else 'right'
        patient_id = uid.rsplit('_', 1)[0]
        
        if patient_id not in formatted_predictions:
            formatted_predictions[patient_id] = {}
        
        # Format probabilities
        formatted_predictions[patient_id][breast_side] = {
            class_names[i]: float(probs[i])
            for i in range(len(class_names))
        }
    
    # Save to file
    with open(output_path, 'w') as f:
        json.dump(formatted_predictions, f, indent=2)
    
    print(f"Saved predictions to {output_path}")


def load_checkpoint(checkpoint_path, model, device, optimizer=None, scheduler=None):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    return checkpoint