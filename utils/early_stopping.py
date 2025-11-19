import numpy as np

class EarlyStopping:
    def __init__(self, patience = 10, delta = 0.00):
        """
        Args:
            patience: How many epochs to wait after last improvement
            delta: Minimum change in monitored quantity to qualify as improvement
        """
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
    
    def __call__(self, val_loss: float):
        score = -val_loss
        
        if self.best_score is None:
            self.best_score = score
            self.val_loss_min = val_loss
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f"Val_loss was not improved. Early stopping: {self.counter} / {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.val_loss_min = val_loss
            self.counter = 0