import numpy as np
import torch
import logging
from torch.utils.tensorboard import SummaryWriter

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class EarlyStopping:
    def __init__(self, patience=5, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            if self.verbose:
                logging.info("Initialization of best score for early stopping.")
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    logging.info("Early stopping triggered.")
        else:
            self.best_score = score
            self.counter = 0
            if self.verbose:
                logging.info(f"Best score updated to {self.best_score}.")

class TensorBoardLogger:
    def __init__(self, log_dir):
        self.writer = SummaryWriter(log_dir)
        logging.info(f"TensorBoard logging started in: {log_dir}")

    def logTraining(self, loss, step):
        self.writer.add_scalar('Loss/train', loss, step)
        logging.info(f"Training loss logged: {loss} at step {step+1}.")

    def logValidation(self, loss, accuracy, step):
        self.writer.add_scalar('Loss/val', loss, step)
        self.writer.add_scalar('Accuracy/val', accuracy, step)
        logging.info(f"Validation loss and accuracy logged: Loss - {loss}, Accuracy - {accuracy} at step {step+1}.")
