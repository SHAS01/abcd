from random import shuffle
import numpy as np
from connect_loss import connect_loss as base_connect_loss, connectivity_matrix, Bilateral_voting
import torch.nn.functional as F
import torch
from torch.autograd import Variable
from torch.optim import lr_scheduler
from lr_update import get_lr
from metrics.cldice import clDice
import os
from torch import amp
import sklearn
import torchvision.utils as utils
from sklearn.metrics import precision_score
from skimage.io import imread, imsave
import torch.nn as nn
from skimage.morphology import skeletonize
from scipy.ndimage import distance_transform_edt
import matplotlib.pyplot as plt
import cv2
from datetime import datetime



class connect_loss(nn.Module):
    def __init__(self, args):
        super(connect_loss, self).__init__()
        self.args = args
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, pred, target):
        # Handle tuple output from model
        if isinstance(pred, tuple):
            pred_main = pred[0]  # Main output
            pred_aux = pred[1]   # Auxiliary output
            
            # Get shapes
            batch, channel, H, W = pred_main.shape
            
            # Ensure target has the same shape
            if pred_main.shape[1] != target.shape[1]:
                target = target.repeat(1, pred_main.shape[1], 1, 1)
            
            # Compute main and auxiliary losses
            loss_main = self.bce_loss(pred_main, target)
            loss_aux = self.bce_loss(pred_aux, target)
            
            # Combined loss (main + weighted auxiliary)
            return loss_main + 0.4 * loss_aux
        else:
            # Single output case
            batch, channel, H, W = pred.shape
            
            # Ensure target has the same shape
            if pred.shape[1] != target.shape[1]:
                target = target.repeat(1, pred.shape[1], 1, 1)
            
            # Compute segmentation loss only
            return self.bce_loss(pred, target)

class Metrics:
    @staticmethod
    def get_skeleton(pred):
        # Convert to numpy array and get skeleton
        pred_np = pred.cpu().numpy().astype(np.uint8)
        return torch.from_numpy(skeletonize(pred_np)).to(pred.device)

    @staticmethod
    def calc_centerline_dice(pred, target, h=None):
        """Calculate centerline Dice score (clDice)"""
        # Get skeletons
        pred_skeleton = Metrics.get_skeleton(pred.squeeze())
        target_skeleton = Metrics.get_skeleton(target.squeeze())
        
        if h is None:
            h = max(pred.shape[-2:]) // 100
            
        # Calculate distance maps
        pred_dt = distance_transform_edt(pred.cpu().numpy().squeeze())
        target_dt = distance_transform_edt(target.cpu().numpy().squeeze())
        
        # Convert back to torch
        pred_dt = torch.from_numpy(pred_dt).to(pred.device)
        target_dt = torch.from_numpy(target_dt).to(target.device)
        
        # Calculate TPc and FNc
        TPc = (pred_dt * target_skeleton).sum()
        FNc = ((1 - pred_dt) * target_skeleton).sum()
        
        # Calculate TPt and FNt
        TPt = (target_dt * pred_skeleton).sum()
        FNt = ((1 - target_dt) * pred_skeleton).sum()
        
        # Calculate clDice
        cl_dice_c = 2 * TPc / (2 * TPc + FNc + 1e-8)
        cl_dice_t = 2 * TPt / (2 * TPt + FNt + 1e-8)
        cl_dice = torch.sqrt(cl_dice_c * cl_dice_t)
        
        return cl_dice.item()

    @staticmethod
    def calc_dice(pred, target):
        """Calculate Dice Similarity Coefficient"""
        smooth = 1e-5
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum()
        dice = (2. * intersection + smooth) / (union + smooth)
        return dice.item()

    @staticmethod
    def calc_iou(pred, target):
        """Calculate Intersection over Union"""
        smooth = 1e-5
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum() - intersection
        iou = (intersection + smooth) / (union + smooth)
        return iou.item()

    @staticmethod
    def calc_beta_score(pred, target, beta=0.5):
        """Calculate β-score (balance between precision and recall)"""
        smooth = 1e-5
        TP = (pred * target).sum()
        FP = pred.sum() - TP
        FN = target.sum() - TP
        
        precision = TP / (TP + FP + smooth)
        recall = TP / (TP + FN + smooth)
        
        beta_square = beta * beta
        beta_score = (1 + beta_square) * precision * recall / (beta_square * precision + recall + smooth)
        return beta_score.item()

class Solver(object):
    def __init__(self, args):
        self.args = args
        self.lr = args.lr
        self.best_metrics = {
            'dice': 0.0,
            'cl_dice': 0.0,
            'iou': 0.0,
            'beta0': 0.0,
            'beta1': 0.0
        }
        self.best_epoch = 0
        self.loss_func = connect_loss(args)
        self.metrics = Metrics()
        
        # Create directories for saving results
        self.results_dir = os.path.join('results', args.exp_id)
        self.train_dir = os.path.join(self.results_dir, 'train')
        self.val_dir = os.path.join(self.results_dir, 'best_predictions')
        
        # Create directories
        os.makedirs(self.train_dir, exist_ok=True)
        os.makedirs(self.val_dir, exist_ok=True)

    def save_prediction(self, image, target, prediction, filename, is_training=True):
        """Save visualization of prediction vs ground truth"""
        # Convert tensors to numpy arrays
        image = image.cpu().numpy().transpose(1, 2, 0)
        target = target.cpu().numpy().squeeze()
        prediction = prediction.cpu().numpy().squeeze()
        
        # Normalize image for visualization
        image = (image - image.min()) / (image.max() - image.min())
        
        # Create figure with subplots
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Plot original image
        axes[0].imshow(image)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Plot ground truth
        axes[1].imshow(target, cmap='gray')
        axes[1].set_title('Ground Truth')
        axes[1].axis('off')
        
        # Plot prediction
        axes[2].imshow(prediction, cmap='gray')
        axes[2].set_title('Prediction')
        axes[2].axis('off')
        
        # Add metrics as text
        plt.suptitle(f'Dice: {self.best_metrics["dice"]:.4f}, clDice: {self.best_metrics["cl_dice"]:.4f}\n'
                    f'IoU: {self.best_metrics["iou"]:.4f}, β0: {self.best_metrics["beta0"]:.4f}, '
                    f'β1: {self.best_metrics["beta1"]:.4f}')
        
        # Save figure in appropriate directory
        save_dir = self.train_dir if is_training else self.val_dir
        save_path = os.path.join(save_dir, filename)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()

    def train(self, model, train_loader, val_loader, exp_id, num_epochs=10, start_epoch=0):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=3, verbose=True
        )
        
        scaler = amp.GradScaler()
        print('START TRAIN.')
        
        for epoch in range(start_epoch, num_epochs):
            model.train()
            epoch_loss = 0
            num_batches = len(train_loader)
            
            print(f'\nEpoch {epoch}/{num_epochs-1}')
            print('-' * 10)
            
            for i_batch, (images, targets) in enumerate(train_loader):
                images = images.to(device)
                targets = targets.float().to(device)
                
                optimizer.zero_grad()
                
                with amp.autocast(device_type='cuda', enabled=True):
                    outputs = model(images)
                    loss = self.loss_func(outputs, targets)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                epoch_loss += loss.item()
                
                print(f'[Epoch: {epoch}][Iter: {i_batch}/{num_batches}] Loss: {loss.item():.3f}')
            
            avg_loss = epoch_loss / num_batches
            print(f'Epoch {epoch} average loss: {avg_loss:.3f}')
            
            # Validation phase
            val_metrics = self.validate(model, val_loader, epoch)
            scheduler.step(val_metrics['dice'])
            
            # Save best model
            if val_metrics['dice'] > self.best_metrics['dice']:
                self.best_metrics = val_metrics
                self.best_epoch = epoch
                print("\nNew best model! Saving checkpoint...")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'metrics': val_metrics,
                }, f'models/{exp_id}/best_model.pth')
            
            # Save regular checkpoint
            if (epoch + 1) % self.args.save_per_epochs == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'metrics': val_metrics,
                }, f'models/{exp_id}/epoch_{epoch+1}_model.pth')

        print(f'Best performance at epoch {self.best_epoch} with metrics {self.best_metrics}')
        print('FINISH.')

    def validate(self, model, loader, epoch):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.eval()
        metrics_sum = {
            'dice': 0.0,
            'cl_dice': 0.0,
            'iou': 0.0,
            'beta0': 0.0,
            'beta1': 0.0
        }
        num_samples = 0
        
        print('\nValidation Progress:')
        with torch.no_grad():
            for i, (images, targets) in enumerate(loader):
                images = images.to(device)
                targets = targets.to(device)
                
                outputs = model(images)
                if isinstance(outputs, tuple):
                    predictions = torch.sigmoid(outputs[0])
                else:
                    predictions = torch.sigmoid(outputs)
                pred_binary = (predictions > 0.5).float()
                
                # Calculate metrics
                metrics_sum['dice'] += self.metrics.calc_dice(pred_binary, targets)
                metrics_sum['cl_dice'] += self.metrics.calc_centerline_dice(pred_binary, targets)
                metrics_sum['iou'] += self.metrics.calc_iou(pred_binary, targets)
                metrics_sum['beta0'] += self.metrics.calc_beta_score(pred_binary, targets, beta=0.5)
                metrics_sum['beta1'] += self.metrics.calc_beta_score(pred_binary, targets, beta=1.0)
                
                num_samples += 1
                
                print(f'Batch [{i+1}/{len(loader)}] - '
                      f'Dice: {metrics_sum["dice"]/num_samples:.4f}, '
                      f'clDice: {metrics_sum["cl_dice"]/num_samples:.4f}, '
                      f'IoU: {metrics_sum["iou"]/num_samples:.4f}, '
                      f'β0: {metrics_sum["beta0"]/num_samples:.4f}, '
                      f'β1: {metrics_sum["beta1"]/num_samples:.4f}')
        
        avg_metrics = {k: v/max(num_samples, 1) for k, v in metrics_sum.items()}
        
        print(f'\nValidation Summary:')
        print(f'Average Dice Score: {avg_metrics["dice"]:.4f}')
        print(f'Average clDice Score: {avg_metrics["cl_dice"]:.4f}')
        print(f'Average IoU Score: {avg_metrics["iou"]:.4f}')
        print(f'Average β0 Score: {avg_metrics["beta0"]:.4f}')
        print(f'Average β1 Score: {avg_metrics["beta1"]:.4f}')
        
        return avg_metrics





