"""
Training script for flood detection models
Supports ResNet, Swin, and MaxViT architectures
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml
from pathlib import Path
from tqdm import tqdm
import argparse
from typing import Dict, Any

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.data.dataset import FloodDataset, create_train_val_test_splits
from src.models.resnet_unet import ResNet50UNet
from src.models.swin_transformer import SwinTransformer
from src.models.max_vit import MaxViTFloodSegmentation
from src.training.losses import get_loss_function
from src.training.metrics import calculate_all_metrics


class Trainer:
    """Training manager for flood detection models"""
    
    def __init__(
        self,
        config: Dict[str, Any],
        model_name: str,
        device: torch.device
    ):
        """
        Args:
            config: Configuration dictionary
            model_name: Name of model ('resnet', 'swin', 'maxvit')
            device: Device for training
        """
        self.config = config
        self.model_name = model_name
        self.device = device
        
        # Create output directory
        os.makedirs(config['paths']['model_checkpoint'], exist_ok=True)
        
        # Initialize model
        self.model = self._create_model()
        self.model.to(device)
        
        # Initialize optimizer and scheduler
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
        # Initialize loss
        self.criterion = get_loss_function(config['training']['loss_type'])
        
        # Training history
        self.train_losses = []
        self.val_metrics = []
        self.best_iou = 0.0
        self.patience_counter = 0
    
    def _create_model(self) -> nn.Module:
        """Create model based on name"""
        input_channels = self.config['model']['input_channels']
        num_classes = self.config['model']['num_classes']
        
        if self.model_name == 'resnet':
            return ResNet50UNet(input_channels=input_channels, num_classes=num_classes)
        elif self.model_name == 'swin':
            return SwinTransformer(input_channels=input_channels, num_classes=num_classes)
        elif self.model_name == 'maxvit':
            return MaxViTFloodSegmentation(input_channels=input_channels, num_classes=num_classes)
        else:
            raise ValueError(f"Unknown model: {self.model_name}")
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer"""
        if self.config['training']['optimizer'] == 'AdamW':
            return optim.AdamW(
                self.model.parameters(),
                lr=self.config['training']['learning_rate'],
                weight_decay=self.config['training']['weight_decay']
            )
        else:
            return optim.Adam(
                self.model.parameters(),
                lr=self.config['training']['learning_rate']
            )
    
    def _create_scheduler(self) -> optim.lr_scheduler._LRScheduler:
        """Create learning rate scheduler"""
        if self.config['training']['scheduler'] == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config['training']['epochs']
            )
        else:
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=10,
                gamma=0.1
            )
    
    def train_epoch(self, dataloader: DataLoader, epoch: int) -> float:
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
        for batch_idx, (images, targets) in enumerate(pbar):
            images = images.to(self.device)
            targets = targets.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            
            # Calculate loss
            loss = self.criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        return running_loss / len(dataloader)
    
    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Validate model"""
        self.model.eval()
        
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for images, targets in tqdm(dataloader, desc="Validating"):
                images = images.to(self.device)
                targets = targets.to(self.device)
                
                outputs = self.model(images)
                preds = torch.argmax(outputs, dim=1)
                
                all_preds.append(preds.cpu())
                all_targets.append(targets.cpu())
        
        # Concatenate all predictions
        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        # Calculate metrics
        metrics = calculate_all_metrics(all_preds, all_targets)
        
        return metrics
    
    def save_checkpoint(self, epoch: int, iou: float):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'iou': iou,
            'config': self.config
        }
        
        filename = f"models/{self.model_name}_best.pt"
        torch.save(checkpoint, filename)
        print(f"Checkpoint saved: {filename}")
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader
    ):
        """Main training loop"""
        epochs = self.config['training']['epochs']
        patience = self.config['training']['early_stopping_patience']
        
        for epoch in range(epochs):
            # Train
            train_loss = self.train_epoch(train_loader, epoch)
            self.train_losses.append(train_loss)
            
            # Validate
            val_metrics = self.validate(val_loader)
            self.val_metrics.append(val_metrics)
            
            # Update learning rate
            self.scheduler.step()
            
            # Print metrics
            print(f"\nEpoch {epoch+1}/{epochs}")
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val IoU: {val_metrics['mean_iou']:.4f}")
            print(f"Val F1: {val_metrics['f1_score']:.4f}")
            print(f"Val Precision: {val_metrics['precision']:.4f}")
            print(f"Val Recall: {val_metrics['recall']:.4f}")
            
            # Save best model
            if val_metrics['mean_iou'] > self.best_iou:
                self.best_iou = val_metrics['mean_iou']
                self.save_checkpoint(epoch, self.best_iou)
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            # Early stopping
            if self.patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break


def main(args):
    """Main training function"""
    # Load config
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create data splits
    print("Loading data...")
    train_data, val_data, test_data = create_train_val_test_splits(
        config['paths']['raw_data'] + 'S1/',
        config['paths']['raw_data'] + 'S2/',
        config['paths']['raw_data'] + 'labels/',
        train_ratio=config['data']['train_split'],
        val_ratio=config['data']['val_split'],
        test_ratio=config['data']['test_split']
    )
    
    # Create datasets
    train_dataset = FloodDataset(
        train_data[0], train_data[1], train_data[2],
        is_train=True, augment=True
    )
    val_dataset = FloodDataset(
        val_data[0], val_data[1], val_data[2],
        is_train=False, augment=False
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=True,
        num_workers=config['data']['num_workers']
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers']
    )
    
    # Create trainer
    model_name = args.model if args.model else 'resnet'
    trainer = Trainer(config, model_name, device)
    
    # Train
    print("\nStarting training...")
    trainer.train(train_loader, val_loader)
    
    print(f"\nBest IoU: {trainer.best_iou:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, choices=['resnet', 'swin', 'maxvit'],
                       help='Model architecture to train')
    args = parser.parse_args()
    
    main(args)

