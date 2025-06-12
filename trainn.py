import argparse
import os
import torch
from torch.utils.data import DataLoader
from model.DconnNet import DconnNet
from solver import Solver
from data_loader.GetDataset_CHASE import MyDataset_CHASE

def get_args():
    parser = argparse.ArgumentParser(description='Train DconnNet on CHASE_DB1 dataset')
    
    # Training settings
    parser.add_argument('--batch-size', type=int, default=4,
                      help='input batch size for training (default: 4)')
    parser.add_argument('--epochs', type=int, default=200,
                      help='number of epochs to train (default: 200)')
    parser.add_argument('--lr', type=float, default=0.001,
                      help='initial learning rate (default: 0.001)')
    parser.add_argument('--lr-update', type=str, default='poly',
                      choices=['step', 'poly', 'cosine'],
                      help='learning rate update strategy (default: poly)')
    parser.add_argument('--save-per-epochs', type=int, default=5,
                      help='save checkpoint every N epochs (default: 5)')
    
    # Data settings
    parser.add_argument('--dataset', type=str, default='chase',
                      choices=['chase', 'drive', 'stare'],
                      help='dataset name (default: chase)')
    parser.add_argument('--data-root', '--data_root', type=str, default='/content/drive/MyDrive/ChaseDb1_code/datas',
                      help='path to dataset root directory')
    parser.add_argument('--resize', nargs=2, type=int, default=[960, 960],
                      help='resize dimensions (default: 960 960)')
    parser.add_argument('--train-split', type=float, default=0.8,
                      help='training/validation split ratio (default: 0.8)')
    parser.add_argument('--folds', type=int, default=5,
                      help='number of cross-validation folds (default: 5)')
    
    # Model settings
    parser.add_argument('--input-channels', type=int, default=3,
                      help='number of input channels (default: 3)')
    parser.add_argument('--num-class', type=int, default=1,
                      help='number of classes (default: 1)')
    
    # Experiment settings
    parser.add_argument('--exp-id', type=str, default='chase_db1_exp1',
                      help='experiment identifier')
    parser.add_argument('--device', type=str, 
                      default='cuda' if torch.cuda.is_available() else 'cpu',
                      help='device to use for training')
    
    args = parser.parse_args()
    
    # Use provided data_root if specified, otherwise use default based on dataset
    if args.data_root == './data/CHASE_DB1':  # Only update if using default
        if args.dataset == 'chase':
            args.data_root = './data/CHASE_DB1'
        elif args.dataset == 'drive':
            args.data_root = './data/DRIVE'
        elif args.dataset == 'stare':
            args.data_root = './data/STARE'
    
    return args

def main():
    # Get arguments
    args = get_args()
    
    # Create experiment directory
    os.makedirs(f'models/{args.exp_id}', exist_ok=True)
    
    # Print data directory for verification
    print(f"\nUsing data from: {args.data_root}")
    
    # Example patient IDs (adjust based on your dataset)
    patient_ids = [1, 2, 3, 4, 5]  # Adjust this list based on the available data

    # Initialize datasets and dataloaders
    train_dataset = MyDataset_CHASE(
        args=args,
        train_root=args.data_root,
        pat_ls=patient_ids,  # Pass a valid list of patient IDs
        mode='train'
    )
    
    val_dataset = MyDataset_CHASE(
        args=args,
        train_root=args.data_root,
        pat_ls=patient_ids,  # Use the same or a different list for validation
        mode='val'
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Initialize model
    model = DconnNet(
        num_class=args.num_class,
        input_channels=args.input_channels
    ).to(args.device)
    
    # Initialize solver
    solver = Solver(args)
    
    # Print training configuration
    print("\nTraining Configuration:")
    print(f"Dataset: {args.dataset.upper()}")
    print(f"Data root: {args.data_root}")
    print(f"Image size: {args.resize[0]}x{args.resize[1]}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"LR update strategy: {args.lr_update}")
    print(f"Number of folds: {args.folds}")
    print(f"Device: {args.device}")
    print(f"Experiment ID: {args.exp_id}\n")
    
    # Start training
    print(f"Starting training experiment: {args.exp_id}")
    print(f"Number of training samples: {len(train_dataset)}")
    print(f"Number of validation samples: {len(val_dataset)}")
    
    solver.train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        exp_id=args.exp_id,
        num_epochs=args.epochs
    )

if __name__ == '__main__':
    main()
