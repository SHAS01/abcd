class Solver(object):
    def __init__(self, args):
        self.args = args
        self.lr = args.lr
        self.best_metrics = {
            'dice': 0.0,
            'cl_dice': 0.0,
            'iou': 0.0,
            'beta': 0.0
        }
        self.best_epoch = 0
        self.loss_func = connect_loss(args)
        self.metrics = Metrics()

    def train(self, model, train_loader, val_loader, exp_id, num_epochs=10):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        # Initialize optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        
        # Load checkpoint if exists
        start_epoch = 0
        checkpoint_path = f'models/{exp_id}/best_model.pth'
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            # Don't load optimizer state or epoch number to start fresh
            prev_metrics = checkpoint['metrics']
            print(f'Loaded previous best model with metrics {prev_metrics}')
        
        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=3, verbose=True
        )
        
        scaler = amp.GradScaler('cuda')
        print('START TRAIN.')
        
        for epoch in range(start_epoch, num_epochs):
            model.train()
            epoch_loss = 0
            num_batches = len(train_loader)
            
            print(f'\nEpoch {epoch}/{num_epochs-1}')
            print('-' * 10)
            
            for i_batch, (images, targets) in enumerate(train_loader):
                try:
                    images = images.to(device)
                    targets = targets.float().to(device)
                    
                    optimizer.zero_grad()
                    
                    with amp.autocast('cuda'):
                        outputs = model(images)
                        loss = self.loss_func(outputs, targets)
                    
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    
                    epoch_loss += loss.item()
                    
                    # Print progress every batch
                    print(f'[Epoch: {epoch}][Iter: {i_batch}/{num_batches}] '
                          f'Loss: {loss.item():.3f}, '
                          f'LR: {optimizer.param_groups[0]["lr"]:.6f}')
                    
                except Exception as e:
                    print(f"Error in batch {i_batch}: {str(e)}")
                    continue
            
            # Calculate average epoch loss
            avg_loss = epoch_loss / num_batches
            print(f'Epoch {epoch} average loss: {avg_loss:.3f}')
            
            # Validation phase
            val_metrics = self.validate(model, val_loader, epoch, exp_id)
            
            # Update learning rate based on validation Dice score
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
            
            # Regular checkpoint
            if (epoch + 1) % self.args.save_per_epochs == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'metrics': val_metrics,
                }, f'models/{exp_id}/epoch_{epoch+1}_model.pth')

        print(f'Best performance at epoch {self.best_epoch} with metrics {self.best_metrics}')
        print('FINISH.')

    def validate(self, model, loader, epoch, exp_id):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.eval()
        metrics_sum = {
            'dice': 0.0,
            'cl_dice': 0.0,
            'iou': 0.0,
            'beta': 0.0
        }
        num_samples = 0
        
        print('\nValidation Progress:')
        with torch.no_grad():
            for i, (images, targets) in enumerate(loader):
                try:
                    images = images.to(device)
                    targets = targets.to(device)
                    
                    outputs = model(images)
                    predictions = (torch.sigmoid(outputs) > 0.5).float()
                    
                    # Calculate metrics
                    metrics_sum['dice'] += self.metrics.calc_dice(predictions, targets)
                    metrics_sum['cl_dice'] += self.metrics.calc_centerline_dice(predictions, targets)
                    metrics_sum['iou'] += self.metrics.calc_iou(predictions, targets)
                    metrics_sum['beta'] += self.metrics.calc_beta_score(predictions, targets)
                    num_samples += 1
                    
                    print(f'Batch [{i+1}/{len(loader)}] - '
                          f'Dice: {metrics_sum["dice"]/num_samples:.4f}, '
                          f'clDice: {metrics_sum["cl_dice"]/num_samples:.4f}, '
                          f'IoU: {metrics_sum["iou"]/num_samples:.4f}, '
                          f'β-score: {metrics_sum["beta"]/num_samples:.4f}')
                
                except Exception as e:
                    print(f"Error in validation batch {i}: {str(e)}")
                    continue
        
        # Calculate averages
        avg_metrics = {k: v/max(num_samples, 1) for k, v in metrics_sum.items()}
        
        print(f'\nValidation Summary:')
        print(f'Average Dice Score: {avg_metrics["dice"]:.4f}')
        print(f'Average clDice Score: {avg_metrics["cl_dice"]:.4f}')
        print(f'Average IoU Score: {avg_metrics["iou"]:.4f}')
        print(f'Average β-score: {avg_metrics["beta"]:.4f}')
        
        return avg_metrics