import os
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from advanced_quantum_pipeline import AdvancedQuantumPipeline, AdvancedParetoLoss

def train_model(num_epochs=50, learning_rate=1e-4, batch_size=4):
    print("="*50)
    print("INITIALIZING ANATOMICALLY GROUNDED QUANTUM TRAINING")
    print("="*50)

    # 1. Device Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Target Device: {device}")

    # 2. Data Loaders
    print("Loading Multi-Modal Datasets (IDRiD, APTOS, DRIVE)...")
    try:
        from data_loader import get_data_loaders
        train_loader, val_loader = get_data_loaders(batch_size=batch_size)
    except Exception as e:
        print(f"Warning: Dataset paths not found or invalid. Using dummy loaders for prototype verification. Error: {e}")
        # Fallback to dummy data for hackathon presentation if datasets aren't downloaded
        train_loader = [(torch.randn(batch_size, 3, 224, 224), 
                        (torch.randint(0, 5, (batch_size,)), 
                         torch.rand(batch_size, 4), 
                         torch.rand(batch_size, 2), 
                         torch.rand(batch_size, 2), 
                         torch.rand(batch_size, 1, 224, 224))) for _ in range(5)]
        val_loader = train_loader

    # 3. Model & Loss Setup
    model = AdvancedQuantumPipeline().to(device)
    criterion = AdvancedParetoLoss(l1=1.0, l2=1.0, l3=1.0, l4=1.0).to(device)
    
    # 4. Optimizer & Scheduler
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)

    best_val_loss = float('inf')
    os.makedirs('checkpoints', exist_ok=True)

    # 5. The Epoch Loop
    print("\nStarting Training Loop...")
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        # Wrapping loader with tqdm for clinical UI progress bar
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        
        for batch_idx, (images, targets) in enumerate(train_pbar):
            images = images.to(device)
            # targets is a tuple: (severity, lesions, fovea, optic_disc, manifold)
            targets = tuple(t.to(device) for t in targets)
            
            # Forward Pass
            optimizer.zero_grad()
            preds = model(images)
            
            # Composite Multi-Task Loss Calculation
            loss, metrics = criterion(preds, targets)
            
            # Backward Pass & Optimize (Quantum + Classical Gradients)
            loss.backward()
            
            # Gradient Clipping to stabilize Quantum gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            running_loss += loss.item()
            train_pbar.set_postfix({'Total Loss': f'{loss.item():.4f}', 'Ord': f'{metrics["loss_ordinal"]:.2f}'})
            
        avg_train_loss = running_loss / len(train_loader)
        
        # Validation Phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Valid]")
            for images, targets in val_pbar:
                images = images.to(device)
                targets = tuple(t.to(device) for t in targets)
                
                preds = model(images)
                loss, _ = criterion(preds, targets)
                val_loss += loss.item()
                
        avg_val_loss = val_loss / len(val_loader)
        scheduler.step()
        
        print(f"\n[Epoch {epoch+1}/{num_epochs}] Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        
        # Save Best Model Checkpoint
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'checkpoints/best_quantum_model.pth')
            print(f"--> Saved new best model (Val Loss: {best_val_loss:.4f})")

    print("\nTraining Complete! Best model saved to 'checkpoints/best_quantum_model.pth'")

if __name__ == "__main__":
    train_model(num_epochs=5) # Set low for rapid prototype testing
