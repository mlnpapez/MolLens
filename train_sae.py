import os
import argparse
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
from sae_lens import SAE

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def load_activation_context(path: str):
    """Load ActivationContext from disk."""
    data = torch.load(path)
    return data


def create_activation_dataloader(activations: torch.Tensor, batch_size: int, shuffle: bool):
    """Create DataLoader from activation tensors."""
    dataset = TensorDataset(activations)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def validate_sae(sae, val_loader, l1_coeff, device):
    """Validate SAE on validation set."""
    sae.eval()
    total_loss = 0.0
    recon_loss_sum = 0.0
    l1_loss_sum = 0.0
    n_batches = 0
    
    with torch.no_grad():
        for batch in val_loader:
            batch = batch[0].to(device)  # TensorDataset wraps in tuple
            
            feature_acts = sae.encode(batch)
            reconstruction = sae.decode(feature_acts)
            
            recon_loss = (reconstruction - batch).pow(2).mean()
            l1_loss = l1_coeff * feature_acts.abs().sum(dim=-1).mean()
            loss = recon_loss + l1_loss
            
            total_loss += loss.item()
            recon_loss_sum += recon_loss.item()
            l1_loss_sum += l1_loss.item()
            n_batches += 1
    
    return {
        'total': total_loss / n_batches,
        'recon': recon_loss_sum / n_batches,
        'l1': l1_loss_sum / n_batches
    }


def train_sae(
    train_loader: DataLoader,
    val_loader: DataLoader,
    d_in: int,
    d_sae: int,
    l1_coeff: float,
    lr: float,
    epochs: int,
    device: str,
    output_dir: str
):
    """Train SAE with validation and best checkpoint saving."""
    
    # Initialize SAE
    sae = SAE.from_dict({
        "d_in": d_in,
        "d_sae": d_sae,
        "l1_coefficient": l1_coeff,
        "architecture": "standard",
    })
    sae = sae.to(device)
    
    optimizer = torch.optim.Adam(sae.parameters(), lr=lr)
    
    # Track training history
    train_losses = {'total': [], 'recon': [], 'l1': []}
    val_losses = {'total': [], 'recon': [], 'l1': []}
    
    # Track feature usage
    feature_max_activations = torch.zeros(d_sae, device=device)
    
    best_val_loss = float('inf')
    best_checkpoint_path = os.path.join(output_dir, "best_sae_model.pt")
    
    print(f"\nTraining SAE: d_in={d_in}, d_sae={d_sae}, l1_coeff={l1_coeff}")
    
    for epoch in range(epochs):
        # Training
        sae.train()
        epoch_total_loss = 0.0
        epoch_recon_loss = 0.0
        epoch_l1_loss = 0.0
        n_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in pbar:
            batch = batch[0].to(device)  # TensorDataset wraps in tuple
            
            feature_acts = sae.encode(batch)
            reconstruction = sae.decode(feature_acts)
            
            recon_loss = (reconstruction - batch).pow(2).mean()
            l1_loss = l1_coeff * feature_acts.abs().sum(dim=-1).mean()
            total_loss = recon_loss + l1_loss
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            # Track max activation per feature
            with torch.no_grad():
                batch_max = feature_acts.abs().max(dim=0).values
                feature_max_activations = torch.max(feature_max_activations, batch_max)
            
            epoch_total_loss += total_loss.item()
            epoch_recon_loss += recon_loss.item()
            epoch_l1_loss += l1_loss.item()
            n_batches += 1
            
            pbar.set_postfix(
                loss=f"{total_loss.item():.4f}",
                recon=f"{recon_loss.item():.4f}",
                l1=f"{l1_loss.item():.4f}"
            )
        
        # Average training losses
        avg_train_loss = {
            'total': epoch_total_loss / n_batches,
            'recon': epoch_recon_loss / n_batches,
            'l1': epoch_l1_loss / n_batches
        }
        train_losses['total'].append(avg_train_loss['total'])
        train_losses['recon'].append(avg_train_loss['recon'])
        train_losses['l1'].append(avg_train_loss['l1'])
        
        # Validation
        val_loss = validate_sae(sae, val_loader, l1_coeff, device)
        val_losses['total'].append(val_loss['total'])
        val_losses['recon'].append(val_loss['recon'])
        val_losses['l1'].append(val_loss['l1'])
        
        # Count dead features
        dead_features_mask = feature_max_activations < 1e-5
        dead_count = dead_features_mask.sum().item()
        
        print(f"Epoch {epoch+1}/{epochs} - "
              f"Train Loss: {avg_train_loss['total']:.4f} - "
              f"Val Loss: {val_loss['total']:.4f} - "
              f"Dead Features: {dead_count}/{d_sae}")
        
        # Save best checkpoint
        if val_loss['total'] < best_val_loss:
            best_val_loss = val_loss['total']
            torch.save({
                'model_state_dict': sae.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'val_loss': val_loss,
                'feature_max_activations': feature_max_activations.cpu(),
                'dead_features_mask': dead_features_mask.cpu(),
            }, best_checkpoint_path)
            print(f"  → Saved best checkpoint (val_loss: {best_val_loss:.4f})")
    
    # Save final training history
    history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'feature_max_activations': feature_max_activations.cpu(),
        'dead_features_mask': dead_features_mask.cpu(),
    }
    history_path = os.path.join(output_dir, "sae_training_history.pt")
    torch.save(history, history_path)
    print(f"\nSaved training history to {history_path}")
    
    return sae, history


def main():
    parser = argparse.ArgumentParser(description="Train Sparse Autoencoder (SAE)")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory with activations and metadata")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory (defaults to input_dir)")
    
    # SAE architecture
    parser.add_argument("--d_sae", type=int, default=None, help="SAE hidden dimension (defaults to 4*d_in)")
    parser.add_argument("--l1_coeff", type=float, default=1e-3, help="L1 sparsity coefficient")
    
    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    
    args = parser.parse_args()
    
    if args.output_dir is None:
        args.output_dir = args.input_dir
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load metadata
    metadata_path = os.path.join(args.input_dir, "metadata.pt")
    print(f"\nLoading metadata from {metadata_path}...")
    metadata = torch.load(metadata_path)
    d_in = metadata['d_model']
    
    if args.d_sae is None:
        args.d_sae = d_in * 4
    
    print(f"Model dimension: {d_in}")
    print(f"SAE dimension: {args.d_sae}")
    
    # Load activations
    print("\nLoading activations...")
    train_data = load_activation_context(os.path.join(args.input_dir, "train_activations.pt"))
    val_data = load_activation_context(os.path.join(args.input_dir, "val_activations.pt"))
    
    train_activations = train_data['activations']
    val_activations = val_data['activations']
    
    print(f"Train activations: {train_activations.shape}")
    print(f"Val activations: {val_activations.shape}")
    
    # Create dataloaders
    train_loader = create_activation_dataloader(train_activations, args.batch_size, shuffle=True)
    val_loader = create_activation_dataloader(val_activations, args.batch_size, shuffle=False)
    
    # Train SAE
    sae, history = train_sae(
        train_loader=train_loader,
        val_loader=val_loader,
        d_in=d_in,
        d_sae=args.d_sae,
        l1_coeff=args.l1_coeff,
        lr=args.lr,
        epochs=args.epochs,
        device=device,
        output_dir=args.output_dir
    )
    
    print(f"\n✅ SAE training complete. Best checkpoint saved to: {os.path.abspath(args.output_dir)}")


if __name__ == "__main__":
    main()