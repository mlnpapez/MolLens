"""
SAE Training and Analysis Pipeline for SMILES Molecular Data
Trains a GPT model, extracts sparse features using SAE, and visualizes with sae_dashboard
"""

import os
from typing import List, Tuple
from dataclasses import dataclass

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import GPT2Config, GPT2LMHeadModel
from transformer_lens import HookedTransformer, HookedTransformerConfig
from sae_lens import SAE
from sae_dashboard.sae_vis_data import SaeVisConfig
from sae_dashboard.sae_vis_runner import SaeVisRunner
from sae_dashboard.data_writing_fns import save_feature_centric_vis
from data import SmilesTokenizer, SmilesDataset

os.environ["TOKENIZERS_PARALLELISM"] = "false"
sns.set_style("whitegrid")




# -----------------------------
# Model Training
# -----------------------------
def train_gpt(
    model: GPT2LMHeadModel,
    dataloader: DataLoader,
    epochs: int = 1,
    lr: float = 5e-4,
    device: str = "cuda"
) -> GPT2LMHeadModel:
    """Train GPT model on SMILES data."""
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    
    for epoch in range(epochs):
        loop = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in loop:
            batch = batch.to(device)
            outputs = model(input_ids=batch, labels=batch)
            loss = outputs.loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            loop.set_postfix(loss=f"{loss.item():.4f}")
    
    return model


# -----------------------------
# HookedTransformer Conversion
# -----------------------------
def convert_to_hooked_transformer(
    gpt2_model: GPT2LMHeadModel,
    tokenizer: SmilesTokenizer,
    device: str = "cuda"
) -> HookedTransformer:
    """Convert HuggingFace GPT2 to HookedTransformer for interpretability."""
    cfg = HookedTransformerConfig(
        n_layers=gpt2_model.config.n_layer,
        d_model=gpt2_model.config.n_embd,
        n_ctx=getattr(gpt2_model.config, 'n_positions', 1024),
        d_head=gpt2_model.config.n_embd // gpt2_model.config.n_head,
        n_heads=gpt2_model.config.n_head,
        d_vocab=gpt2_model.config.vocab_size,
        act_fn="gelu",
        d_mlp=gpt2_model.config.n_embd * 4,
        normalization_type="LN",
        device=device,
    )
    
    hooked_model = HookedTransformer(cfg, move_to_device=True)
    hooked_model.load_state_dict(gpt2_model.state_dict(), strict=False)
    hooked_model.tokenizer = tokenizer
    
    return hooked_model


# -----------------------------
# Activation Collection
# -----------------------------
def collect_activations(
    model: HookedTransformer,
    dataloader: DataLoader,
    hook_point: str = "blocks.0.hook_resid_post",
    device: str = "cuda"
) -> torch.Tensor:
    """Collect activations from specific hook point."""
    model.eval()
    activations = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Collecting {hook_point}"):
            _, cache = model.run_with_cache(batch.to(device))
            acts = cache[hook_point]  # [batch, seq, d_model]
            activations.append(acts.reshape(-1, acts.shape[-1]).cpu())

    all_acts = torch.cat(activations, dim=0)
    print(f"Collected {all_acts.shape[0]} activations of dim {all_acts.shape[1]}")
    return all_acts


# -----------------------------
# SAE Training
# -----------------------------
def train_sae(
    activations: torch.Tensor,
    d_in: int,
    d_sae: int = None,
    l1_coeff: float = 1e-3,
    lr: float = 3e-4,
    steps: int = 3000,
    batch_size: int = 1024,
    device: str = "cuda"
) -> Tuple[SAE, List[float]]:
    """Train Sparse Autoencoder on activations."""
    if d_sae is None:
        d_sae = d_in * 4
    
    sae = SAE.from_dict({
        "d_in": d_in,
        "d_sae": d_sae,
        "l1_coefficient": l1_coeff,
        "architecture": "standard",
    })
    sae = sae.to(device)
    
    optimizer = torch.optim.Adam(sae.parameters(), lr=lr)
    sae.train()
    
    losses = []
    recon_losses = []
    l1_losses = []
    
    pbar = tqdm(range(steps), desc="Training SAE")
    for _ in pbar:
        indices = torch.randint(0, activations.shape[0], (batch_size,))
        batch = activations[indices].to(device)

        feature_acts = sae.encode(batch)
        reconstruction = sae.decode(feature_acts)

        recon_loss = (reconstruction - batch).pow(2).mean()
        l1_loss = l1_coeff * feature_acts.abs().sum(dim=-1).mean()
        loss = recon_loss + l1_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        recon_losses.append(recon_loss.item())
        l1_losses.append(l1_loss.item())

        # Update tqdm bar with current losses
        pbar.set_postfix(
            loss=f"{loss.item():.4f}",
            recon=f"{recon_loss.item():.4f}",
            l1=f"{l1_loss.item():.4f}",
        )

    return sae, losses, recon_losses, l1_losses


# -----------------------------
# Dashboard Generation
# -----------------------------
def generate_dashboard(
    sae: SAE,
    model: HookedTransformer,
    tokens: torch.Tensor,
    hook_point: str,
    device: str,
    output_dir: str = "sae_analysis"
) -> None:
    """Generate interactive SAE dashboard."""
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n=== Generating SAE Dashboard ===")
    
    config = SaeVisConfig(
        hook_point=hook_point,
        features=list(range(min(50, sae.cfg.d_sae))),
        minibatch_size_features=16,
        minibatch_size_tokens=64,
        device=device,
        verbose=True,
    )
    
    print("Running SaeVisRunner (this may take a few minutes)...")
    runner = SaeVisRunner(config)
    vis_data = runner.run(
        encoder=sae,
        model=model,
        tokens=tokens[:100]
    )
    
    dashboard_path = os.path.join(output_dir, "feature_dashboard.html")
    save_feature_centric_vis(sae_vis_data=vis_data, filename=dashboard_path)
    
    print(f"✅ Dashboard saved: {dashboard_path}")


# -----------------------------
# Feature Analysis
# -----------------------------
@dataclass
class FeatureAnalysis:
    """Results from SAE feature analysis."""
    smiles: str
    features: torch.Tensor
    recon_error: float
    tokens: List[str]
    l0: float
    max_activating_features: List[Tuple[int, float]]


def analyze_features(
    model: HookedTransformer,
    sae: SAE,
    tokenizer: SmilesTokenizer,
    test_smiles: List[str],
    device: str = "cuda"
) -> List[FeatureAnalysis]:
    """Analyze SAE features on test molecules."""
    model.eval()
    sae.eval()
    results = []
    
    with torch.no_grad():
        for smiles in test_smiles:
            token_ids = tokenizer.encode(smiles)
            input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
            
            _, cache = model.run_with_cache(input_ids)
            hidden = cache["blocks.0.hook_resid_post"]
            
            hidden_flat = hidden.reshape(-1, hidden.shape[-1])
            feature_acts = sae.encode(hidden_flat)
            reconstruction = sae.decode(feature_acts)
            
            recon_error = (hidden_flat - reconstruction).pow(2).mean().item()
            l0 = (feature_acts.abs() > 1e-5).float().sum(dim=-1).mean().item()
            
            # Find top activating features
            max_acts = feature_acts.max(dim=0).values
            top_k = 10
            top_indices = torch.argsort(max_acts, descending=True)[:top_k]
            top_features = [(idx.item(), max_acts[idx].item()) for idx in top_indices]
            
            results.append(FeatureAnalysis(
                smiles=smiles,
                features=feature_acts.cpu(),
                recon_error=recon_error,
                tokens=[tokenizer.i2c[i] for i in token_ids],
                l0=l0,
                max_activating_features=top_features
            ))
    
    return results


# -----------------------------
# Visualizations
# -----------------------------
def plot_training_curves(
    losses: List[float],
    recon_losses: List[float],
    l1_losses: List[float],
    output_dir: str
) -> None:
    """Plot SAE training curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Total loss
    ax1.plot(losses, linewidth=2, color='steelblue')
    ax1.set_xlabel('Training Step')
    ax1.set_ylabel('Total Loss')
    ax1.set_title('SAE Training Loss')
    ax1.grid(True, alpha=0.3)
    
    # Component losses
    ax2.plot(recon_losses, label='Reconstruction', linewidth=2, color='coral')
    ax2.plot(l1_losses, label='L1 Sparsity', linewidth=2, color='mediumseagreen')
    ax2.set_xlabel('Training Step')
    ax2.set_ylabel('Loss')
    ax2.set_title('Loss Components')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/training_curves.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_dir}/training_curves.png")


def plot_feature_heatmaps(results: List[FeatureAnalysis], output_dir: str) -> None:
    """Plot feature activation heatmaps per molecule."""
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(5*n, 15))
    if n == 1:
        axes = [axes]

    for ax, result in zip(axes, results):
        features = result.features.numpy()
        im = ax.imshow(features.T, aspect='auto', cmap='viridis', interpolation='nearest')
        ax.set_xlabel('Token Position', fontsize=10)
        ax.set_ylabel('SAE Feature', fontsize=10)
        ax.set_title(f"{result.smiles} | L0={result.l0:.1f} | Error={result.recon_error:.4f}", fontsize=11)
        
        cbar = plt.colorbar(im, ax=ax, fraction=0.046)
        cbar.set_label('Activation', fontsize=9)
        
        ax.set_xticks(range(len(result.tokens)))
        ax.set_xticklabels(result.tokens, rotation=45, ha='right', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/feature_heatmaps.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_dir}/feature_heatmaps.png")


def plot_sparsity_metrics(results: List[FeatureAnalysis], output_dir: str) -> None:
    """Plot sparsity and reconstruction metrics."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    smiles = [r.smiles for r in results]
    l0_values = [r.l0 for r in results]
    errors = [r.recon_error for r in results]
    
    # L0 sparsity
    bars1 = ax1.bar(range(len(l0_values)), l0_values, color='steelblue', edgecolor='black', linewidth=1.5)
    ax1.set_xticks(range(len(smiles)))
    ax1.set_xticklabels(smiles, rotation=45, ha='right')
    ax1.set_ylabel('L0 (Active Features)')
    ax1.set_title('SAE Sparsity per Molecule')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, val in zip(bars1, l0_values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}', ha='center', va='bottom', fontsize=9)
    
    # Reconstruction error
    bars2 = ax2.bar(range(len(errors)), errors, color='coral', edgecolor='black', linewidth=1.5)
    ax2.set_xticks(range(len(smiles)))
    ax2.set_xticklabels(smiles, rotation=45, ha='right')
    ax2.set_ylabel('Reconstruction Error (MSE)')
    ax2.set_title('SAE Reconstruction Quality')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, val in zip(bars2, errors):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.4f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/sparsity_metrics.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_dir}/sparsity_metrics.png")


def plot_feature_distribution(results: List[FeatureAnalysis], output_dir: str) -> None:
    """Plot feature activation distribution and sparsity."""
    all_features = torch.cat([r.features for r in results], dim=0).numpy()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Activation magnitude distribution
    activations = all_features[all_features > 1e-5]
    ax1.hist(activations.flatten(), bins=100, edgecolor='black', alpha=0.7, color='steelblue')
    ax1.set_xlabel('Activation Magnitude')
    ax1.set_ylabel('Count')
    ax1.set_title('Distribution of Non-Zero Activations')
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)
    
    # Feature usage frequency
    feature_usage = (all_features > 1e-5).mean(axis=0)
    ax2.hist(feature_usage, bins=50, edgecolor='black', alpha=0.7, color='mediumseagreen')
    ax2.axvline(feature_usage.mean(), color='red', linestyle='--', linewidth=2,
               label=f'Mean: {feature_usage.mean():.3f}')
    ax2.set_xlabel('Activation Frequency')
    ax2.set_ylabel('Number of Features')
    ax2.set_title('Feature Sparsity Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/feature_distribution.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_dir}/feature_distribution.png")


def plot_top_features(results: List[FeatureAnalysis], output_dir: str) -> None:
    """Plot top activating features per molecule."""
    n = len(results)
    fig, axes = plt.subplots(n, 1, figsize=(10, 4*n))
    if n == 1:
        axes = [axes]
    
    for ax, result in zip(axes, results):
        features, activations = zip(*result.max_activating_features)
        
        y_pos = np.arange(len(features))
        bars = ax.barh(y_pos, activations, color='steelblue', edgecolor='black', linewidth=1.5)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([f"F{f}" for f in features])
        ax.set_xlabel('Max Activation')
        ax.set_title(f"Top 10 Features: {result.smiles}")
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for bar, val in zip(bars, activations):
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2.,
                   f'{val:.2f}', ha='left', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/top_features.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_dir}/top_features.png")


def plot_feature_correlation(results: List[FeatureAnalysis], output_dir: str) -> None:
    """Plot feature co-activation correlation matrix."""
    all_features = torch.cat([r.features for r in results], dim=0).numpy()
    
    # Binarize activations
    active = (all_features > 1e-5).astype(float)
    
    # Select subset of most active features for visualization
    feature_activity = active.sum(axis=0)
    top_features_idx = np.argsort(feature_activity)[-50:]  # Top 50
    
    # Compute correlation
    corr = np.corrcoef(active[:, top_features_idx].T)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, cmap='coolwarm', center=0, vmin=-1, vmax=1,
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
    plt.title('Feature Co-activation Correlation (Top 50 Features)')
    plt.xlabel('Feature Index')
    plt.ylabel('Feature Index')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/feature_correlation.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_dir}/feature_correlation.png")


def plot_token_attribution(results: List[FeatureAnalysis], output_dir: str) -> None:
    """Plot token-level feature attribution heatmap."""
    n = len(results)
    fig, axes = plt.subplots(n, 1, figsize=(14, 4*n))
    if n == 1:
        axes = [axes]

    for ax, result in zip(axes, results[:n]):
        features = result.features.numpy()
        
        # Get top 20 features for this molecule
        max_acts = features.max(axis=0)
        top_k = min(20, features.shape[1])
        top_indices = np.argsort(max_acts)[-top_k:]

        # Plot heatmap
        im = ax.imshow(features[:, top_indices].T, aspect='auto', cmap='YlOrRd', interpolation='nearest')
        ax.set_xlabel('Token Position', fontsize=10)
        ax.set_ylabel('Feature Index', fontsize=10)
        ax.set_title(f"Token Attribution: {result.smiles}", fontsize=11)

        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Activation', fontsize=9)

        ax.set_xticks(range(len(result.tokens)))
        ax.set_xticklabels(result.tokens, rotation=45, ha='right', fontsize=9)
        ax.set_yticks(range(top_k))
        ax.set_yticklabels([f"F{top_indices[i]}" for i in range(top_k)], fontsize=8)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/token_attribution.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_dir}/token_attribution.png")


# -----------------------------
# Main Pipeline
# -----------------------------
def main():
    """Main training and analysis pipeline."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}\n")
    
    # Load data
    print("=== Loading SMILES Data ===")
    # expects qm9.csv with column 'smile'
    smiles = pd.read_csv("qm9.csv", sep=",", dtype=str)["smile"].values.tolist()
    tokenizer = SmilesTokenizer.from_data(smiles)
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    
    # Prepare dataset
    dataset = SmilesDataset(smiles, tokenizer, max_len=96)
    dataloader = DataLoader(dataset, batch_size=512, shuffle=True)
    
    # Train GPT
    print("\n=== Training GPT Model ===")
    config = GPT2Config(
        n_layer=4,
        n_head=4,
        n_embd=128,
        vocab_size=tokenizer.vocab_size,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    model = GPT2LMHeadModel(config)
    model = train_gpt(model, dataloader, epochs=20, device=device)
    
    # Convert to HookedTransformer
    print("\n=== Converting to HookedTransformer ===")
    hooked_model = convert_to_hooked_transformer(model, tokenizer, device)
    
    # Collect activations
    print("\n=== Collecting Activations ===")
    hook_point = "blocks.0.hook_resid_post"
    activations = collect_activations(
        hooked_model,
        dataloader,
        hook_point,
        device=device
    )
    
    # Train SAE
    print("\n=== Training SAE ===")
    sae, losses, recon_losses, l1_losses = train_sae(
        activations,
        d_in=config.n_embd,
        d_sae=config.n_embd * 4,
        l1_coeff=7e-2,
        lr=3e-4,
        steps=120000,
        device=device
    )
    
    # Prepare tokens for dashboard
    print("\n=== Preparing Token Dataset ===")
    token_list = [torch.tensor(tokenizer.encode(s)) for s in smiles[:200]]
    max_len = max(len(t) for t in token_list)
    padded = [torch.cat([t, torch.full((max_len - len(t),), tokenizer.pad_token_id, dtype=torch.long)]) 
              if len(t) < max_len else t for t in token_list]
    tokens = torch.stack(padded).to(device)
    
    output_dir = "sae_analysis"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate dashboard
    try:
        generate_dashboard(sae, hooked_model, tokens, hook_point, device, output_dir=output_dir)
    except Exception as e:
        print(f"⚠️ Dashboard generation failed: {e}")
    
    # Analyze features
    print("\n=== Analyzing Features ===")
    test_smiles = ["CCO", "CC(C)O", "c1ccccc1", "CC(=O)O", "CCN", "CCCC", "CC(C)C"]
    results = analyze_features(hooked_model, sae, tokenizer, test_smiles, device=device)
    
    # Save results summary
    summary_rows = []
    for r in results:
        top_feats_str = ";".join([f"{idx}:{val:.4f}" for idx, val in r.max_activating_features])
        summary_rows.append({
            "smiles": r.smiles,
            "l0": r.l0,
            "recon_error": r.recon_error,
            "top_features": top_feats_str,
            "n_tokens": len(r.tokens)
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_csv = os.path.join(output_dir, "feature_summary.csv")
    summary_df.to_csv(summary_csv, index=False)
    print(f"✓ Saved summary: {summary_csv}")
    
    # Save raw feature activations (npz)
    for i, r in enumerate(results):
        fname = os.path.join(output_dir, f"features_{i}_{r.smiles.replace('/', '_').replace('\\', '_')}.npz")
        np.savez_compressed(fname, features=r.features.numpy())
        print(f"✓ Saved features: {fname}")
    
    # Plotting
    print("\n=== Generating Plots ===")
    plot_training_curves(losses, recon_losses, l1_losses, output_dir)
    plot_feature_heatmaps(results, output_dir)
    plot_sparsity_metrics(results, output_dir)
    plot_feature_distribution(results, output_dir)
    plot_top_features(results, output_dir)
    plot_feature_correlation(results, output_dir)
    plot_token_attribution(results, output_dir)
    
    print("\nAll done. Outputs are in the directory:", os.path.abspath(output_dir))


if __name__ == "__main__":
    main()
