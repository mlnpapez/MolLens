import os
import warnings
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from dataclasses import dataclass
from typing import List, Tuple
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2Config, GPT2LMHeadModel, Trainer, TrainingArguments
from transformer_lens import HookedTransformer, HookedTransformerConfig
from sae_lens import SAE
from sae_dashboard.sae_vis_data import SaeVisConfig
from sae_dashboard.sae_vis_runner import SaeVisRunner
from sae_dashboard.data_writing_fns import save_feature_centric_vis
from data import SmilesTokenizer, SmilesDataset

os.environ["TOKENIZERS_PARALLELISM"] = "false"
sns.set_style("whitegrid")


# ============================================================================
# Model Training
# ============================================================================

def train_gpt(
    model: GPT2LMHeadModel,
    dataset: Dataset,
    epochs: int,
    lr: float,
    device: str,
    pad_token_id: int = None
) -> GPT2LMHeadModel:
    """Train GPT model using HuggingFace Trainer."""
    training_args = TrainingArguments(
        # output_dir="./gpt_checkpoints",
        num_train_epochs=epochs,
        per_device_train_batch_size=512,
        learning_rate=lr,
        logging_steps=100,
        save_strategy="no",
        report_to="none",
        remove_unused_columns=False,
    )

    def data_collator(examples):
        batch = torch.stack(examples)  # shape: (B, seq_len)
        if pad_token_id is not None:
            attention_mask = (batch != pad_token_id).long()
        else:
            attention_mask = torch.ones_like(batch, dtype=torch.long)
        return {"input_ids": batch, "labels": batch, "attention_mask": attention_mask}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )
    trainer.train()
    return model


def convert_to_hooked(
    gpt2_model: GPT2LMHeadModel,
    tokenizer: SmilesTokenizer
) -> HookedTransformer:
    """Convert GPT2 to HookedTransformer."""
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
    )

    hooked = HookedTransformer(cfg, move_to_device=True)
    hooked.load_state_dict(gpt2_model.state_dict(), strict=False)
    hooked.tokenizer = tokenizer
    return hooked


# ============================================================================
# Activation Collection
# ============================================================================

def collect_activations(
    model: HookedTransformer,
    dataloader: DataLoader,
    hook_point: str
) -> torch.Tensor:
    """Collect activations from hook point."""
    model.eval()
    activations = []

    with torch.no_grad():
        for batch in dataloader:
            _, cache = model.run_with_cache(batch.to(model.cfg.device))
            acts = cache[hook_point]
            activations.append(acts.reshape(-1, acts.shape[-1]).cpu())

    return torch.cat(activations, dim=0)


# ============================================================================
# SAE Training
# ============================================================================

def train_sae(
    activations: torch.Tensor,
    d_in: int,
    d_sae: int = None,
    l1_coeff: float = 1e-3,
    lr: float = 3e-4,
    steps: int = 3000,
    batch_size: int = 1024,
    device: str = "cuda"
) -> Tuple[SAE, dict]:
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

    total_losses = []
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
        total_loss = recon_loss + l1_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        total_losses.append(total_loss.item())
        recon_losses.append(recon_loss.item())
        l1_losses.append(l1_loss.item())

        pbar.set_postfix(
            loss=f"{total_loss.item():.4f}",
            recon=f"{recon_loss.item():.4f}",
            l1=f"{l1_loss.item():.4f}",
        )

    return sae, {"total": total_losses, "recon": recon_losses, "l1": l1_losses}


# ============================================================================
# Feature Analysis
# ============================================================================

@dataclass
class FeatureAnalysis:
    """SAE feature analysis results."""
    smiles: str
    tokens: List[str]
    features: torch.Tensor

    # Reconstruction metrics
    recon_error: float
    explained_variance: float

    # Sparsity metrics
    l0: float
    l1: float
    dead_features: int

    # Feature importance
    max_activating_features: List[tuple[int, float]]
    feature_entropy: float

    # Token-level metrics
    token_variance: np.ndarray
    max_token_activation: float


def compute_feature_metrics(
    features: torch.Tensor,
    reconstruction: torch.Tensor, 
    original: torch.Tensor
) -> dict:
    """Compute feature metrics."""
    active_mask = features.abs() > 1e-5

    return {
        'recon_error': (original - reconstruction).pow(2).mean().item(),
        'explained_variance': 1 - ((original - reconstruction).var() / original.var()).item(),
        'l0': active_mask.float().sum(dim=-1).mean().item(),
        'l1': features.abs().sum(dim=-1).mean().item(),
        'dead_features': (features.abs().max(dim=0).values < 1e-5).sum().item(),
        'feature_entropy': -(features.abs() * torch.log(features.abs() + 1e-10)).sum(dim=-1).mean().item(),
        'token_variance': features.var(dim=0).cpu().numpy(),
        'max_token_activation': features.max().item(),
    }


def analyze_features(
    model: HookedTransformer,
    sae: SAE,
    tokenizer: SmilesTokenizer,
    test_smiles: List[str],
    hook_point: str
) -> List[FeatureAnalysis]:
    """Analyze SAE features on test molecules."""
    model.eval()
    sae.eval()
    results = []

    with torch.no_grad():
        for smiles in test_smiles:
            token_ids = tokenizer.encode(smiles)
            input_ids = torch.tensor([token_ids], dtype=torch.long, device=model.cfg.device)

            _, cache = model.run_with_cache(input_ids)
            hidden = cache[hook_point].reshape(-1, cache[hook_point].shape[-1])

            features = sae.encode(hidden)
            reconstruction = sae.decode(features)

            metrics = compute_feature_metrics(features, reconstruction, hidden)

            # Top features
            max_acts = features.max(dim=0).values
            top_indices = torch.argsort(max_acts, descending=True)[:10]
            top_features = [(idx.item(), max_acts[idx].item()) for idx in top_indices]

            results.append(FeatureAnalysis(
                smiles=smiles,
                tokens=[tokenizer.i2c[i] for i in token_ids],
                features=features.cpu(),
                recon_error=metrics['recon_error'],
                explained_variance=metrics['explained_variance'],
                l0=metrics['l0'],
                l1=metrics['l1'],
                dead_features=metrics['dead_features'],
                max_activating_features=top_features,
                feature_entropy=metrics['feature_entropy'],
                token_variance=metrics['token_variance'],
                max_token_activation=metrics['max_token_activation'],
            ))

    return results


# ============================================================================
# Visualization
# ============================================================================

def create_plots(
    results: List[FeatureAnalysis],
    losses: dict,
    output_dir: str
) -> None:
    """Generate all analysis plots."""
    os.makedirs(output_dir, exist_ok=True)
    smiles = [r.smiles for r in results]
    n = len(results)

    # 1. Training Curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(losses['total'], linewidth=2, color='steelblue')
    ax1.set_xlabel('Training Step')
    ax1.set_ylabel('Total Loss')
    ax1.set_title('SAE Training Loss')
    ax1.grid(True, alpha=0.3)

    ax2.plot(losses['recon'], label='Reconstruction', linewidth=2, color='coral')
    ax2.plot(losses['l1'], label='L1 Sparsity', linewidth=2, color='mediumseagreen')
    ax2.set_xlabel('Training Step')
    ax2.set_ylabel('Loss')
    ax2.set_title('Loss Components')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/training_curves.png", dpi=150, bbox_inches='tight')
    plt.close()

    # 2. Feature Heatmaps
    fig, axes = plt.subplots(1, n, figsize=(5*n, 15))
    if n == 1:
        axes = [axes]

    for ax, r in zip(axes, results):
        im = ax.imshow(r.features.numpy().T, aspect='auto', cmap='viridis')
        ax.set_xlabel('Token Position', fontsize=10)
        ax.set_ylabel('SAE Feature', fontsize=10)
        ax.set_title(f"{r.smiles} | L0={r.l0:.1f} | Error={r.recon_error:.4f}", fontsize=11)
        ax.set_xticks(range(len(r.tokens)))
        ax.set_xticklabels(r.tokens, rotation=45, ha='right', fontsize=8)
        cbar = plt.colorbar(im, ax=ax, fraction=0.046)
        cbar.set_label('Activation', fontsize=9)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/feature_heatmaps.png", dpi=150, bbox_inches='tight')
    plt.close()

    # 3. Sparsity Metrics
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    bars1 = ax1.bar(range(n), [r.l0 for r in results], color='steelblue', edgecolor='black', linewidth=1.5)
    ax1.set_xticks(range(n))
    ax1.set_xticklabels(smiles, rotation=45, ha='right')
    ax1.set_ylabel('L0 (Active Features)')
    ax1.set_title('SAE Sparsity per Molecule')
    ax1.grid(True, alpha=0.3, axis='y')
    for bar, r in zip(bars1, results):
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{r.l0:.1f}', ha='center', va='bottom', fontsize=9)
    
    bars2 = ax2.bar(range(n), [r.recon_error for r in results], color='coral', edgecolor='black', linewidth=1.5)
    ax2.set_xticks(range(n))
    ax2.set_xticklabels(smiles, rotation=45, ha='right')
    ax2.set_ylabel('Reconstruction Error (MSE)')
    ax2.set_title('SAE Reconstruction Quality')
    ax2.grid(True, alpha=0.3, axis='y')
    for bar, r in zip(bars2, results):
        ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{r.recon_error:.4f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/sparsity_metrics.png", dpi=150, bbox_inches='tight')
    plt.close()

    # 4. Feature Distribution
    all_features = torch.cat([r.features for r in results], dim=0).numpy()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    activations = all_features[all_features > 1e-5]
    ax1.hist(activations.flatten(), bins=100, edgecolor='black', alpha=0.7, color='steelblue')
    ax1.set_xlabel('Activation Magnitude')
    ax1.set_ylabel('Count')
    ax1.set_title('Distribution of Non-Zero Activations')
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)

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

    # 5. Top Features
    fig, axes = plt.subplots(n, 1, figsize=(10, 4*n))
    if n == 1:
        axes = [axes]

    for ax, r in zip(axes, results):
        features, activations = zip(*r.max_activating_features)
        y_pos = np.arange(len(features))
        bars = ax.barh(y_pos, activations, color='steelblue', edgecolor='black', linewidth=1.5)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([f"F{f}" for f in features])
        ax.set_xlabel('Max Activation')
        ax.set_title(f"Top 10 Features: {r.smiles}")
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3, axis='x')
        for bar, val in zip(bars, activations):
            ax.text(bar.get_width(), bar.get_y() + bar.get_height()/2.,
                   f'{val:.2f}', ha='left', va='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/top_features.png", dpi=150, bbox_inches='tight')
    plt.close()

    # 6. Feature Correlation
    active = (all_features > 1e-5).astype(float)
    feature_activity = active.sum(axis=0)
    top_features_idx = np.argsort(feature_activity)[-50:]
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

    # 7. Token Attribution
    fig, axes = plt.subplots(n, 1, figsize=(14, 4*n))
    if n == 1:
        axes = [axes]

    for ax, r in zip(axes, results):
        features = r.features.numpy()
        max_acts = features.max(axis=0)
        top_k = min(20, features.shape[1])
        top_indices = np.argsort(max_acts)[-top_k:]

        im = ax.imshow(features[:, top_indices].T, aspect='auto', cmap='YlOrRd')
        ax.set_xlabel('Token Position', fontsize=10)
        ax.set_ylabel('Feature Index', fontsize=10)
        ax.set_title(f"Token Attribution: {r.smiles}", fontsize=11)
        ax.set_xticks(range(len(r.tokens)))
        ax.set_xticklabels(r.tokens, rotation=45, ha='right', fontsize=9)
        ax.set_yticks(range(top_k))
        ax.set_yticklabels([f"F{top_indices[i]}" for i in range(top_k)], fontsize=8)
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Activation', fontsize=9)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/token_attribution.png", dpi=150, bbox_inches='tight')
    plt.close()

    # 8. Enhanced Metrics (new plots)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Explained variance
    axes[0, 0].bar(range(n), [r.explained_variance for r in results], color='steelblue')
    axes[0, 0].set_xticks(range(n))
    axes[0, 0].set_xticklabels(smiles, rotation=45, ha='right')
    axes[0, 0].set_ylabel('R²')
    axes[0, 0].set_title('Explained Variance')
    axes[0, 0].axhline(0.9, color='red', linestyle='--', alpha=0.5)
    axes[0, 0].grid(True, alpha=0.3, axis='y')

    # Dead features
    axes[0, 1].bar(range(n), [r.dead_features for r in results], color='indianred')
    axes[0, 1].set_xticks(range(n))
    axes[0, 1].set_xticklabels(smiles, rotation=45, ha='right')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title('Dead Features')
    axes[0, 1].grid(True, alpha=0.3, axis='y')

    # Feature entropy
    axes[1, 0].bar(range(n), [r.feature_entropy for r in results], color='darkorange')
    axes[1, 0].set_xticks(range(n))
    axes[1, 0].set_xticklabels(smiles, rotation=45, ha='right')
    axes[1, 0].set_ylabel('Entropy')
    axes[1, 0].set_title('Feature Entropy')
    axes[1, 0].grid(True, alpha=0.3, axis='y')

    # Sparsity vs quality scatter
    l0_vals = [r.l0 for r in results]
    ev_vals = [r.explained_variance for r in results]
    axes[1, 1].scatter(l0_vals, ev_vals, s=100, alpha=0.6, color='mediumpurple')
    axes[1, 1].set_xlabel('L0 Sparsity')
    axes[1, 1].set_ylabel('Explained Variance')
    axes[1, 1].set_title('Sparsity vs Reconstruction Quality')
    axes[1, 1].grid(True, alpha=0.3)
    for i, txt in enumerate(smiles):
        axes[1, 1].annotate(txt, (l0_vals[i], ev_vals[i]), fontsize=8, alpha=0.7)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/enhanced_metrics.png", dpi=150, bbox_inches='tight')
    plt.close()


def generate_dashboard(
    sae: SAE,
    model: HookedTransformer,
    tokens: torch.Tensor,
    hook_point: str,
    output_dir: str
) -> None:
    """Generate SAE dashboard."""
    config = SaeVisConfig(
        hook_point=hook_point,
        features=list(range(min(10, sae.cfg.d_sae))),
        minibatch_size_features=16,
        minibatch_size_tokens=64,
        device=str(model.cfg.device),
        verbose=False,
    )

    runner = SaeVisRunner(config)
    vis_data = runner.run(encoder=sae, model=model, tokens=tokens[:100])

    save_feature_centric_vis(
        sae_vis_data=vis_data, 
        filename=os.path.join(output_dir, "dashboard.html")
    )


# ============================================================================
# Main Pipeline
# ============================================================================

def main():
    """Main training and analysis pipeline."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = "sae_analysis"
    os.makedirs(output_dir, exist_ok=True)

    smiles = pd.read_csv("qm9.csv")["smile"].values.tolist()
    tokenizer = SmilesTokenizer.from_data(smiles)
    dataset = SmilesDataset(smiles, tokenizer, max_len=96)
    dataloader = DataLoader(dataset, batch_size=512, shuffle=True)

    config = GPT2Config(
        n_layer=4, n_head=8, n_embd=32, vocab_size=tokenizer.vocab_size,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        loss_type="ForCausalLMLoss",
    )
    model = train_gpt(
        GPT2LMHeadModel(config),
        dataset,
        epochs=20,
        lr=5e-4,
        device=device,
        pad_token_id=tokenizer.pad_token_id
    )

    hooked_model = convert_to_hooked(model, tokenizer)
    hook_point = "blocks.0.hook_resid_post"

    activations = collect_activations(hooked_model, dataloader, hook_point)

    sae, losses = train_sae(
        activations,
        d_in=config.n_embd,
        d_sae=config.n_embd * 4,
        l1_coeff=8e-2,
        lr=3e-4,
        steps=120000,
        device=device
    )

    test_smiles = ["CCO", "CC(C)O", "c1ccccc1", "CC(=O)O", "CCN", "CCCC", "CC(C)C"]
    results = analyze_features(hooked_model, sae, tokenizer, test_smiles, hook_point)

    pd.DataFrame([{
        "smiles": r.smiles,
        "l0": r.l0,
        "l1": r.l1,
        "recon_error": r.recon_error,
        "explained_variance": r.explained_variance,
        "dead_features": r.dead_features,
        "feature_entropy": r.feature_entropy,
    } for r in results]).to_csv(f"{output_dir}/summary.csv", index=False)

    create_plots(results, losses, output_dir)

    tokens = torch.stack([
        torch.cat([torch.tensor(tokenizer.encode(s)), 
                   torch.full((96 - len(tokenizer.encode(s)),), tokenizer.pad_token_id)])
        for s in smiles[200:250]
    ]).to(device)
    generate_dashboard(sae, hooked_model, tokens, hook_point, output_dir)

    print(f"\n✅ Analysis complete. Results in: {os.path.abspath(output_dir)}")


if __name__ == "__main__":
    main()
