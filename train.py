import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
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
        num_train_epochs=epochs,
        per_device_train_batch_size=512,
        learning_rate=lr,
        logging_steps=100,
        save_strategy="no",
        report_to="none",
        remove_unused_columns=False,
    )

    def data_collator(examples):
        batch = torch.stack(examples)
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


@dataclass
class ActivationContext:
    """Store activations with their context."""
    activations: torch.Tensor  # (batch*seq, d_model)
    molecule_ids: List[int]    # Which molecule each activation came from
    positions: List[int]       # Which token position
    tokens: List[str]          # The actual token
    smiles: List[str]          # The full SMILES string

def collect_activations_with_context(
    model: HookedTransformer,
    dataloader: DataLoader,
    hook_point: str,
    tokenizer: SmilesTokenizer,
    smiles_list: List[str]
) -> ActivationContext:
    """
    Collect activations while preserving position and molecule context.
    """
    model.eval()
    all_activations = []
    all_molecule_ids = []
    all_positions = []
    all_tokens = []
    all_smiles = []
    
    batch_offset = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Collecting activations")):
            batch_size = batch.shape[0]
            seq_len = batch.shape[1]
            
            _, cache = model.run_with_cache(batch.to(model.cfg.device))
            acts = cache[hook_point]  # (batch, seq, d_model)
            
            # Flatten but track context
            flat_acts = acts.reshape(-1, acts.shape[-1])
            all_activations.append(flat_acts.cpu())
            
            # Track which molecule and position each activation came from
            for b in range(batch_size):
                mol_idx = batch_offset + b
                for pos in range(seq_len):
                    all_molecule_ids.append(mol_idx)
                    all_positions.append(pos)
                    token_id = batch[b, pos].item()
                    all_tokens.append(tokenizer.i2c.get(token_id, '<unk>'))
                    all_smiles.append(smiles_list[mol_idx] if mol_idx < len(smiles_list) else '')
            
            batch_offset += batch_size

    return ActivationContext(
        activations=torch.cat(all_activations, dim=0),
        molecule_ids=all_molecule_ids,
        positions=all_positions,
        tokens=all_tokens,
        smiles=all_smiles
    )


# ============================================================================
# SAE Training with Dead Feature Tracking
# ============================================================================

def train_sae(
    activation_context: ActivationContext,
    d_in: int,
    d_sae: int = None,
    l1_coeff: float = 1e-3,
    lr: float = 3e-4,
    steps: int = 3000,
    batch_size: int = 1024,
    device: str = "cuda"
) -> Tuple[SAE, dict]:
    """
    Train SAE and track dead features.
    """
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

    activations = activation_context.activations
    total_losses = []
    recon_losses = []
    l1_losses = []

    # Track feature usage across all training
    feature_max_activations = torch.zeros(d_sae, device=device)

    pbar = tqdm(range(steps), desc="Training SAE")
    for step in pbar:
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

        # Track max activation per feature across training
        with torch.no_grad():
            batch_max = feature_acts.abs().max(dim=0).values
            feature_max_activations = torch.max(feature_max_activations, batch_max)

        total_losses.append(total_loss.item())
        recon_losses.append(recon_loss.item())
        l1_losses.append(l1_loss.item())

        if step % 100 == 0:
            dead_count = (feature_max_activations < 1e-5).sum().item()
            pbar.set_postfix(
                loss=f"{total_loss.item():.4f}",
                recon=f"{recon_loss.item():.4f}",
                l1=f"{l1_loss.item():.4f}",
                dead=dead_count
            )

    # Final dead feature count
    dead_features_mask = feature_max_activations < 1e-5
    
    return sae, {
        "total": total_losses, 
        "recon": recon_losses, 
        "l1": l1_losses,
        "dead_features_mask": dead_features_mask.cpu(),
        "feature_max_activations": feature_max_activations.cpu()
    }


# ============================================================================
# Feature Analysis
# ============================================================================

@dataclass
class FeatureAnalysis:
    """SAE feature analysis results."""
    smiles: str
    tokens: List[str]
    features: torch.Tensor  # (seq_len, d_sae)

    # Per-molecule reconstruction metrics
    recon_error: float
    explained_variance: float

    # Sparsity metrics
    l0_mean: float  # Average active features per token
    l0_std: float   # Variance in sparsity across tokens
    l1: float
    sparsity_gini: float  # Sparsity measure

    # Per-position metrics
    position_sparsity: List[float]  # L0 per position

    # Feature importance with context
    max_activating_features: List[Dict]  # {feature_idx, activation, position, token}

    # Baseline comparison data
    neuron_recon_error: Optional[float] = None


def compute_gini_coefficient(activations: torch.Tensor) -> float:
    """
    Gini coefficient measures inequality in activation distribution.
    """
    abs_acts = activations.abs().flatten()
    abs_acts = abs_acts[abs_acts > 1e-10]  # Only non-zero
    if len(abs_acts) == 0:
        return 1.0
    
    sorted_acts = torch.sort(abs_acts)[0]
    n = len(sorted_acts)
    index = torch.arange(1, n + 1, dtype=torch.float32, device=activations.device)
    return (2 * (index * sorted_acts).sum() / (n * sorted_acts.sum()) - (n + 1) / n).item()


def analyze_features_with_context(
    model: HookedTransformer,
    sae: SAE,
    tokenizer: SmilesTokenizer,
    test_smiles: List[str],
    hook_point: str,
    dead_features_mask: torch.Tensor
) -> List[FeatureAnalysis]:
    """
    Feature analysis with proper context.
    """
    model.eval()
    sae.eval()
    results = []

    with torch.no_grad():
        for smiles in test_smiles:
            token_ids = tokenizer.encode(smiles)
            input_ids = torch.tensor([token_ids], dtype=torch.long, device=model.cfg.device)

            _, cache = model.run_with_cache(input_ids)
            hidden = cache[hook_point].squeeze(0)  # (seq_len, d_model)

            # SAE reconstruction
            features = sae.encode(hidden)  # (seq_len, d_sae)
            reconstruction = sae.decode(features)

            # Baseline: direct neuron basis
            neuron_recon_error = None  # Identity reconstruction error is 0

            # Per-molecule metrics
            recon_error = (reconstruction - hidden).pow(2).mean().item()
            explained_variance = 1 - (reconstruction - hidden).var().item() / hidden.var().item()

            # Sparsity metrics
            active_mask = features.abs() > 1e-5
            l0_per_token = active_mask.float().sum(dim=-1)  # (seq_len,)
            l0_mean = l0_per_token.mean().item()
            l0_std = l0_per_token.std().item()
            l1 = features.abs().sum(dim=-1).mean().item()
            sparsity_gini = compute_gini_coefficient(features)

            # Per-position sparsity
            position_sparsity = l0_per_token.cpu().tolist()

            # Top features with context
            max_acts_per_feature = features.abs().max(dim=0)
            top_k = min(10, features.shape[1])
            top_indices = torch.argsort(max_acts_per_feature.values, descending=True)[:top_k]

            max_activating_features = []
            for idx in top_indices:
                feature_idx = idx.item()
                # Find which position had max activation
                feature_acts = features[:, feature_idx].abs()
                max_pos = feature_acts.argmax().item()
                max_val = feature_acts[max_pos].item()

                max_activating_features.append({
                    'feature_idx': feature_idx,
                    'activation': max_val,
                    'position': max_pos,
                    'token': tokenizer.i2c[token_ids[max_pos]],
                    'is_dead': dead_features_mask[feature_idx].item()
                })

            results.append(FeatureAnalysis(
                smiles=smiles,
                tokens=[tokenizer.i2c[i] for i in token_ids],
                features=features.cpu(),
                recon_error=recon_error,
                explained_variance=explained_variance,
                l0_mean=l0_mean,
                l0_std=l0_std,
                l1=l1,
                sparsity_gini=sparsity_gini,
                position_sparsity=position_sparsity,
                max_activating_features=max_activating_features,
                neuron_recon_error=neuron_recon_error,
            ))

    return results


# ============================================================================
# Max-Activating Examples
# ============================================================================

@dataclass
class FeatureInterpretation:
    """Interpretation of a single SAE feature."""
    feature_idx: int
    max_activating_examples: List[Dict]  # {smiles, token, position, activation}
    activation_frequency: float  # How often this feature activates
    mean_activation: float  # Average activation when active
    is_dead: bool
    co_activating_features: List[Tuple[int, float]]  # Other features that co-activate


def interpret_features(
    sae: SAE,
    model: HookedTransformer,
    activation_context: ActivationContext,
    hook_point: str,
    top_k_examples: int = 20,
    top_k_features: int = 50
) -> List[FeatureInterpretation]:
    """
    Interpret SAE features by finding max-activating examples.
    This is essential for understanding what each feature detects.
    """
    model.eval()
    sae.eval()

    print("Computing feature activations for entire dataset...")

    # Get all feature activations
    with torch.no_grad():
        all_feature_acts = sae.encode(activation_context.activations.to(sae.cfg.device))

    all_feature_acts = all_feature_acts.cpu()
    n_features = all_feature_acts.shape[1]

    interpretations = []

    # Analyze top-K most active features
    feature_max_acts = all_feature_acts.abs().max(dim=0).values
    top_feature_indices = torch.argsort(feature_max_acts, descending=True)[:top_k_features]

    for feature_idx in tqdm(top_feature_indices, desc="Interpreting features"):
        feature_idx = feature_idx.item()
        feature_acts = all_feature_acts[:, feature_idx]

        # Find max-activating examples
        top_activations = torch.topk(feature_acts.abs(), min(top_k_examples, len(feature_acts)))

        max_examples = []
        for idx, act_val in zip(top_activations.indices, top_activations.values):
            idx = idx.item()
            max_examples.append({
                'smiles': activation_context.smiles[idx],
                'token': activation_context.tokens[idx],
                'position': activation_context.positions[idx],
                'activation': act_val.item(),
                'molecule_id': activation_context.molecule_ids[idx]
            })

        # Activation statistics
        active_mask = feature_acts.abs() > 1e-5
        activation_freq = active_mask.float().mean().item()
        mean_activation = feature_acts[active_mask].abs().mean().item() if active_mask.any() else 0.0
        is_dead = activation_freq < 0.001

        # Co-activation analysis (within same forward pass)
        co_activating = []
        if not is_dead:
            # Get activations from same molecules as top activations
            molecule_ids = [ex['molecule_id'] for ex in max_examples[:5]]
            mask = torch.tensor([mid in molecule_ids for mid in activation_context.molecule_ids])

            if mask.any():
                coactive_features = all_feature_acts[mask]
                feature_corr = torch.corrcoef(torch.stack([
                    feature_acts[mask],
                    *[coactive_features[:, i] for i in range(min(100, n_features)) if i != feature_idx]
                ]).T.T)

                if feature_corr.shape[0] > 1:
                    corr_values = feature_corr[0, 1:]
                    top_corr_idx = torch.argsort(corr_values.abs(), descending=True)[:5]
                    co_activating = [(idx.item(), corr_values[idx].item()) for idx in top_corr_idx]

        interpretations.append(FeatureInterpretation(
            feature_idx=feature_idx,
            max_activating_examples=max_examples,
            activation_frequency=activation_freq,
            mean_activation=mean_activation,
            is_dead=is_dead,
            co_activating_features=co_activating
        ))

    return interpretations


# ============================================================================
# Visualization
# ============================================================================

def create_plots(
    results: List[FeatureAnalysis],
    losses: dict,
    interpretations: List[FeatureInterpretation],
    output_dir: str
) -> None:
    """Generate comprehensive analysis plots."""
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

    # 2. Feature Heatmaps with Position Info
    fig, axes = plt.subplots(1, n, figsize=(5*n, 15))
    if n == 1:
        axes = [axes]

    for ax, r in zip(axes, results):
        im = ax.imshow(r.features.numpy().T, aspect='auto', cmap='viridis')
        ax.set_xlabel('Token Position', fontsize=10)
        ax.set_ylabel('SAE Feature', fontsize=10)
        ax.set_title(f"{r.smiles}\nL0={r.l0_mean:.1f}±{r.l0_std:.1f} | Gini={r.sparsity_gini:.3f}", fontsize=11)
        ax.set_xticks(range(len(r.tokens)))
        ax.set_xticklabels(r.tokens, rotation=45, ha='right', fontsize=8)
        cbar = plt.colorbar(im, ax=ax, fraction=0.046)
        cbar.set_label('Activation', fontsize=9)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/feature_heatmaps.png", dpi=150, bbox_inches='tight')
    plt.close()

    # 3. Sparsity Metrics
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # L0 with error bars
    l0_means = [r.l0_mean for r in results]
    l0_stds = [r.l0_std for r in results]
    axes[0, 0].bar(range(n), l0_means, yerr=l0_stds, capsize=5, color='steelblue', edgecolor='black')
    axes[0, 0].set_xticks(range(n))
    axes[0, 0].set_xticklabels(smiles, rotation=45, ha='right')
    axes[0, 0].set_ylabel('L0 (Active Features per Token)')
    axes[0, 0].set_title('Sparsity: Mean ± Std across Positions')
    axes[0, 0].grid(True, alpha=0.3, axis='y')

    # Gini coefficient
    axes[0, 1].bar(range(n), [r.sparsity_gini for r in results], color='mediumseagreen', edgecolor='black')
    axes[0, 1].set_xticks(range(n))
    axes[0, 1].set_xticklabels(smiles, rotation=45, ha='right')
    axes[0, 1].set_ylabel('Gini Coefficient')
    axes[0, 1].set_title('Activation Inequality (higher = sparser)')
    axes[0, 1].axhline(0.5, color='red', linestyle='--', alpha=0.5, label='Moderate')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3, axis='y')

    # Reconstruction error
    axes[1, 0].bar(range(n), [r.recon_error for r in results], color='coral', edgecolor='black')
    axes[1, 0].set_xticks(range(n))
    axes[1, 0].set_xticklabels(smiles, rotation=45, ha='right')
    axes[1, 0].set_ylabel('MSE')
    axes[1, 0].set_title('Reconstruction Error')
    axes[1, 0].grid(True, alpha=0.3, axis='y')

    # Explained variance
    axes[1, 1].bar(range(n), [r.explained_variance for r in results], color='mediumpurple', edgecolor='black')
    axes[1, 1].set_xticks(range(n))
    axes[1, 1].set_xticklabels(smiles, rotation=45, ha='right')
    axes[1, 1].set_ylabel('R²')
    axes[1, 1].set_title('Explained Variance')
    axes[1, 1].axhline(0.9, color='red', linestyle='--', alpha=0.5)
    axes[1, 1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(f"{output_dir}/enhanced_metrics.png", dpi=150, bbox_inches='tight')
    plt.close()

    # 4. Position-wise Sparsity
    fig, axes = plt.subplots(n, 1, figsize=(12, 4*n))
    if n == 1:
        axes = [axes]

    for ax, r in zip(axes, results):
        ax.plot(r.position_sparsity, marker='o', linewidth=2, markersize=6, color='steelblue')
        ax.set_xticks(range(len(r.tokens)))
        ax.set_xticklabels(r.tokens, rotation=45, ha='right')
        ax.set_ylabel('Active Features (L0)')
        ax.set_xlabel('Token Position')
        ax.set_title(f"Position-wise Sparsity: {r.smiles}")
        ax.grid(True, alpha=0.3)
        ax.axhline(r.l0_mean, color='red', linestyle='--', alpha=0.5, label=f'Mean: {r.l0_mean:.1f}')
        ax.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/position_sparsity.png", dpi=150, bbox_inches='tight')
    plt.close()

    # 5. Top Features With Context
    fig, axes = plt.subplots(n, 1, figsize=(12, 4*n))
    if n == 1:
        axes = [axes]

    for ax, r in zip(axes, results):
        features = [f['feature_idx'] for f in r.max_activating_features]
        activations = [f['activation'] for f in r.max_activating_features]
        labels = [f"F{f['feature_idx']}\n@{f['token']}\npos={f['position']}" 
                  for f in r.max_activating_features]

        colors = ['red' if f['is_dead'] else 'steelblue' for f in r.max_activating_features]

        y_pos = np.arange(len(features))
        bars = ax.barh(y_pos, activations, color=colors, edgecolor='black', linewidth=1.5)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel('Max Activation')
        ax.set_title(f"Top Features with Context: {r.smiles}\n(Red = dead features)")
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/top_features_with_context.png", dpi=150, bbox_inches='tight')
    plt.close()

    # 6. Feature Interpretations
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Activation frequency distribution
    act_freqs = [interp.activation_frequency for interp in interpretations]
    axes[0, 0].hist(act_freqs, bins=30, color='steelblue', edgecolor='black', alpha=0.7)
    axes[0, 0].set_xlabel('Activation Frequency')
    axes[0, 0].set_ylabel('Number of Features')
    axes[0, 0].set_title('Feature Activation Frequency Distribution')
    axes[0, 0].axvline(np.mean(act_freqs), color='red', linestyle='--', label=f'Mean: {np.mean(act_freqs):.3f}')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Dead features
    dead_count = sum(1 for interp in interpretations if interp.is_dead)
    alive_count = len(interpretations) - dead_count
    axes[0, 1].bar(['Alive', 'Dead'], [alive_count, dead_count], color=['green', 'red'], edgecolor='black')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title(f'Feature Status (Dead = {dead_count}/{len(interpretations)})')
    axes[0, 1].grid(True, alpha=0.3, axis='y')

    # Mean activation when active
    mean_acts = [interp.mean_activation for interp in interpretations if not interp.is_dead]
    axes[1, 0].hist(mean_acts, bins=30, color='coral', edgecolor='black', alpha=0.7)
    axes[1, 0].set_xlabel('Mean Activation (when active)')
    axes[1, 0].set_ylabel('Number of Features')
    axes[1, 0].set_title('Feature Strength Distribution')
    axes[1, 0].grid(True, alpha=0.3)

    # Top activating feature examples
    top_5 = sorted(interpretations, key=lambda x: x.mean_activation, reverse=True)[:5]
    feature_names = [f"F{interp.feature_idx}" for interp in top_5]
    mean_acts_top = [interp.mean_activation for interp in top_5]
    axes[1, 1].barh(range(len(top_5)), mean_acts_top, color='mediumpurple', edgecolor='black')
    axes[1, 1].set_yticks(range(len(top_5)))
    axes[1, 1].set_yticklabels(feature_names)
    axes[1, 1].set_xlabel('Mean Activation')
    axes[1, 1].set_title('Top 5 Strongest Features')
    axes[1, 1].invert_yaxis()
    axes[1, 1].grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/feature_interpretations.png", dpi=150, bbox_inches='tight')
    plt.close()

    # 7. Max-Activating Examples per Feature
    # Create detailed report for top features
    with open(f"{output_dir}/feature_interpretations.txt", "w") as f:
        f.write("=" * 80 + "\n")
        f.write("SAE FEATURE INTERPRETATION REPORT\n")
        f.write("=" * 80 + "\n\n")

        for interp in interpretations[:20]:  # Top 20 features
            f.write(f"\nFeature {interp.feature_idx}\n")
            f.write(f"{'-' * 40}\n")
            f.write(f"Activation Frequency: {interp.activation_frequency:.4f}\n")
            f.write(f"Mean Activation (when active): {interp.mean_activation:.4f}\n")
            f.write(f"Status: {'DEAD' if interp.is_dead else 'ACTIVE'}\n")
            f.write(f"\nTop Activating Examples:\n")

            for i, ex in enumerate(interp.max_activating_examples[:10], 1):
                f.write(f"  {i}. {ex['smiles'][:30]:30s} | Token: {ex['token']:5s} | "
                    f"Pos: {ex['position']:2d} | Act: {ex['activation']:.4f}\n")

            if interp.co_activating_features:
                f.write(f"\nCo-activating Features:\n")
                for feat_idx, corr in interp.co_activating_features:
                    f.write(f"  Feature {feat_idx}: correlation = {corr:.3f}\n")

            f.write("\n")

def generate_dashboard(
    sae: SAE,
    model: HookedTransformer,
    test_smiles: List[str],
    tokenizer: SmilesTokenizer,
    hook_point: str,
    output_dir: str
    ) -> None:
    """
    Generate dashboard using same test molecules.
    """
    tokens_list = []
    for s in test_smiles:
        token_ids = tokenizer.encode(s)
        # Pad to consistent length
        padded = token_ids + [tokenizer.pad_token_id] * (96 - len(token_ids))
        tokens_list.append(torch.tensor(padded[:96]))
        tokens = torch.stack(tokens_list).to(model.cfg.device)

        config = SaeVisConfig(
            hook_point=hook_point,
            features=list(range(min(10, sae.cfg.d_sae))),
            minibatch_size_features=16,
            minibatch_size_tokens=64,
            device=str(model.cfg.device),
            verbose=False,
        )

        runner = SaeVisRunner(config)
        vis_data = runner.run(encoder=sae, model=model, tokens=tokens)

        save_feature_centric_vis(
            sae_vis_data=vis_data, 
            filename=os.path.join(output_dir, "dashboard.html")
        )


# ============================================================================
# Main Pipeline with Train/Val/Test Split
# ============================================================================
def main():
    """Main training and analysis pipeline with proper splits."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = "sae_analysis"
    os.makedirs(output_dir, exist_ok=True)
    # Load data
    smiles_df = pd.read_csv("qm9.csv")
    all_smiles = smiles_df["smile"].values.tolist()

    # Proper train/val/test split
    n_total = len(all_smiles)
    n_train = int(0.7 * n_total)
    n_val = int(0.15 * n_total)

    train_smiles = all_smiles[:n_train]
    val_smiles = all_smiles[n_train:n_train + n_val]
    test_smiles = all_smiles[n_train + n_val:n_train + n_val + 100]  # 100 test molecules

    print(f"Dataset split: {n_train} train, {n_val} val, {len(test_smiles)} test")

    # Create tokenizer from training data only
    tokenizer = SmilesTokenizer.from_data(train_smiles)

    # Create datasets
    train_dataset = SmilesDataset(train_smiles, tokenizer, max_len=96)
    val_dataset = SmilesDataset(val_smiles, tokenizer, max_len=96)
    test_dataset = SmilesDataset(test_smiles, tokenizer, max_len=96)

    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False)

    # Train GPT model
    config = GPT2Config(
        n_layer=4, n_head=8, n_embd=32, vocab_size=tokenizer.vocab_size,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        loss_type="ForCausalLMLoss",
    )

    print("\nTraining GPT model...")
    model = train_gpt(
        GPT2LMHeadModel(config),
        train_dataset,
        epochs=20,
        lr=5e-4,
        device=device,
        pad_token_id=tokenizer.pad_token_id
    )

    hooked_model = convert_to_hooked(model, tokenizer)
    hook_point = "blocks.0.hook_resid_post"

    # Collect activations with context preservation
    print("\nCollecting activations with context...")
    train_context = collect_activations_with_context(
        hooked_model, train_loader, hook_point, tokenizer, train_smiles
    )

    # Train SAE with proper dead feature tracking
    print("\nTraining SAE...")
    sae, losses = train_sae(
        train_context,
        d_in=config.n_embd,
        d_sae=config.n_embd * 4,
        l1_coeff=1e-3,
        lr=3e-4,
        steps=80000,
        device=device
    )

    dead_features_mask = losses['dead_features_mask']
    print(f"Dead features: {dead_features_mask.sum().item()} / {len(dead_features_mask)}")

    # Analyze on held-out test set
    print("\nAnalyzing test set features...")
    test_smiles_subset = ["CCO", "CC(C)O", "c1ccccc1", "CC(=O)O", "CCN", "CCCC", "CC(C)C"]
    results = analyze_features_with_context(
        hooked_model, sae, tokenizer, test_smiles_subset, hook_point, dead_features_mask
    )

    # Interpret features with max-activating examples
    print("\nInterpreting SAE features...")
    interpretations = interpret_features(
        sae, hooked_model, train_context, hook_point, 
        top_k_examples=20, top_k_features=50
    )

    # Save results
    pd.DataFrame([{
        "smiles": r.smiles,
        "l0_mean": r.l0_mean,
        "l0_std": r.l0_std,
        "l1": r.l1,
        "sparsity_gini": r.sparsity_gini,
        "recon_error": r.recon_error,
        "explained_variance": r.explained_variance,
    } for r in results]).to_csv(f"{output_dir}/summary.csv", index=False)

    # Save feature interpretation summary
    pd.DataFrame([{
        "feature_idx": interp.feature_idx,
        "activation_frequency": interp.activation_frequency,
        "mean_activation": interp.mean_activation,
        "is_dead": interp.is_dead,
        "top_token": interp.max_activating_examples[0]['token'] if interp.max_activating_examples else '',
        "top_smiles": interp.max_activating_examples[0]['smiles'] if interp.max_activating_examples else '',
    } for interp in interpretations]).to_csv(f"{output_dir}/feature_interpretations.csv", index=False)

    # Generate all plots with proper context
    print("\nGenerating visualizations...")
    create_plots(results, losses, interpretations, output_dir)

    # Dashboard with consistent test set
    print("\nGenerating interactive dashboard...")
    generate_dashboard(sae, hooked_model, test_smiles_subset, tokenizer, hook_point, output_dir)

    print(f"\n✅ Analysis complete. Results in: {os.path.abspath(output_dir)}")


if __name__ == "__main__":
    main()