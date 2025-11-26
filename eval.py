import os
import argparse
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from transformers import GPT2Config, GPT2LMHeadModel
from transformer_lens import HookedTransformer, HookedTransformerConfig
from sae_lens import SAE
from sae_dashboard.sae_vis_data import SaeVisConfig
from sae_dashboard.sae_vis_runner import SaeVisRunner
from sae_dashboard.data_writing_fns import save_feature_centric_vis
from data import SmilesTokenizer

os.environ["TOKENIZERS_PARALLELISM"] = "false"
sns.set_style("whitegrid")


@dataclass
class ActivationContext:
    """Store activations with their context."""
    activations: torch.Tensor
    molecule_ids: List[int]
    positions: List[int]
    tokens: List[str]
    smiles: List[str]


@dataclass
class FeatureAnalysis:
    """SAE feature analysis results."""
    smiles: str
    tokens: List[str]
    features: torch.Tensor  # (seq_len, d_sae)
    recon_error: float
    explained_variance: float
    l0_mean: float
    l0_std: float
    l1: float
    sparsity_gini: float
    position_sparsity: List[float]
    max_activating_features: List[Dict]
    neuron_recon_error: Optional[float] = None


@dataclass
class FeatureInterpretation:
    """Interpretation of a single SAE feature."""
    feature_idx: int
    max_activating_examples: List[Dict]
    activation_frequency: float
    mean_activation: float
    is_dead: bool
    co_activating_features: List[Tuple[int, float]]


def load_activation_context(path: str) -> ActivationContext:
    """Load ActivationContext from disk."""
    data = torch.load(path)
    return ActivationContext(
        activations=data['activations'],
        molecule_ids=data['molecule_ids'],
        positions=data['positions'],
        tokens=data['tokens'],
        smiles=data['smiles']
    )


def convert_to_hooked(gpt2_model: GPT2LMHeadModel, tokenizer: SmilesTokenizer) -> HookedTransformer:
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


def compute_gini_coefficient(activations: torch.Tensor) -> float:
    """Gini coefficient measures inequality in activation distribution."""
    abs_acts = activations.abs().flatten()
    abs_acts = abs_acts[abs_acts > 1e-10]
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
    """Feature analysis with proper context."""
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

            # Per-molecule metrics
            recon_error = (reconstruction - hidden).pow(2).mean().item()
            explained_variance = 1 - (reconstruction - hidden).var().item() / hidden.var().item()

            # Sparsity metrics
            active_mask = features.abs() > 1e-5
            l0_per_token = active_mask.float().sum(dim=-1)
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
                neuron_recon_error=None,
            ))

    return results


def interpret_features(
    sae: SAE,
    activation_context: ActivationContext,
    device: str,
    top_k_examples: int = 20,
    top_k_features: int = 50
) -> List[FeatureInterpretation]:
    """Interpret SAE features by finding max-activating examples."""
    sae.eval()

    print("Computing feature activations for entire dataset...")

    # Get all feature activations
    with torch.no_grad():
        all_feature_acts = sae.encode(activation_context.activations.to(device))

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

        # Co-activation analysis
        co_activating = []
        if not is_dead:
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


def create_plots(
    results: List[FeatureAnalysis],
    losses: dict,
    interpretations: List[FeatureInterpretation],
    output_dir: str,
    split_name: str
) -> None:
    """Generate comprehensive analysis plots."""
    plot_dir = os.path.join(output_dir, f"{split_name}_plots")
    os.makedirs(plot_dir, exist_ok=True)
    
    smiles = [r.smiles for r in results]
    n = len(results)

    # 1. Training Curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(losses['train_losses']['total'], linewidth=2, color='steelblue', label='Train')
    ax1.plot(losses['val_losses']['total'], linewidth=2, color='coral', label='Val')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Total Loss')
    ax1.set_title('SAE Training Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(losses['train_losses']['recon'], label='Train Recon', linewidth=2, color='steelblue')
    ax2.plot(losses['val_losses']['recon'], label='Val Recon', linewidth=2, color='coral')
    ax2.plot(losses['train_losses']['l1'], label='Train L1', linewidth=2, color='mediumseagreen', linestyle='--')
    ax2.plot(losses['val_losses']['l1'], label='Val L1', linewidth=2, color='orange', linestyle='--')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.set_title('Loss Components')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/training_curves.png", dpi=150, bbox_inches='tight')
    plt.close()

    # 2. Feature Heatmaps
    fig, axes = plt.subplots(1, min(n, 5), figsize=(5*min(n, 5), 15))
    if n == 1:
        axes = [axes]
    elif n > 5:
        axes = axes[:5]
        results_subset = results[:5]
    else:
        results_subset = results

    for ax, r in zip(axes, results_subset):
        im = ax.imshow(r.features.numpy().T, aspect='auto', cmap='viridis')
        ax.set_xlabel('Token Position', fontsize=10)
        ax.set_ylabel('SAE Feature', fontsize=10)
        ax.set_title(f"{r.smiles}\nL0={r.l0_mean:.1f}±{r.l0_std:.1f} | Gini={r.sparsity_gini:.3f}", fontsize=11)
        ax.set_xticks(range(len(r.tokens)))
        ax.set_xticklabels(r.tokens, rotation=45, ha='right', fontsize=8)
        cbar = plt.colorbar(im, ax=ax, fraction=0.046)
        cbar.set_label('Activation', fontsize=9)

    plt.tight_layout()
    plt.savefig(f"{plot_dir}/feature_heatmaps.png", dpi=150, bbox_inches='tight')
    plt.close()

    # 3. Sparsity Metrics
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    l0_means = [r.l0_mean for r in results]
    l0_stds = [r.l0_std for r in results]
    x_pos = range(min(n, 20))
    axes[0, 0].bar(x_pos, l0_means[:20], yerr=l0_stds[:20], capsize=5, color='steelblue', edgecolor='black')
    axes[0, 0].set_xticks(x_pos)
    axes[0, 0].set_xticklabels([s[:10] for s in smiles[:20]], rotation=45, ha='right', fontsize=8)
    axes[0, 0].set_ylabel('L0 (Active Features per Token)')
    axes[0, 0].set_title('Sparsity: Mean ± Std across Positions')
    axes[0, 0].grid(True, alpha=0.3, axis='y')

    axes[0, 1].bar(x_pos, [r.sparsity_gini for r in results[:20]], color='mediumseagreen', edgecolor='black')
    axes[0, 1].set_xticks(x_pos)
    axes[0, 1].set_xticklabels([s[:10] for s in smiles[:20]], rotation=45, ha='right', fontsize=8)
    axes[0, 1].set_ylabel('Gini Coefficient')
    axes[0, 1].set_title('Activation Inequality (higher = sparser)')
    axes[0, 1].axhline(0.5, color='red', linestyle='--', alpha=0.5, label='Moderate')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3, axis='y')

    axes[1, 0].bar(x_pos, [r.recon_error for r in results[:20]], color='coral', edgecolor='black')
    axes[1, 0].set_xticks(x_pos)
    axes[1, 0].set_xticklabels([s[:10] for s in smiles[:20]], rotation=45, ha='right', fontsize=8)
    axes[1, 0].set_ylabel('MSE')
    axes[1, 0].set_title('Reconstruction Error')
    axes[1, 0].grid(True, alpha=0.3, axis='y')

    axes[1, 1].bar(x_pos, [r.explained_variance for r in results[:20]], color='mediumpurple', edgecolor='black')
    axes[1, 1].set_xticks(x_pos)
    axes[1, 1].set_xticklabels([s[:10] for s in smiles[:20]], rotation=45, ha='right', fontsize=8)
    axes[1, 1].set_ylabel('R²')
    axes[1, 1].set_title('Explained Variance')
    axes[1, 1].axhline(0.9, color='red', linestyle='--', alpha=0.5)
    axes[1, 1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(f"{plot_dir}/enhanced_metrics.png", dpi=150, bbox_inches='tight')
    plt.close()

    # 4. Position-wise Sparsity
    n_plot = min(n, 5)
    fig, axes = plt.subplots(n_plot, 1, figsize=(12, 4*n_plot))
    if n_plot == 1:
        axes = [axes]

    for ax, r in zip(axes, results[:n_plot]):
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
    plt.savefig(f"{plot_dir}/position_sparsity.png", dpi=150, bbox_inches='tight')
    plt.close()

    # 5. Top Features With Context
    n_plot = min(n, 5)
    fig, axes = plt.subplots(n_plot, 1, figsize=(12, 4*n_plot))
    if n_plot == 1:
        axes = [axes]

    for ax, r in zip(axes, results[:n_plot]):
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
    plt.savefig(f"{plot_dir}/top_features_with_context.png", dpi=150, bbox_inches='tight')
    plt.close()

    # 6. Feature Interpretations
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    act_freqs = [interp.activation_frequency for interp in interpretations]
    axes[0, 0].hist(act_freqs, bins=30, color='steelblue', edgecolor='black', alpha=0.7)
    axes[0, 0].set_xlabel('Activation Frequency')
    axes[0, 0].set_ylabel('Number of Features')
    axes[0, 0].set_title('Feature Activation Frequency Distribution')
    axes[0, 0].axvline(np.mean(act_freqs), color='red', linestyle='--', label=f'Mean: {np.mean(act_freqs):.3f}')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    dead_count = sum(1 for interp in interpretations if interp.is_dead)
    alive_count = len(interpretations) - dead_count
    axes[0, 1].bar(['Alive', 'Dead'], [alive_count, dead_count], color=['green', 'red'], edgecolor='black')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title(f'Feature Status (Dead = {dead_count}/{len(interpretations)})')
    axes[0, 1].grid(True, alpha=0.3, axis='y')

    mean_acts = [interp.mean_activation for interp in interpretations if not interp.is_dead]
    if mean_acts:
        axes[1, 0].hist(mean_acts, bins=30, color='coral', edgecolor='black', alpha=0.7)
    axes[1, 0].set_xlabel('Mean Activation (when active)')
    axes[1, 0].set_ylabel('Number of Features')
    axes[1, 0].set_title('Feature Strength Distribution')
    axes[1, 0].grid(True, alpha=0.3)

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
    plt.savefig(f"{plot_dir}/feature_interpretations.png", dpi=150, bbox_inches='tight')
    plt.close()

    # 7. Feature Interpretation Report
    with open(f"{plot_dir}/feature_interpretations.txt", "w") as f:
        f.write("=" * 80 + "\n")
        f.write(f"SAE FEATURE INTERPRETATION REPORT - {split_name.upper()}\n")
        f.write("=" * 80 + "\n\n")

        for interp in interpretations[:20]:
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

    print(f"Saved plots to {plot_dir}")


def generate_dashboard(
    sae: SAE,
    model: HookedTransformer,
    test_smiles: List[str],
    tokenizer: SmilesTokenizer,
    hook_point: str,
    output_dir: str,
    max_len: int
) -> None:
    """Generate dashboard using test molecules."""
    tokens_list = []
    for s in test_smiles[:10]:  # Limit to 10 for dashboard
        token_ids = tokenizer.encode(s)
        padded = token_ids + [tokenizer.pad_token_id] * (max_len - len(token_ids))
        tokens_list.append(torch.tensor(padded[:max_len]))
    
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
    print(f"Saved dashboard to {os.path.join(output_dir, 'dashboard.html')}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate SAE on all data splits")
    parser.add_argument("--exp_dir", type=str, required=True, help="Experiment directory with checkpoints and activations")
    parser.add_argument("--test_smiles", type=str, nargs='+', default=None, 
                        help="Specific SMILES for detailed analysis (default: use first 7 from test set)")
    parser.add_argument("--top_k_features", type=int, default=50, help="Number of features to interpret")
    parser.add_argument("--top_k_examples", type=int, default=20, help="Number of max-activating examples per feature")
    parser.add_argument("--generate_dashboard", action="store_true", help="Generate interactive dashboard")
    
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load metadata
    print("\nLoading metadata...")
    metadata = torch.load(os.path.join(args.exp_dir, "metadata.pt"))
    hook_point = metadata['hook_point']
    d_model = metadata['d_model']
    n_layers = metadata['n_layers']
    n_heads = metadata['n_heads']
    max_len = metadata['max_len']
    vocab_size = metadata['vocab_size']
    
    print(f"Hook point: {hook_point}")
    print(f"Model dimension: {d_model}")
    
    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = torch.load(os.path.join(args.exp_dir, "tokenizer.pt"), weights_only=False)
    
    # Load GPT model
    print("\nLoading GPT model...")
    config = GPT2Config(
        n_layer=n_layers,
        n_head=n_heads,
        n_embd=d_model,
        vocab_size=vocab_size,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        n_positions=max_len,
    )
    gpt_model = GPT2LMHeadModel(config)
    gpt_model.load_state_dict(torch.load(os.path.join(args.exp_dir, "best_gpt_model.pt")))
    hooked_model = convert_to_hooked(gpt_model, tokenizer)
    
    # Load SAE
    print("\nLoading SAE...")
    sae_checkpoint = torch.load(os.path.join(args.exp_dir, "best_sae_model.pt"))
    d_sae = None
    for key in sae_checkpoint['model_state_dict'].keys():
        if 'W_enc' in key:
            d_sae = sae_checkpoint['model_state_dict'][key].shape[1]
            break
    
    sae = SAE.from_dict({
        "d_in": d_model,
        "d_sae": d_sae,
        "l1_coefficient": 1e-3,  # Not used during eval
        "architecture": "standard",
    })
    sae.load_state_dict(sae_checkpoint['model_state_dict'])
    sae = sae.to(device)
    
    dead_features_mask = sae_checkpoint['dead_features_mask']
    print(f"Dead features: {dead_features_mask.sum().item()} / {len(dead_features_mask)}")
    
    # Load training history
    history = torch.load(os.path.join(args.exp_dir, "sae_training_history.pt"))
    
    # Load activations for all splits
    print("\nLoading activations...")
    train_context = load_activation_context(os.path.join(args.exp_dir, "train_activations.pt"))
    val_context = load_activation_context(os.path.join(args.exp_dir, "val_activations.pt"))
    test_context = load_activation_context(os.path.join(args.exp_dir, "test_activations.pt"))
    
    print(f"Train activations: {train_context.activations.shape}")
    print(f"Val activations: {val_context.activations.shape}")
    print(f"Test activations: {test_context.activations.shape}")
    
    # Get unique test SMILES for detailed analysis
    unique_test_smiles = []
    seen = set()
    for s in test_context.smiles:
        if s not in seen and len(unique_test_smiles) < 100:
            seen.add(s)
            unique_test_smiles.append(s)
    
    if args.test_smiles is not None:
        analysis_smiles = args.test_smiles
    else:
        analysis_smiles = unique_test_smiles[:7]
    
    print(f"\nAnalyzing {len(analysis_smiles)} molecules in detail")
    
    # Evaluate on all splits
    all_results = {}
    all_interpretations = {}
    
    for split_name, context in [('train', train_context), ('val', val_context), ('test', test_context)]:
        print(f"\n{'='*60}")
        print(f"Evaluating {split_name.upper()} split")
        print(f"{'='*60}")
        
        # Detailed feature analysis on subset
        print(f"\nAnalyzing features for {len(analysis_smiles)} molecules...")
        results = analyze_features_with_context(
            hooked_model, sae, tokenizer, analysis_smiles, hook_point, dead_features_mask
        )
        all_results[split_name] = results
        
        # Interpret features on full split
        print(f"\nInterpreting features on full {split_name} split...")
        interpretations = interpret_features(
            sae, context, device, 
            top_k_examples=args.top_k_examples, 
            top_k_features=args.top_k_features
        )
        all_interpretations[split_name] = interpretations
        
        # Save summary statistics
        summary_df = pd.DataFrame([{
            "smiles": r.smiles,
            "l0_mean": r.l0_mean,
            "l0_std": r.l0_std,
            "l1": r.l1,
            "sparsity_gini": r.sparsity_gini,
            "recon_error": r.recon_error,
            "explained_variance": r.explained_variance,
        } for r in results])
        summary_df.to_csv(os.path.join(args.exp_dir, f"{split_name}_summary.csv"), index=False)
        
        # Save feature interpretations
        interp_df = pd.DataFrame([{
            "feature_idx": interp.feature_idx,
            "activation_frequency": interp.activation_frequency,
            "mean_activation": interp.mean_activation,
            "is_dead": interp.is_dead,
            "top_token": interp.max_activating_examples[0]['token'] if interp.max_activating_examples else '',
            "top_smiles": interp.max_activating_examples[0]['smiles'] if interp.max_activating_examples else '',
        } for interp in interpretations])
        interp_df.to_csv(os.path.join(args.exp_dir, f"{split_name}_feature_interpretations.csv"), index=False)
        
        # Generate plots
        print(f"\nGenerating visualizations for {split_name}...")
        create_plots(results, history, interpretations, args.exp_dir, split_name)
        
        # Print summary statistics
        print(f"\n{split_name.upper()} Summary:")
        print(f"  Mean L0: {summary_df['l0_mean'].mean():.2f} ± {summary_df['l0_mean'].std():.2f}")
        print(f"  Mean Gini: {summary_df['sparsity_gini'].mean():.3f} ± {summary_df['sparsity_gini'].std():.3f}")
        print(f"  Mean Recon Error: {summary_df['recon_error'].mean():.4f} ± {summary_df['recon_error'].std():.4f}")
        print(f"  Mean Explained Variance: {summary_df['explained_variance'].mean():.3f} ± {summary_df['explained_variance'].std():.3f}")
        print(f"  Dead features: {sum(1 for i in interpretations if i.is_dead)} / {len(interpretations)}")
    
    # Generate dashboard on test set
    if args.generate_dashboard:
        print("\n" + "="*60)
        print("Generating interactive dashboard...")
        print("="*60)
        generate_dashboard(
            sae, hooked_model, analysis_smiles, tokenizer, 
            hook_point, args.exp_dir, max_len
        )
    
    print(f"\n✅ Evaluation complete. Results saved to: {os.path.abspath(args.exp_dir)}")


if __name__ == "__main__":
    main()