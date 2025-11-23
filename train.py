import os
import torch
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Config, GPT2LMHeadModel
import re
from typing import List
from sae_lens import SAE
from transformer_lens import HookedTransformer
import matplotlib.pyplot as plt

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# -----------------------------
# 1) SMILES Tokenizer
# -----------------------------
RE_PATTERN = re.compile(
    r"(\[[^\]]+]|<|Br?|Cl?|N|O|S|P|F|I|"
    r"b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|/|:|~|@|\?|>"
    r"|\*|\$|%[0-9]{2}|[0-9])"
)

BOS, EOS, PAD, UNK = "<bos>", "<eos>", "<pad>", "<unk>"

def build_vocabulary(data: List[str]) -> List[str]:
    tokens = set()
    for smiles in data:
        tokens.update(RE_PATTERN.findall(smiles.strip()))
    return sorted(tokens)


class SmilesTokenizer:
    def __init__(self, vocabulary: List[str]):
        all_tokens = vocabulary + [BOS, EOS, PAD, UNK]
        self.c2i = {tok: i for i, tok in enumerate(all_tokens)}
        self.i2c = {i: tok for tok, i in self.c2i.items()}
        self.vocab_size = len(self.c2i)
        self.bos_token_id = self.c2i[BOS]
        self.eos_token_id = self.c2i[EOS]
        self.pad_token_id = self.c2i[PAD]
        self.unk_token_id = self.c2i[UNK]

    @classmethod
    def from_data(cls, smiles_list: List[str]):
        return cls(build_vocabulary(smiles_list))

    def encode(self, string: str, add_bos=True, add_eos=True):
        tokens = RE_PATTERN.findall(string.strip())
        ids = [self.c2i.get(tok, self.unk_token_id) for tok in tokens]
        if add_bos:
            ids.insert(0, self.bos_token_id)
        if add_eos:
            ids.append(self.eos_token_id)
        return ids

    def decode(self, ids, skip_special_tokens=True):
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        toks = []
        for i in ids:
            tok = self.i2c.get(i, UNK)
            if skip_special_tokens and tok in {BOS, EOS, PAD}:
                continue
            toks.append(tok)
        return "".join(toks)


class SmilesDataset(Dataset):
    def __init__(self, smiles, tokenizer, max_len=100):
        self.smiles = smiles
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        ids = self.tok.encode(self.smiles[idx])
        if len(ids) > self.max_len:
            ids = ids[:self.max_len]
        else:
            ids = ids + [self.tok.pad_token_id] * (self.max_len - len(ids))
        return torch.tensor(ids)


# -----------------------------
# 2) Train GPT Model
# -----------------------------
def train_gpt(model, dataloader, epochs=1, lr=5e-4, device="cuda"):
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    
    for epoch in range(epochs):
        loop = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for x in loop:
            x = x.to(device)
            outputs = model(input_ids=x, labels=x)
            loss = outputs.loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loop.set_postfix(loss=loss.item())
    
    return model


# -----------------------------
# 3) Collect Activations
# -----------------------------
def collect_activations(model, dataloader, layer_idx=0, max_batches=50, device="cuda"):
    """Collect hidden state activations from a specific layer."""
    model.eval()
    activations = []
    
    with torch.no_grad():
        for i, x in enumerate(tqdm(dataloader, desc="Collecting activations")):
            if i >= max_batches:
                break
            
            x = x.to(device)
            outputs = model(input_ids=x, output_hidden_states=True)
            hidden = outputs.hidden_states[layer_idx + 1]  # +1 because index 0 is embeddings
            hidden_flat = hidden.reshape(-1, hidden.shape[-1])
            activations.append(hidden_flat.cpu())
    
    all_activations = torch.cat(activations, dim=0)
    print(f"Collected {all_activations.shape[0]} activation vectors")
    return all_activations


# -----------------------------
# 4) Train SAE - Correct sae_lens imports
# -----------------------------


def train_sae_simple(activations, d_in, d_sae=None, l1_coeff=1e-3, 
                     lr=3e-4, steps=3000, batch_size=1024, device="cuda"):
    """Train SAE using sae_lens SAE class."""
    if d_sae is None:
        d_sae = d_in * 4
    
    # Initialize SAE using from_dict (correct API)
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
    
    for step in tqdm(range(steps), desc="Training SAE"):
        indices = torch.randint(0, activations.shape[0], (batch_size,))
        batch = activations[indices].to(device)
        
        # Use encode/decode methods
        feature_acts = sae.encode(batch)
        reconstruction = sae.decode(feature_acts)
        
        # Loss calculation
        recon_loss = (reconstruction - batch).pow(2).mean()
        l1_loss = l1_coeff * feature_acts.abs().sum(dim=-1).mean()
        loss = recon_loss + l1_loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        if step % 500 == 0:
            print(f"Step {step}: Loss={loss.item():.4f}, Recon={recon_loss.item():.4f}, L1={l1_loss.item():.4f}")
    
    return sae, losses


# -----------------------------
# 5) Wrap model for sae_dashboard compatibility
# -----------------------------


def wrap_gpt2_for_dashboard(gpt2_model, tokenizer):
    """
    Wrap HuggingFace GPT2 model to work with transformer_lens.
    This allows compatibility with sae_dashboard.
    """
    # Create HookedTransformer from pretrained
    hooked_model = HookedTransformer.from_pretrained(
        "gpt2-small",
        device=str(gpt2_model.device)
    )
    
    # Copy weights from our trained model
    hooked_model.load_state_dict(gpt2_model.state_dict(), strict=False)
    
    return hooked_model


# -----------------------------
# 6) Create tokens dataset for dashboard
# -----------------------------
def create_token_dataset(smiles_list, tokenizer, max_samples=100):
    """Create a token dataset for sae_dashboard."""
    tokens_list = []
    
    for smiles in smiles_list[:max_samples]:
        token_ids = tokenizer.encode(smiles)
        tokens_list.append(torch.tensor(token_ids))
    
    # Pad to same length
    max_len = max(len(t) for t in tokens_list)
    padded_tokens = []
    for t in tokens_list:
        if len(t) < max_len:
            padded = torch.cat([t, torch.full((max_len - len(t),), tokenizer.pad_token_id)])
        else:
            padded = t
        padded_tokens.append(padded)
    
    return torch.stack(padded_tokens)


# -----------------------------
# 7) Generate SAE Dashboard Visualization
# -----------------------------
def generate_sae_dashboard(sae, model, tokens, output_dir="sae_analysis"):
    """
    Generate interactive SAE dashboard using sae_dashboard.
    """
    try:
        from sae_dashboard.sae_vis_data import SaeVisConfig
        from sae_dashboard.sae_vis_runner import SaeVisRunner
        from sae_dashboard.data_writing_fns import save_feature_centric_vis
        
        os.makedirs(output_dir, exist_ok=True)
        
        print("\n=== Generating SAE Dashboard ===")
        
        # Configure visualization
        config = SaeVisConfig(
            hook_point=sae.cfg.hook_name if hasattr(sae.cfg, 'hook_name') else "blocks.0.hook_resid_post",
            features=list(range(min(50, sae.cfg.d_sae))),  # Visualize first 50 features
            minibatch_size_features=32,
            minibatch_size_tokens=128,
            device=str(model.device),
            verbose=True,
        )
        
        # Generate visualization data
        print("Running SaeVisRunner...")
        runner = SaeVisRunner(config)
        vis_data = runner.run(
            encoder=sae,
            model=model,
            tokens=tokens
        )
        
        # Save interactive HTML dashboard
        dashboard_path = os.path.join(output_dir, "feature_dashboard.html")
        save_feature_centric_vis(
            sae_vis_data=vis_data,
            filename=dashboard_path
        )
        
        print(f"\n✓ Interactive dashboard saved to: {dashboard_path}")
        print(f"  Open this file in a web browser to explore features!")
        
        return vis_data
        
    except ImportError as e:
        print(f"\n⚠ sae_dashboard not installed: {e}")
        print("Install with: pip install sae-dashboard")
        print("Falling back to matplotlib visualizations...")
        return None


# -----------------------------
# 8) Analysis
# -----------------------------
def analyze_sae_features(model, sae, tokenizer, test_smiles_list, device="cuda"):
    """Analyze SAE features on test SMILES."""
    model.eval()
    sae.eval()
    results = []
    
    with torch.no_grad():
        for smiles in test_smiles_list:
            token_ids = tokenizer.encode(smiles)
            input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
            
            outputs = model(input_ids, output_hidden_states=True)
            hidden = outputs.hidden_states[1]  # Layer 0
            
            hidden_flat = hidden.reshape(-1, hidden.shape[-1])
            feature_acts = sae.encode(hidden_flat)
            reconstruction = sae.decode(feature_acts)
            
            recon_error = (hidden_flat - reconstruction).pow(2).mean().item()
            l0 = (feature_acts.abs() > 1e-5).float().sum(dim=-1).mean().item()
            
            results.append({
                "smiles": smiles,
                "features": feature_acts.cpu(),
                "recon_error": recon_error,
                "tokens": [tokenizer.i2c[i] for i in token_ids],
                "l0": l0,
            })
    
    return results


# -----------------------------
# 9) Fallback Matplotlib Visualizations
# -----------------------------


def create_matplotlib_visualizations(results, losses, output_dir="sae_analysis"):
    """Create matplotlib visualizations as fallback."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Training loss
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.xlabel('Training Step')
    plt.ylabel('Loss')
    plt.title('SAE Training Loss')
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{output_dir}/training_loss.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Feature heatmaps
    n_plots = len(results)
    fig, axes = plt.subplots(n_plots, 1, figsize=(12, 3*n_plots))
    if n_plots == 1:
        axes = [axes]
    
    for ax, result in zip(axes, results):
        features = result["features"].numpy()
        im = ax.imshow(features.T, aspect='auto', cmap='viridis', interpolation='nearest')
        ax.set_xlabel('Token Position')
        ax.set_ylabel('SAE Feature')
        ax.set_title(f"{result['smiles']} | L0={result['l0']:.1f} | Error={result['recon_error']:.4f}")
        plt.colorbar(im, ax=ax, fraction=0.046)
        ax.set_xticks(range(len(result['tokens'])))
        ax.set_xticklabels(result['tokens'], rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/feature_heatmap.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Metrics
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    smiles_labels = [r["smiles"] for r in results]
    l0_values = [r["l0"] for r in results]
    errors = [r["recon_error"] for r in results]
    
    ax1.bar(range(len(l0_values)), l0_values, color='steelblue', edgecolor='black')
    ax1.set_xticks(range(len(smiles_labels)))
    ax1.set_xticklabels(smiles_labels, rotation=45, ha='right')
    ax1.set_ylabel('L0 (Active Features)')
    ax1.set_title('SAE Sparsity per Molecule')
    ax1.grid(True, alpha=0.3, axis='y')
    
    ax2.bar(range(len(errors)), errors, color='coral', edgecolor='black')
    ax2.set_xticks(range(len(smiles_labels)))
    ax2.set_xticklabels(smiles_labels, rotation=45, ha='right')
    ax2.set_ylabel('Reconstruction Error (MSE)')
    ax2.set_title('SAE Reconstruction Quality')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/metrics.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: {output_dir}/training_loss.png")
    print(f"✓ Saved: {output_dir}/feature_heatmap.png")
    print(f"✓ Saved: {output_dir}/metrics.png")


# -----------------------------
# Main
# -----------------------------
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}\n")
    
    # Load data
    print("=== Loading SMILES Data ===")
    smiles = pd.read_csv("qm9.csv", sep=",", dtype=str)["smile"].values.tolist()[:10000]
    tokenizer = SmilesTokenizer.from_data(smiles)
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    
    # Prepare dataset
    dataset = SmilesDataset(smiles, tokenizer, max_len=96)
    dataloader = DataLoader(dataset, batch_size=512, shuffle=True)
    
    # Build and train GPT
    print("\n=== Training GPT Model ===")
    config = GPT2Config(
        n_layer=4,
        n_head=4,
        n_embd=64,
        vocab_size=tokenizer.vocab_size,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    model = GPT2LMHeadModel(config)
    model = train_gpt(model, dataloader, epochs=2, device=device)
    
    # Collect activations
    print("\n=== Collecting Activations ===")
    activations = collect_activations(model, dataloader, layer_idx=0, max_batches=50, device=device)
    
    # Train SAE
    print("\n=== Training SAE ===")
    sae, losses = train_sae_simple(
        activations,
        d_in=config.n_embd,
        d_sae=config.n_embd * 4,
        l1_coeff=1e-3,
        lr=3e-4,
        steps=3000,
        device=device
    )
    
    # Analyze features
    print("\n=== Analyzing SAE Features ===")
    test_smiles = ["CCO", "CC(C)O", "c1ccccc1", "CC(=O)O", "CCN"]
    results = analyze_sae_features(model, sae, tokenizer, test_smiles, device)
    
    # Print summary
    print("\n=== Analysis Summary ===")
    for r in results:
        print(f"{r['smiles']:15s} | L0: {r['l0']:6.2f} | Recon Error: {r['recon_error']:.4f}")
    
    # Try to generate interactive dashboard
    print("\n=== Attempting Dashboard Generation ===")
    try:
        # Wrap model for transformer_lens compatibility
        hooked_model = wrap_gpt2_for_dashboard(model, tokenizer)
        
        # Create token dataset
        tokens = create_token_dataset(smiles, tokenizer, max_samples=100)
        tokens = tokens.to(device)
        
        # Generate dashboard
        vis_data = generate_sae_dashboard(sae, hooked_model, tokens)
        
    except Exception as e:
        print(f"Dashboard generation failed: {e}")
        print("Using matplotlib fallback...")
    
    # Always create matplotlib visualizations
    print("\n=== Creating Matplotlib Visualizations ===")
    create_matplotlib_visualizations(results, losses)
    
    print("\n=== Complete! ===")
    print("Results saved in 'sae_analysis/' directory")
    print("\nGenerated files:")
    print("  - feature_dashboard.html (if sae_dashboard available)")
    print("  - training_loss.png")
    print("  - feature_heatmap.png")
    print("  - metrics.png")


if __name__ == "__main__":
    main()