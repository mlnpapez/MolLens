import os
import argparse
import torch
import pandas as pd

from tqdm import tqdm
from typing import List
from dataclasses import dataclass
from torch.utils.data import DataLoader
from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    Trainer,
    TrainingArguments
)
from transformer_lens import (
    HookedTransformer,
    HookedTransformerConfig
)
from data import SmilesTokenizer, SmilesDataset

os.environ["TOKENIZERS_PARALLELISM"] = "false"


@dataclass
class ActivationContext:
    """Store activations with their context."""
    activations: torch.Tensor  # (batch*seq, d_model)
    molecule_ids: List[int]    # Which molecule each activation came from
    positions: List[int]       # Which token position
    tokens: List[str]          # The actual token
    smiles: List[str]          # The full SMILES string


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


def collect_activations_with_context(
    model: HookedTransformer,
    dataloader: DataLoader,
    hook_point: str,
    tokenizer: SmilesTokenizer,
    smiles_list: List[str]
) -> ActivationContext:
    """Collect activations while preserving position and molecule context."""
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


def save_activation_context(context: ActivationContext, path: str):
    """Save ActivationContext to disk."""
    torch.save({
        'activations': context.activations,
        'molecule_ids': context.molecule_ids,
        'positions': context.positions,
        'tokens': context.tokens,
        'smiles': context.smiles
    }, path)
    print(f"Saved activations to {path}")


def main():
    parser = argparse.ArgumentParser(description="Train GPT model and collect activations")
    parser.add_argument("--data_path", type=str, required=True, help="Path to CSV file with SMILES")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for checkpoints and activations")
    parser.add_argument("--hook_point", type=str, default="blocks.0.hook_resid_post", help="Hook point for activation collection")
    
    # Data splits
    parser.add_argument("--train_ratio", type=float, default=0.7, help="Training data ratio")
    parser.add_argument("--val_ratio", type=float, default=0.15, help="Validation data ratio")
    parser.add_argument("--test_size", type=int, default=100, help="Number of test samples")
    
    # Model architecture
    parser.add_argument("--n_layers", type=int, default=4, help="Number of transformer layers")
    parser.add_argument("--n_heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--d_model", type=int, default=32, help="Model dimension")
    parser.add_argument("--max_len", type=int, default=96, help="Maximum sequence length")
    
    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--logging_steps", type=int, default=100, help="Logging frequency")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load data
    print(f"\nLoading data from {args.data_path}...")
    smiles_df = pd.read_csv(args.data_path)
    all_smiles = smiles_df["smile"].values.tolist()
    
    # Create splits
    n_total = len(all_smiles)
    n_train = int(args.train_ratio * n_total)
    n_val = int(args.val_ratio * n_total)
    
    train_smiles = all_smiles[:n_train]
    val_smiles = all_smiles[n_train:n_train + n_val]
    test_smiles = all_smiles[n_train + n_val:n_train + n_val + args.test_size]
    
    print(f"Dataset split: {len(train_smiles)} train, {len(val_smiles)} val, {len(test_smiles)} test")
    
    # Create tokenizer from training data only
    print("\nCreating tokenizer...")
    tokenizer = SmilesTokenizer.from_data(train_smiles)
    tokenizer_path = os.path.join(args.output_dir, "tokenizer.pt")
    torch.save(tokenizer, tokenizer_path)
    print(f"Saved tokenizer to {tokenizer_path}")
    
    # Create datasets
    train_dataset = SmilesDataset(train_smiles, tokenizer, max_len=args.max_len)
    val_dataset = SmilesDataset(val_smiles, tokenizer, max_len=args.max_len)
    
    # Initialize model
    print("\nInitializing GPT model...")
    config = GPT2Config(
        n_layer=args.n_layers,
        n_head=args.n_heads,
        n_embd=args.d_model,
        vocab_size=tokenizer.vocab_size,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        n_positions=args.max_len,
    )
    
    model = GPT2LMHeadModel(config).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Data collator
    def data_collator(examples):
        batch = torch.stack(examples)
        attention_mask = (batch != tokenizer.pad_token_id).long()
        return {"input_ids": batch, "labels": batch, "attention_mask": attention_mask}
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        logging_steps=args.logging_steps,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,  # Only keep best checkpoint
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        remove_unused_columns=False,
    )
    
    # Train model
    print("\nTraining GPT model...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )
    
    trainer.train()
    
    # Save best model
    best_model_path = os.path.join(args.output_dir, "best_gpt_model.pt")
    torch.save(model.state_dict(), best_model_path)
    print(f"\nSaved best model to {best_model_path}")
    
    # Convert to HookedTransformer
    print("\nConverting to HookedTransformer...")
    hooked_model = convert_to_hooked(model, tokenizer)
    
    # Collect activations for all splits
    print(f"\nCollecting activations at hook point: {args.hook_point}")
    
    # Train activations
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
    train_context = collect_activations_with_context(
        hooked_model, train_loader, args.hook_point, tokenizer, train_smiles
    )
    save_activation_context(train_context, os.path.join(args.output_dir, "train_activations.pt"))
    
    # Val activations
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    val_context = collect_activations_with_context(
        hooked_model, val_loader, args.hook_point, tokenizer, val_smiles
    )
    save_activation_context(val_context, os.path.join(args.output_dir, "val_activations.pt"))
    
    # Test activations
    test_dataset = SmilesDataset(test_smiles, tokenizer, max_len=args.max_len)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    test_context = collect_activations_with_context(
        hooked_model, test_loader, args.hook_point, tokenizer, test_smiles
    )
    save_activation_context(test_context, os.path.join(args.output_dir, "test_activations.pt"))
    
    # Save metadata
    metadata = {
        "hook_point": args.hook_point,
        "d_model": args.d_model,
        "n_layers": args.n_layers,
        "n_heads": args.n_heads,
        "max_len": args.max_len,
        "vocab_size": tokenizer.vocab_size,
        "train_samples": len(train_smiles),
        "val_samples": len(val_smiles),
        "test_samples": len(test_smiles),
    }
    metadata_path = os.path.join(args.output_dir, "metadata.pt")
    torch.save(metadata, metadata_path)
    print(f"\nSaved metadata to {metadata_path}")
    
    print(f"\n✅ Training complete. All artifacts saved to: {os.path.abspath(args.output_dir)}")


if __name__ == "__main__":
    main()
