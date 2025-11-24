import re
import torch

from torch.utils.data import Dataset
from typing import List, Dict


# -----------------------------
# Constants
# -----------------------------
RE_PATTERN = re.compile(
    r"(\[[^\]]+]|<|Br?|Cl?|N|O|S|P|F|I|"
    r"b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|/|:|~|@|\?|>"
    r"|\*|\$|%[0-9]{2}|[0-9])"
)

BOS = "<bos>"
EOS = "<eos>"
PAD = "<pad>"
UNK = "<unk>"


# -----------------------------
# SMILES Tokenizer (HuggingFace Compatible)
# -----------------------------
def build_vocabulary(data: List[str]) -> List[str]:
    """Extract unique SMILES tokens from dataset."""
    tokens = set()
    for smiles in data:
        tokens.update(RE_PATTERN.findall(smiles.strip()))
    return sorted(tokens)


class SmilesTokenizer:
    """HuggingFace-compatible tokenizer for SMILES strings."""
    
    def __init__(self, vocabulary: List[str]):
        all_tokens = vocabulary + [BOS, EOS, PAD, UNK]
        self.c2i = {tok: i for i, tok in enumerate(all_tokens)}
        self.i2c = {i: tok for tok, i in self.c2i.items()}
        
        # Standard HuggingFace attributes
        self.vocab = self.c2i
        self.vocab_size = len(self.c2i)
        self.pad_token_id = self.c2i[PAD]
        self.bos_token_id = self.c2i[BOS]
        self.eos_token_id = self.c2i[EOS]
        self.unk_token_id = self.c2i[UNK]
        self.pad_token = PAD
        self.bos_token = BOS
        self.eos_token = EOS
        self.unk_token = UNK
        self.padding_side = "right"
        self.model_max_length = 1024

    @classmethod
    def from_data(cls, smiles_list: List[str]) -> "SmilesTokenizer":
        """Build tokenizer from SMILES dataset."""
        vocab = build_vocabulary(smiles_list)
        return cls(vocab)

    def encode(self, text: str, add_bos: bool = True, add_eos: bool = True) -> List[int]:
        """Encode SMILES string to token IDs."""
        tokens = RE_PATTERN.findall(text.strip())
        ids = [self.c2i.get(tok, self.unk_token_id) for tok in tokens]
        if add_bos:
            ids.insert(0, self.bos_token_id)
        if add_eos:
            ids.append(self.eos_token_id)
        return ids

    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decode token IDs to SMILES string."""
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        toks = []
        for i in ids:
            tok = self.i2c.get(i, UNK)
            if skip_special_tokens and tok in {BOS, EOS, PAD}:
                continue
            toks.append(tok)
        return "".join(toks)

    def batch_decode(self, ids_list: List[List[int]], skip_special_tokens: bool = True) -> List[str]:
        """Batch decode token IDs."""
        return [self.decode(ids, skip_special_tokens=skip_special_tokens) for ids in ids_list]

    def convert_ids_to_tokens(self, token_ids: List[int]) -> List[str]:
        """Convert token IDs to token strings."""
        if isinstance(token_ids, int):
            return self.i2c.get(token_ids, UNK)
        return [self.i2c.get(i, UNK) for i in token_ids]

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        """Convert tokens to string."""
        return "".join(tokens)

    def __call__(self, text: str, **kwargs) -> Dict[str, List[int]]:
        """Make tokenizer callable."""
        return {"input_ids": self.encode(text)}


# -----------------------------
# Dataset
# -----------------------------
class SmilesDataset(Dataset):
    """PyTorch Dataset for SMILES strings."""
    
    def __init__(self, smiles: List[str], tokenizer: SmilesTokenizer, max_len: int = 100):
        self.smiles = smiles
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.smiles)

    def __getitem__(self, idx: int) -> torch.Tensor:
        ids = self.tokenizer.encode(self.smiles[idx])
        if len(ids) > self.max_len:
            ids = ids[:self.max_len]
        else:
            ids = ids + [self.tokenizer.pad_token_id] * (self.max_len - len(ids))
        return torch.tensor(ids, dtype=torch.long)


