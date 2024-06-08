"""
Source: 
https://github.com/rasbt/LLMs-from-scratch/
"""

import tiktoken
import torch
from torch.utils.data import Dataset


class GPTDataset_v1(Dataset):

    def __init__(self, txt, tokenizer, max_length, stride) -> None:
        super().__init__()
        self.input_ids = []
        self.target_ids = []

        # Tokenizer the entire text
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        # Use a sliding window to chunk the book into overlapping sequences of max_length
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i: i+max_length]
            target_chunk = token_ids[i+1: i+1+max_length]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx) -> Any:
        return self.input_ids[idx], self.target_ids[idx]
    