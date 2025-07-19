import random
from typing import Any, Dict, List

import numpy as np
import torch


def set_seed(seed_val: int = 42) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)


def build_prompt_response_pairs(dataset: Any) -> List[Dict[str, str]]:
    """Build prompt-response pairs from OASST dataset."""
    id_to_msg = {msg["message_id"]: msg for msg in dataset}
    pairs = []
    for msg in dataset:
        if msg["role"] == "assistant":
            parent_id = msg.get("parent_id")
            if not parent_id:
                continue
            parent = id_to_msg.get(parent_id)
            if parent and parent["role"] == "prompter":
                prompt = parent["text"]
                response = msg["text"]
                pairs.append({"prompt": prompt, "response": response})
    return pairs
