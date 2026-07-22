from typing import Dict, List, Tuple

class Solution:
    def build_vocab(self, text: str) -> Tuple[Dict[str, int], Dict[int, str]]:
        # Return (stoi, itos) where:
        # - stoi maps each unique character to a unique integer (sorted alphabetically)
        # - itos is the reverse mapping (integer to character)
        sorted_unique = sorted(set(text))
        stoi = {}
        itos = {}
        for idx,char in enumerate(sorted_unique):
            stoi[char] = idx
            itos[idx] = char
        return (stoi,itos)

    def encode(self, text: str, stoi: Dict[str, int]) -> List[int]:
        # Convert a string to a list of integers using stoi mapping
        encoding = []
        for char in text:
            encoding.append(stoi[char])
        return encoding

    def decode(self, ids: List[int], itos: Dict[int, str]) -> str:
        # Convert a list of integers back to a string using itos mapping
        decoding = ""
        for pos in ids:
            decoding += itos[pos]
        return decoding
