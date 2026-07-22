import torch
import torch.nn as nn
from torchtyping import TensorType
from typing import List

class Solution:
    def get_dataset(self, positive: List[str], negative: List[str]) -> TensorType[float]:
        # 1. Build vocabulary: collect all unique words, sort them, assign integer IDs starting at 1
        # 2. Encode each sentence by replacing words with their IDs
        # 3. Combine positive + negative into one list of tensors
        # 4. Pad shorter sequences with 0s using nn.utils.rnn.pad_sequence(tensors, batch_first=True)
        unique_words = set()
        for sentence in positive:
            words = sentence.split()
            for word in words:
                unique_words.add(word)
        for sentence in negative:
            words = sentence.split()
            for word in words:
                unique_words.add(word)
        sorted_unique_words = sorted(unique_words)
        vocab = {}
        for idx,word in enumerate(sorted_unique_words,start=1):
            vocab[word]=idx
        positive_list = []
        for sentence in positive:
            words = sentence.split()
            ids = []
            for word in words:
                ids.append(vocab[word])
            tensor = torch.tensor(ids)
            positive_list.append(tensor)
        
        negative_list = []
        for sentence in negative:
            words = sentence.split()
            ids = []
            for word in words:
                ids.append(vocab[word])
            tensor = torch.tensor(ids)
            negative_list.append(tensor)
        all_tensors = positive_list+negative_list
        ans = nn.utils.rnn.pad_sequence(all_tensors,batch_first=True)
        return ans