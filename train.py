import torch
import torch.nn as nn
import torch.nn.functional as F

# The GPT model is provided for you. It returns raw logits (not probabilities).
# You only need to implement the training loop below.

class Solution:
    def train(self, model: nn.Module, data: torch.Tensor, epochs: int, context_length: int, batch_size: int, lr: float) -> float:
        # Train the GPT model using AdamW and cross_entropy loss.
        # For each epoch: seed with torch.manual_seed(epoch),
        # sample batches from data, run forward/backward, update weights.
        # Return the final loss rounded to 4 decimals.
        optimizer = torch.optim.AdamW(model.parameters(),lr=lr)
        for epoch in range(epochs):
            torch.manual_seed(epoch)
            starts = torch.randint(
                0,
                len(data)-context_length,
                (batch_size,)
            )
            X = torch.empty(
                (batch_size,context_length),
                dtype = data.dtype
            )
            Y = torch.empty(
                (batch_size,context_length),
                dtype = data.dtype
            )

            for i,start in enumerate(starts):
                start = start.item()
                X[i] = data[start:start+context_length]
                Y[i] = data[start+1:start+1+context_length]
            
            logits = model(X)
            
            vocab_size = logits.shape[-1]

            logits = logits.view(-1,vocab_size)
            targets = Y.view(-1)
            loss = F.cross_entropy(logits,targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(f"Epoch: {i}, Loss: {loss.item():.4f}")
        return round(loss.item(),4)

