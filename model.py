import torch
import torch.nn as nn


class MFModel(nn.Module):
    def __init__(self, n_users, n_items, n_factors=64):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, n_factors)
        self.item_emb = nn.Embedding(n_items, n_factors)
        self.user_bias = nn.Embedding(n_users, 1)
        self.item_bias = nn.Embedding(n_items, 1)

        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)

    def forward(self, user_idxs, item_idxs):
        u = self.user_emb(user_idxs)
        v = self.item_emb(item_idxs)
        dot = (u * v).sum(dim=1, keepdim=True)
        bias = self.user_bias(user_idxs) + self.item_bias(item_idxs)
        return (dot + bias).squeeze(1)
