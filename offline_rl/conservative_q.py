# DEPRECATED: kept for reference only

# offline_rl/conservative_q.py

import torch
import torch.nn as nn


class ConservativeQNetwork(nn.Module):
    def __init__(self, num_items, embed_dim):
        super().__init__()
        self.embedding = nn.Embedding(num_items, embed_dim)
        self.q = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_items)
        )

    def forward(self, state):
        emb = self.embedding(state)      # [B, H, D]
        pooled = emb.mean(dim=1)         # [B, D]
        return self.q(pooled)            # [B, num_items]


def cql_loss(q_net, state, action, reward, next_state, done, gamma=0.99, alpha=1.0):
    q_values = q_net(state)
    q_sa = q_values.gather(1, action.unsqueeze(1)).squeeze(1)

    with torch.no_grad():
        next_q = q_net(next_state).max(dim=1)[0]
        target = reward + gamma * next_q * (1 - done)

    td_loss = torch.mean((q_sa - target) ** 2)

    conservative_penalty = (
        torch.logsumexp(q_values, dim=1).mean() - q_sa.mean()
    )

    return td_loss + alpha * conservative_penalty