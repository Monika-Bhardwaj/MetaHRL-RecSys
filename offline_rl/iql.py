import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------
# GRU-based State Encoder
# ---------------------------
class StateEncoder(nn.Module):
    """
    Encode sequence of item IDs via embedding + GRU
    """
    def __init__(self, num_items, embed_dim, gru_hidden_dim=None, num_layers=1, proj_dim=None):
        super().__init__()
        self.item_emb = nn.Embedding(num_items, embed_dim)
        if gru_hidden_dim is None:
            gru_hidden_dim = embed_dim
        self.gru = nn.GRU(embed_dim, gru_hidden_dim, num_layers=num_layers, batch_first=True)
        self.proj = nn.Linear(gru_hidden_dim, proj_dim) if proj_dim is not None else None

    def forward(self, state):
        # state: [B, H]
        emb = self.item_emb(state)           # [B, H, D]
        _, h = self.gru(emb)                 # h: [num_layers, B, HIDDEN]
        out = h[-1]                          # [B, HIDDEN]
        if self.proj is not None:
            out = self.proj(out)            # project to desired dim
        return out                            # [B, proj_dim or HIDDEN]

# ---------------------------
# Action Encoder
# ---------------------------
class ActionEncoder(nn.Module):
    def __init__(self, num_items, embed_dim):
        super().__init__()
        self.item_emb = nn.Embedding(num_items, embed_dim)

    def forward(self, action):
        # action: [B] or [B, K]
        return self.item_emb(action)          # [B, D] or [B, K, D]

# ---------------------------
# Q Network
# ---------------------------
class QNetwork(nn.Module):
    def __init__(self, num_items, embed_dim, gru_hidden_dim=None):
        super().__init__()
        self.state_enc = StateEncoder(num_items, embed_dim, gru_hidden_dim, proj_dim=embed_dim)
        self.action_enc = ActionEncoder(num_items, embed_dim)
        self.fc = nn.Sequential(
            nn.Linear(embed_dim * 2, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, state, action):
        s = self.state_enc(state)            # [B, D]
        a = self.action_enc(action)          # [B, D]
        return self.fc(torch.cat([s, a], dim=1)).squeeze(-1)

# ---------------------------
# Value Network
# ---------------------------
class ValueNetwork(nn.Module):
    def __init__(self, num_items, embed_dim, gru_hidden_dim=None):
        super().__init__()
        self.state_enc = StateEncoder(num_items, embed_dim, gru_hidden_dim, proj_dim=embed_dim)
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, state):
        s = self.state_enc(state)
        return self.fc(s).squeeze(-1)

# ---------------------------
# Policy Network
# ---------------------------
class PolicyNetwork(nn.Module):
    def __init__(self, num_items, embed_dim, gru_hidden_dim=None):
        super().__init__()
        self.state_enc = StateEncoder(num_items, embed_dim, gru_hidden_dim, proj_dim=embed_dim)
        self.action_enc = ActionEncoder(num_items, embed_dim)

    def forward(self, state, actions):
        # state: [B,H], actions: [B,K]
        s = self.state_enc(state).unsqueeze(1)  # [B,1,D]
        a = self.action_enc(actions)            # [B,K,D]
        return (s * a).sum(dim=-1)              # [B,K]

# ---------------------------
# Expectile loss
# ---------------------------
def expectile_loss(diff, tau):
    w = torch.where(diff > 0, tau, 1 - tau)
    return w * diff.pow(2)