# offline_rl/fqe_double.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    def __init__(self, state_dim, num_actions, action_embed_dim=64, hidden_dim=256):
        super().__init__()

        # 🔥 embedding for discrete item IDs
        self.action_embedding = nn.Embedding(num_actions, action_embed_dim)

        self.net = nn.Sequential(
            nn.Linear(state_dim + action_embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state, action):
        # action expected shape: (batch,)
        action_emb = self.action_embedding(action.long())

        x = torch.cat([state, action_emb], dim=-1)
        return self.net(x)


class DoubleFQE:
    def __init__(
        self,
        state_dim,
        num_actions,
        device,
        gamma=0.99,
        lr=1e-4,
        tau=0.005,
    ):
        self.device = device
        self.gamma = gamma
        self.tau = tau

        self.q1 = QNetwork(state_dim, num_actions).to(device)
        self.q2 = QNetwork(state_dim, num_actions).to(device)

        self.q1_target = QNetwork(state_dim, num_actions).to(device)
        self.q2_target = QNetwork(state_dim, num_actions).to(device)

        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        self.optimizer = torch.optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()),
            lr=lr,
        )

    def train_step(self, batch):
        state = batch["state"].to(self.device)
        action = batch["action"].to(self.device).squeeze()
        reward = batch["reward"].to(self.device)
        next_state = batch["next_state"].to(self.device)
        done = batch["done"].to(self.device)

        with torch.no_grad():
            next_action = action  # behavior policy

            q1_next = self.q1_target(next_state, next_action)
            q2_next = self.q2_target(next_state, next_action)

            q_next = torch.min(q1_next, q2_next)

            target = reward + self.gamma * (1 - done) * q_next

        q1 = self.q1(state, action)
        q2 = self.q2(state, action)

        loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)

        self.optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.q1.parameters(), 5.0)
        torch.nn.utils.clip_grad_norm_(self.q2.parameters(), 5.0)

        self.optimizer.step()

        self._soft_update(self.q1, self.q1_target)
        self._soft_update(self.q2, self.q2_target)

        return loss.item()

    def _soft_update(self, net, target_net):
        for param, target_param in zip(net.parameters(), target_net.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )

    @torch.no_grad()
    def evaluate(self, dataloader, max_batches=500):
        values = []

        for i, batch in enumerate(dataloader):
            if i >= max_batches:
                break

            state = batch["state"].to(self.device)
            action = batch["action"].to(self.device).squeeze()

            q = torch.min(
                self.q1(state, action),
                self.q2(state, action),
            )

            values.append(q.mean().item())

        return sum(values) / len(values)