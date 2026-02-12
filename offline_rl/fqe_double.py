# offline_rl/fqe_double.py

import copy
import torch
import torch.nn as nn
import torch.optim as optim


class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.net(x)


class DoubleFQE:
    def __init__(
        self,
        state_dim,
        action_dim,
        gamma=0.99,
        lr=1e-4,
        target_update_freq=500,
        device="cpu",
    ):
        self.device = device
        self.gamma = gamma
        self.target_update_freq = target_update_freq

        self.q1 = QNetwork(state_dim, action_dim).to(device)
        self.q2 = QNetwork(state_dim, action_dim).to(device)

        self.q1_target = copy.deepcopy(self.q1)
        self.q2_target = copy.deepcopy(self.q2)

        self.opt = optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()),
            lr=lr,
        )

        self.loss_fn = nn.MSELoss()

    def train_step(self, batch, step):

        states, actions, rewards, next_states, next_actions, dones = batch

        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-6)
        next_states = next_states.to(self.device)
        next_actions = next_actions.to(self.device)
        dones = dones.to(self.device)

        with torch.no_grad():
            q1_next = self.q1_target(next_states, next_actions)
            q2_next = self.q2_target(next_states, next_actions)

            q_next = torch.min(q1_next, q2_next)
            target = rewards + self.gamma * (1 - dones) * q_next


        q1 = self.q1(states, actions)
        q2 = self.q2(states, actions)

        loss = self.loss_fn(q1, target) + self.loss_fn(q2, target)

        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.q1.parameters()) + list(self.q2.parameters()), 5.0
        )
        self.opt.step()

        tau = 0.005

        for param, target_param in zip(self.q1.parameters(), self.q1_target.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        for param, target_param in zip(self.q2.parameters(), self.q2_target.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        return loss.item()

    def evaluate(self, dataloader):

        self.q1.eval()

        total = 0.0
        count = 0

        with torch.no_grad():
            for batch in dataloader:
                states, actions, _, _, _, _ = batch
                states = states.to(self.device)
                actions = actions.to(self.device)

                q = self.q1(states, actions)
                total += q.sum().item()
                count += q.shape[0]

        return total / count