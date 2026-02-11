# offline_rl/fqe.py

import copy
import torch
import torch.nn as nn
import torch.optim as optim


class FQENetwork(nn.Module):
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


class FQETrainer:
    def __init__(
        self,
        state_dim,
        action_dim,
        gamma=0.99,
        lr=3e-4,
        target_update_freq=50,
        clip_target=True,
        device="cpu",
    ):
        self.device = device
        self.gamma = gamma
        self.clip_target = clip_target
        self.target_update_freq = target_update_freq

        self.q_net = FQENetwork(state_dim, action_dim).to(device)
        self.target_q_net = copy.deepcopy(self.q_net).to(device)
        self.target_q_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

    def train_step(self, batch, step):

        states, actions, rewards, next_states, next_actions, dones = batch

        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        next_actions = next_actions.to(self.device)
        dones = dones.to(self.device)

        with torch.no_grad():
            target_q = self.target_q_net(next_states, next_actions)
            td_target = rewards + self.gamma * (1 - dones) * target_q

            if self.clip_target:
                td_target = torch.clamp(td_target, -10.0, 10.0)

        current_q = self.q_net(states, actions)

        loss = self.loss_fn(current_q, td_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Deterministic hard target update
        if step % self.target_update_freq == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())

        return loss.item()

    def evaluate_policy(self, dataloader):
        """
        Deterministic evaluation.
        """
        self.q_net.eval()

        total_q = 0.0
        total_count = 0

        with torch.no_grad():
            for batch in dataloader:
                states, actions, _, _, _, _ = batch
                states = states.to(self.device)
                actions = actions.to(self.device)

                q_vals = self.q_net(states, actions)
                total_q += q_vals.sum().item()
                total_count += q_vals.shape[0]

        return total_q / total_count