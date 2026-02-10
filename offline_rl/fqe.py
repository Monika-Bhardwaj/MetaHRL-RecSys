import torch
import torch.nn as nn
import torch.nn.functional as F

from offline_rl.iql import StateEncoder, ActionEncoder


class FQENetwork(nn.Module):
    """
    Q(s,a) evaluator for a fixed policy π
    Architecture mirrors QNetwork but trained separately
    """

    def __init__(self, num_items, embed_dim, gru_hidden_dim):
        super().__init__()
        self.state_enc = StateEncoder(
            num_items=num_items,
            embed_dim=embed_dim,
            gru_hidden_dim=gru_hidden_dim,
            proj_dim=embed_dim,
        )
        self.action_enc = ActionEncoder(num_items, embed_dim)

        self.fc = nn.Sequential(
            nn.Linear(embed_dim * 2, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, state, action):
        s = self.state_enc(state)     # [B, D]
        a = self.action_enc(action)   # [B, D]
        return self.fc(torch.cat([s, a], dim=1)).squeeze(-1)


def train_fqe(
    fqe_net,
    policy_net,
    dataloader,
    optimizer,
    gamma=0.99,
    device="cpu",
    steps=500,
):
    """
    Train FQE with Bellman evaluation updates
    """

    fqe_net.train()
    policy_net.eval()

    for step, batch in enumerate(dataloader):
        state = batch["state"].to(device)
        action = batch["action"].to(device)
        reward = batch["reward"].to(device)
        next_state = batch["next_state"].to(device)
        done = batch["done"].to(device)

        # ----- Bellman target -----
        with torch.no_grad():
            # π(s') → greedy action from policy network
            # sample top action by dot product
            num_items = policy_net.action_enc.item_emb.num_embeddings
            candidates = torch.randint(
                0, num_items, (state.size(0), 20), device=device
            )
            scores = policy_net(next_state, candidates)
            next_action = candidates.gather(
                1, scores.argmax(dim=1, keepdim=True)
            ).squeeze(1)

            target_q = reward + gamma * fqe_net(next_state, next_action) * (1 - done)

        q_pred = fqe_net(state, action)
        loss = F.mse_loss(q_pred, target_q)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            print(f"[FQE] Step {step} | Loss {loss.item():.4f}")

        if step == steps:
            break