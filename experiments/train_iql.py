import sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from offline_rl.kuairec_dataset import KuaiRecOfflineDataset
from offline_rl.iql import QNetwork, ValueNetwork, PolicyNetwork, expectile_loss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def sample_negatives(batch_size, num_items, k=10, device="cpu"):
    return torch.randint(0, num_items, (batch_size, k), device=device)

def main():
    dataset = KuaiRecOfflineDataset(
        csv_path="data/raw/kuairec/data/big_matrix.csv",
        history_len=5,
    )

    loader = DataLoader(
        dataset,
        batch_size=512,                  # bigger batch for speed
        shuffle=True,
        num_workers=8,                   # use all CPU cores available
        pin_memory=True,
        drop_last=True,
    )

    num_items = dataset.max_item_id + 1
    embed_dim = 64
    gru_hidden_dim = 128

    q_net = QNetwork(num_items, embed_dim, gru_hidden_dim).to(device)
    v_net = ValueNetwork(num_items, embed_dim, gru_hidden_dim).to(device)
    pi_net = PolicyNetwork(num_items, embed_dim, gru_hidden_dim).to(device)

    q_opt = torch.optim.Adam(q_net.parameters(), lr=3e-4)
    v_opt = torch.optim.Adam(v_net.parameters(), lr=3e-4)
    pi_opt = torch.optim.Adam(pi_net.parameters(), lr=3e-4)

    gamma = 0.99
    tau = 0.7
    beta = 3.0
    K = 10

    for step, batch in enumerate(loader):
        state = batch["state"].to(device, non_blocking=True)
        action = batch["action"].to(device, non_blocking=True)
        reward = batch["reward"].to(device, non_blocking=True)
        next_state = batch["next_state"].to(device, non_blocking=True)
        done = batch["done"].to(device, non_blocking=True)

        # ----- Q update -----
        q_sa = q_net(state, action)
        with torch.no_grad():
            q_target = reward + gamma * v_net(next_state) * (1 - done)
        q_loss = F.mse_loss(q_sa, q_target)
        q_opt.zero_grad()
        q_loss.backward()
        q_opt.step()

        # ----- V update -----
        v = v_net(state)
        v_loss = expectile_loss(q_sa.detach() - v, tau).mean()
        v_opt.zero_grad()
        v_loss.backward()
        v_opt.step()

        # ----- Policy update -----
        with torch.no_grad():
            adv = q_sa.detach() - v
            weights = torch.exp(beta * adv).clamp(max=20.0)

        neg_actions = sample_negatives(state.size(0), num_items, K, device=device)
        all_actions = torch.cat([action.unsqueeze(1), neg_actions], dim=1)
        scores = pi_net(state, all_actions)
        labels = torch.zeros(state.size(0), dtype=torch.long, device=device)
        pi_loss = (weights * F.cross_entropy(scores, labels, reduction="none")).mean()
        pi_opt.zero_grad()
        pi_loss.backward()
        pi_opt.step()

        if step % 50 == 0:
            print(f"Step {step} | Q {q_loss.item():.3f} | V {v_loss.item():.3f} | Pi {pi_loss.item():.3f}")

        if step == 500:  # safety stop
            break

if __name__ == "__main__":
    main()