# DEPRECATED: kept for reference only

# experiments/train_cql.py
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import torch
from torch.utils.data import DataLoader

from offline_rl.kuairec_dataset import KuaiRecOfflineDataset
from offline_rl.conservative_q import ConservativeQNetwork, cql_loss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    # 1. Dataset
    dataset = KuaiRecOfflineDataset(
        csv_path="data/raw/kuairec/data/big_matrix.csv",
        history_len=5,
    )

    loader = DataLoader(
        dataset,
        batch_size=256,
        shuffle=True,
        drop_last=True,
        num_workers=4,
        pin_memory=True,
    )

    # 2. Model
    num_items = 50000   # set conservatively high
    embed_dim = 64

    q_net = ConservativeQNetwork(
        num_items=num_items,
        embed_dim=embed_dim,
    ).to(device)

    optimizer = torch.optim.Adam(q_net.parameters(), lr=3e-4)

    # 3. Training loop
    for step, batch in enumerate(loader):
        state = batch["state"].to(device)          # [B, H]
        action = batch["action"].to(device)        # [B]
        reward = batch["reward"].to(device)        # [B]
        next_state = batch["next_state"].to(device)
        done = batch["done"].to(device)

        loss = cql_loss(
            q_net,
            state,
            action,
            reward,
            next_state,
            done,
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 100 == 0:
            print(f"Step {step} | CQL Loss: {loss.item():.4f}")

        if step == 1000:
            break  # safety stop for first run


if __name__ == "__main__":
    main()