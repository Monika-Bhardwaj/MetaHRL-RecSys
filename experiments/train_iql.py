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

csv_path = ROOT / "data" / "raw" / "kuairec" / "data" / "big_matrix.csv"


def sample_negatives(batch_size, num_items, k=10, device="cpu"):
    return torch.randint(0, num_items, (batch_size, k), device=device)


def main():
    dataset = KuaiRecOfflineDataset(
        csv_path=str(csv_path),
        history_len=20,
    )

    loader = DataLoader(
        dataset,
        batch_size=128,
        shuffle=True,
        num_workers=8,
        pin_memory=torch.cuda.is_available(),
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
    beta = 0.5
    K = 10

    last_adv_mean = 0.0
    last_adv_std = 0.0

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

        # ----- Policy update (delayed) -----
        if step % 2 == 0:
            with torch.no_grad():
                adv = q_sa.detach() - v
                adv = (adv - adv.mean()) / (adv.std() + 1e-6)
                weights = torch.exp(beta * adv).clamp(max=20.0)

                last_adv_mean = adv.mean().item()
                last_adv_std = adv.std().item()

            neg_actions = sample_negatives(state.size(0), num_items, K, device=device)
            all_actions = torch.cat([action.unsqueeze(1), neg_actions], dim=1)
            scores = pi_net(state, all_actions)

            labels = torch.zeros(state.size(0), dtype=torch.long, device=device)
            pi_loss = (weights * F.cross_entropy(scores, labels, reduction="none")).mean()

            pi_opt.zero_grad()
            pi_loss.backward()
            torch.nn.utils.clip_grad_norm_(pi_net.parameters(), 5.0)
            pi_opt.step()
        else:
            pi_loss = torch.tensor(0.0)

        # ----- Logging -----
        if step % 50 == 0:
            print(
                f"Step {step} | "
                f"Q {q_loss.item():.3f} | "
                f"V {v_loss.item():.3f} | "
                f"Pi {pi_loss.item():.3f} | "
                f"adv mean {last_adv_mean:.3f} | "
                f"adv std {last_adv_std:.3f}"
            )

        if step == 500:
            break

    # ----- Save FINAL checkpoint -----
    ckpt_dir = ROOT / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)

    torch.save(
        {
            "q": q_net.state_dict(),
            "v": v_net.state_dict(),
            "pi": pi_net.state_dict(),
        },
        ckpt_dir / "iql_kuairec.pt",
    )

    print("✅ Saved FINAL IQL checkpoint")


if __name__ == "__main__":
    main()