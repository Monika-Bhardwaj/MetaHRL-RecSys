import sys
import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from offline_rl.kuairec_dataset import KuaiRecOfflineDataset
from offline_rl.iql import QNetwork, ValueNetwork, PolicyNetwork, expectile_loss
from utils.seed import set_global_seed


def sample_negatives(batch_size, num_items, k=10, device="cpu"):
    return torch.randint(
        low=0,
        high=num_items,
        size=(batch_size, k),
        device=device,
    )


def resolve_device(requested_device: str):
    """
    Safely resolve device without crashing.
    """
    if requested_device == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        else:
            print("⚠ CUDA requested but not available. Falling back to CPU.")
            return torch.device("cpu")
    return torch.device("cpu")


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--steps", type=int, default=500)
    args = parser.parse_args()

    # 1️⃣ Deterministic setup
    set_global_seed(args.seed)

    # 2️⃣ Safe device resolution
    device = resolve_device(args.device)

    print(f"[Seed] {args.seed}")
    print(f"[Device] {device}")

    csv_path = ROOT / "data" / "raw" / "kuairec" / "data" / "big_matrix.csv"

    dataset = KuaiRecOfflineDataset(
        csv_path=str(csv_path),
        history_len=20,
    )

    loader = DataLoader(
        dataset,
        batch_size=128,
        shuffle=False,      # deterministic
        num_workers=0,      # deterministic
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

    print("[Training] Starting IQL...")

    for step, batch in enumerate(loader):

        state = batch["state"].to(device)
        action = batch["action"].to(device)
        reward = batch["reward"].to(device)
        next_state = batch["next_state"].to(device)
        done = batch["done"].to(device)

        # ----- Q update -----
        q_sa = q_net(state, action)

        with torch.no_grad():
            v_next = v_net(next_state)
            q_target = reward + gamma * v_next * (1 - done)

        q_loss = F.mse_loss(q_sa, q_target)

        q_opt.zero_grad()
        q_loss.backward()
        torch.nn.utils.clip_grad_norm_(q_net.parameters(), 5.0)
        q_opt.step()

        # ----- V update -----
        v = v_net(state)
        v_loss = expectile_loss(q_sa.detach() - v, tau).mean()

        v_opt.zero_grad()
        v_loss.backward()
        torch.nn.utils.clip_grad_norm_(v_net.parameters(), 5.0)
        v_opt.step()

        # ----- Policy update -----
        if step % 2 == 0:

            with torch.no_grad():
                adv = q_sa.detach() - v.detach()
                adv = (adv - adv.mean()) / (adv.std() + 1e-6)
                weights = torch.exp(beta * adv).clamp(max=20.0)

                last_adv_mean = adv.mean().item()
                last_adv_std = adv.std().item()

            neg_actions = sample_negatives(
                state.size(0),
                num_items,
                K,
                device=device,
            )

            all_actions = torch.cat(
                [action.unsqueeze(1), neg_actions],
                dim=1,
            )

            scores = pi_net(state, all_actions)

            labels = torch.zeros(
                state.size(0),
                dtype=torch.long,
                device=device,
            )

            ce = F.cross_entropy(scores, labels, reduction="none")
            pi_loss = (weights * ce).mean()

            pi_opt.zero_grad()
            pi_loss.backward()
            torch.nn.utils.clip_grad_norm_(pi_net.parameters(), 5.0)
            pi_opt.step()

        else:
            pi_loss = torch.tensor(0.0, device=device)

        if step % 50 == 0:
            print(
                f"Step {step} | "
                f"Q {q_loss.item():.3f} | "
                f"V {v_loss.item():.3f} | "
                f"Pi {pi_loss.item():.3f} | "
                f"adv mean {last_adv_mean:.3f} | "
                f"adv std {last_adv_std:.3f}"
            )

        if step >= args.steps:
            break

    ckpt_dir = ROOT / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)

    torch.save(
        {
            "q": q_net.state_dict(),
            "v": v_net.state_dict(),
            "pi": pi_net.state_dict(),
            "seed": args.seed,
        },
        ckpt_dir / f"iql_kuairec_seed_{args.seed}.pt",
    )

    print("✅ Saved FINAL IQL checkpoint")


if __name__ == "__main__":
    main()