import sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from offline_rl.kuairec_dataset import KuaiRecOfflineDataset
from offline_rl.iql import QNetwork, ValueNetwork, PolicyNetwork
from offline_rl.fqe import FQENetwork, train_fqe

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

csv_path = ROOT / "data" / "raw" / "kuairec" / "data" / "big_matrix.csv"
ckpt_path = ROOT / "checkpoints" / "iql_kuairec.pt"


def main():
    # -------- Dataset --------
    dataset = KuaiRecOfflineDataset(
        csv_path=str(csv_path),
        history_len=20,
    )

    loader = DataLoader(
        dataset,
        batch_size=256,
        shuffle=True,
        num_workers=8,
        drop_last=True,
    )

    num_items = dataset.max_item_id + 1
    embed_dim = 64
    gru_hidden_dim = 128

    # -------- Load policy --------
    pi_net = PolicyNetwork(num_items, embed_dim, gru_hidden_dim).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    pi_net.load_state_dict(ckpt["pi"])
    pi_net.eval()

    # -------- FQE network --------
    fqe_net = FQENetwork(num_items, embed_dim, gru_hidden_dim).to(device)
    optimizer = torch.optim.Adam(fqe_net.parameters(), lr=3e-4)

    # -------- Train FQE --------
    train_fqe(
        fqe_net=fqe_net,
        policy_net=pi_net,
        dataloader=loader,
        optimizer=optimizer,
        device=device,
        steps=500,
    )

    # -------- Final metric --------
    fqe_net.eval()
    total_q = 0.0
    count = 0

    with torch.no_grad():
        for batch in loader:
            state = batch["state"].to(device)
            action = batch["action"].to(device)
            q = fqe_net(state, action)
            total_q += q.sum().item()
            count += q.size(0)

            if count > 100_000:
                break

    print("\n==============================")
    print(f"Estimated J(pi) = {total_q / count:.4f}")
    print("==============================\n")
    
    from evaluation.behavior_eval import evaluate_behavior_policy

    # --- Behavior baseline ---
    behavior_j = evaluate_behavior_policy(
        csv_path=str(csv_path),
        history_len=20,
        num_items=num_items,
        device=device,
    )

    print(f"Behavior policy J = {behavior_j:.4f}")
    print(f"IQL policy J = {total_q / count:.4f}")
    print(f"Lift = {(total_q / count - behavior_j) / abs(behavior_j) * 100:.2f}%")


if __name__ == "__main__":
    main()