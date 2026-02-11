import sys
import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader

# ---- Fix import path ----
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.seed import set_global_seed
from offline_rl.fqe import FQETrainer
from offline_rl.kuairec_dataset import KuaiRecOfflineDataset


def resolve_device(requested_device: str):
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

    device = resolve_device(args.device)

    print(f"[Seed] {args.seed}")
    print(f"[Device] {device}")

    csv_path = ROOT / "data" / "raw" / "kuairec" / "data" / "big_matrix.csv"

    dataset = KuaiRecOfflineDataset(
        csv_path=str(csv_path),
        history_len=20,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=512,
        shuffle=False,     # IMPORTANT for determinism
        num_workers=0,
        drop_last=False,
    )

    num_items = dataset.max_item_id + 1

    # Infer dimensions from first batch
    sample = next(iter(dataloader))
    state_dim = sample["state"].shape[-1]
    action_dim = 1  # because action is item id (scalar embedding index)

    trainer = FQETrainer(
        state_dim=state_dim,
        action_dim=action_dim,
        device=device,
        target_update_freq=100,
        clip_target=True,
    )

    print("[FQE] Training started")

    for step in range(args.steps):
        for batch in dataloader:

            states = batch["state"]
            actions = batch["action"].unsqueeze(-1).float()
            rewards = batch["reward"].unsqueeze(-1)
            next_states = batch["next_state"]
            dones = batch["done"].unsqueeze(-1)

            # IMPORTANT:
            # For FQE we need next_action from same logged data
            next_actions = batch["action"].unsqueeze(-1).float()

            fqe_batch = (
                states,
                actions,
                rewards,
                next_states,
                next_actions,
                dones,
            )

            loss = trainer.train_step(fqe_batch, step)

        if step % 50 == 0:
            print(f"[FQE] Step {step} | Loss {loss:.4f}")

    J = trainer.evaluate_policy(
        [
            (
                batch["state"],
                batch["action"].unsqueeze(-1).float(),
                batch["reward"],
                batch["next_state"],
                batch["action"].unsqueeze(-1).float(),
                batch["done"],
            )
            for batch in dataloader
        ]
    )

    print("\n==============================")
    print(f"FQE Estimated J = {J:.4f}")
    print("==============================\n")


if __name__ == "__main__":
    main()