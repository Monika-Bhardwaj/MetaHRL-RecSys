# experiments/run_fqe_multi_seed.py

import sys
import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import numpy as np
import itertools

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.seed import set_global_seed
from offline_rl.fqe_double import DoubleFQE
from offline_rl.kuairec_dataset import KuaiRecOfflineDataset


def resolve_device(requested):
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def infinite_loader(loader):
    while True:
        for batch in loader:
            yield batch


def run_single_seed(seed, device, steps, subsample):

    set_global_seed(seed)

    csv_path = ROOT / "data" / "raw" / "kuairec" / "data" / "big_matrix.csv"

    dataset = KuaiRecOfflineDataset(str(csv_path), history_len=20)

    from torch.utils.data import Subset
    import numpy as np

    if subsample is not None and subsample < len(dataset):
        indices = np.random.choice(len(dataset), subsample, replace=False)
        dataset = Subset(dataset, indices)

    dataloader = DataLoader(
        dataset,
        batch_size=512,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )

    loader = infinite_loader(dataloader)

    sample = next(loader)
    state_dim = sample["state"].shape[-1]
    action_dim = 1

    fqe = DoubleFQE(state_dim, action_dim, device=device)

    for step in range(steps):
        batch = next(loader)

        states = batch["state"]
        actions = batch["action"].unsqueeze(-1).float()
        rewards = batch["reward"].unsqueeze(-1)
        next_states = batch["next_state"]
        next_actions = batch["action"].unsqueeze(-1).float()
        dones = batch["done"].unsqueeze(-1)

        fqe_batch = (
            states,
            actions,
            rewards,
            next_states,
            next_actions,
            dones,
        )

        fqe.train_step(fqe_batch, step)

        if step % 2000 == 0:
            print(f"Step {step}")

    # Fast evaluation
    fqe.q1.eval()
    total = 0.0
    count = 0

    with torch.no_grad():
        for batch in itertools.islice(dataloader, 200):  # evaluate on 200 batches only
            states = batch["state"].to(device)
            actions = batch["action"].unsqueeze(-1).float().to(device)
            q = fqe.q1(states, actions)
            total += q.sum().item()
            count += q.shape[0]

    return total / count


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[0,1,2])
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--subsample", type=int, default=2000000)
    args = parser.parse_args()

    device = resolve_device(args.device)

    results = []

    for seed in args.seeds:
        print(f"\nRunning seed {seed}")
        J = run_single_seed(seed, device, args.steps, args.subsample)
        print(f"Seed {seed} J = {J:.4f}")
        results.append(J)

    mean = np.mean(results)
    std = np.std(results)

    print("\n==============================")
    print(f"Behavior J = {mean:.4f} ± {std:.4f}")
    print("==============================\n")


if __name__ == "__main__":
    main()