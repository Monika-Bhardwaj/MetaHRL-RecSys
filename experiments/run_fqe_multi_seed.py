# experiments/run_fqe_multi_seed.py

import os
import sys
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader, Subset

# Add project root to path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT_DIR)

from offline_rl.fqe_double import DoubleFQE
from offline_rl.kuairec_dataset import KuaiRecOfflineDataset


def set_global_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_path(path):
    """
    Make path relative to project root.
    """
    if os.path.isabs(path):
        return path
    return os.path.join(ROOT_DIR, path)


def run_single_seed(seed, device, steps, subsample, csv_path):

    print(f"\nRunning seed {seed}")
    set_global_seed(seed)

    csv_path = resolve_path(csv_path)

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found at: {csv_path}")

    dataset = KuaiRecOfflineDataset(csv_path)

    print(f"[Dataset] Total transitions: {len(dataset)}")

    # deterministic subsampling
    rng = np.random.RandomState(seed)
    indices = rng.choice(len(dataset), size=subsample, replace=False)
    dataset = Subset(dataset, indices)

    train_loader = DataLoader(
        dataset,
        batch_size=1024,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )

    eval_loader = DataLoader(
        dataset,
        batch_size=2048,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    sample = dataset[0]
    state_dim = sample["state"].shape[-1]
    num_actions = dataset.dataset.num_actions

    fqe = DoubleFQE(
        state_dim=state_dim,
        num_actions=num_actions,
        device=device,
    )

    loader_iter = iter(train_loader)

    for step in range(steps):
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(train_loader)
            batch = next(loader_iter)

        fqe.train_step(batch)

        if step % 2000 == 0:
            print(f"Step {step}")

    J = fqe.evaluate(eval_loader)
    print(f"Seed {seed} J = {J:.4f}")

    return J


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv_path",
        type=str,
        default="data/raw/kuairec/data/small_matrix.csv",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--subsample", type=int, default=1000000)
    args = parser.parse_args()

    device = torch.device(args.device)

    seeds = [0, 1, 2]
    results = []

    for seed in seeds:
        J = run_single_seed(
            seed,
            device,
            args.steps,
            args.subsample,
            args.csv_path,
        )
        results.append(J)

    mean = np.mean(results)
    std = np.std(results)

    print("\n==============================")
    print(f"Behavior J = {mean:.4f} ± {std:.4f}")
    print("==============================")


if __name__ == "__main__":
    main()