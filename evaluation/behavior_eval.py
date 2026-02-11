import torch
from torch.utils.data import DataLoader

from offline_rl.fqe import FQENetwork, train_fqe
from offline_rl.kuairec_dataset import KuaiRecOfflineDataset


def evaluate_behavior_policy(
    csv_path,
    history_len,
    num_items,
    embed_dim=64,
    gru_hidden_dim=128,
    device="cpu",
):
    """
    FQE evaluation of the logging (behavior) policy
    Uses logged actions as π(s')
    """

    dataset = KuaiRecOfflineDataset(csv_path=csv_path, history_len=history_len)
    loader = DataLoader(
        dataset,
        batch_size=256,
        shuffle=True,
        num_workers=8,
        drop_last=True,
    )

    fqe_net = FQENetwork(num_items, embed_dim, gru_hidden_dim).to(device)
    optimizer = torch.optim.Adam(fqe_net.parameters(), lr=3e-4)

    fqe_net.train()

    for step, batch in enumerate(loader):
        state = batch["state"].to(device)
        action = batch["action"].to(device)
        reward = batch["reward"].to(device)
        next_state = batch["next_state"].to(device)
        done = batch["done"].to(device)

        with torch.no_grad():
            # Behavior policy = logged action
            target_q = reward + 0.99 * fqe_net(
                next_state, action
            ) * (1 - done)

        q_pred = fqe_net(state, action)
        loss = torch.nn.functional.mse_loss(q_pred, target_q)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            print(f"[Behavior FQE] Step {step} | Loss {loss.item():.4f}")

        if step >= 500:
            break

    # Estimate J(pi_behavior)
    fqe_net.eval()
    total_q, count = 0.0, 0

    with torch.no_grad():
        for batch in loader:
            state = batch["state"].to(device)
            action = batch["action"].to(device)
            q = fqe_net(state, action)
            total_q += q.sum().item()
            count += q.size(0)
            if count > 100_000:
                break

    return total_q / count