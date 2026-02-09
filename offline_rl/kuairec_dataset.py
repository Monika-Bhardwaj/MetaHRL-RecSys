import pandas as pd
import torch
from torch.utils.data import Dataset

class KuaiRecOfflineDataset(Dataset):
    """
    Highly optimized lazy offline RL dataset for KuaiRec.
    Precomputes embeddings for faster training.
    """

    def __init__(self, csv_path, history_len=20, reward_col="watch_ratio", max_rows=None):
        self.history_len = history_len
        self.reward_col = reward_col

        df = pd.read_csv(csv_path)
        if max_rows is not None:
            df = df.iloc[:max_rows]

        df = df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)

        # store raw columns
        self.video_ids = torch.tensor(df["video_id"].values, dtype=torch.long)
        self.rewards = torch.tensor(df[reward_col].values, dtype=torch.float)
        self.user_ids = df["user_id"].values

        # precompute user start and position
        self.user_start = torch.zeros(len(df), dtype=torch.long)
        self.pos_in_user = torch.zeros(len(df), dtype=torch.long)

        last_user = None
        start = 0
        pos = 0
        for i, uid in enumerate(self.user_ids):
            if uid != last_user:
                start = i
                pos = 0
                last_user = uid
            self.user_start[i] = start
            self.pos_in_user[i] = pos
            pos += 1

        # valid transitions
        self.valid_indices = [i for i in range(len(df) - 1) if self.user_ids[i] == self.user_ids[i + 1]]

        # max video id
        self.max_item_id = int(self.video_ids.max().item())

        print(f"[Dataset] Rows: {len(df)}")
        print(f"[Dataset] Transitions: {len(self.valid_indices)}")

    def __len__(self):
        return len(self.valid_indices)

    def _get_state(self, idx):
        start = self.user_start[idx]
        hist_start = max(start, idx - self.history_len + 1)
        hist = self.video_ids[hist_start: idx + 1]
        state = torch.zeros(self.history_len, dtype=torch.long)
        state[-len(hist):] = hist
        return state

    def __getitem__(self, i):
        idx = self.valid_indices[i]
        return {
            "state": self._get_state(idx),
            "action": self.video_ids[idx],
            "reward": self.rewards[idx],
            "next_state": self._get_state(idx + 1),
            "done": torch.tensor(0.0),
        }