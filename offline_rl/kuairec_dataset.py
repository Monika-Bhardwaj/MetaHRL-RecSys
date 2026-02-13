# offline_rl/kuairec_dataset.py

import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np


class KuaiRecOfflineDataset(Dataset):
    def __init__(self, csv_path):

        print("[Dataset] Loading CSV...")
        df = pd.read_csv(csv_path)

        print(f"[Dataset] Rows: {len(df)}")
        print(f"[Dataset] Columns: {list(df.columns)}")

        # ---------------------------------------------------
        # 🔥 Detect correct action column automatically
        # ---------------------------------------------------
        if "item_id" in df.columns:
            action_col = "item_id"
        elif "video_id" in df.columns:
            action_col = "video_id"
        else:
            raise ValueError("Could not find action column (item_id or video_id)")

        # ---------------------------------------------------
        # 🔥 Detect reward column automatically
        # ---------------------------------------------------
        if "reward" in df.columns:
            reward_col = "reward"
        elif "watch_ratio" in df.columns:
            reward_col = "watch_ratio"
        elif "watch_time" in df.columns:
            reward_col = "watch_time"
        else:
            raise ValueError("Could not find reward column")

        # ---------------------------------------------------
        # Remap actions to contiguous indices
        # ---------------------------------------------------
        df[action_col], action_mapping = pd.factorize(df[action_col])
        self.num_actions = len(action_mapping)

        # ---------------------------------------------------
        # State construction
        # For big_matrix: use user_id as state feature
        # ---------------------------------------------------
        if "user_id" in df.columns:
            df["user_id"], _ = pd.factorize(df["user_id"])
            state = df[["user_id"]].values.astype(np.float32)
        else:
            raise ValueError("Could not find user_id column")

        action = df[action_col].values.astype(np.int64)

        reward = df[reward_col].values.astype(np.float32)

        # Normalize reward
        reward = (reward - reward.mean()) / (reward.std() + 1e-6)

        # Next state (shifted)
        next_state = np.roll(state, -1, axis=0)

        done = np.zeros(len(df), dtype=np.float32)
        done[-1] = 1.0

        self.state = torch.tensor(state)
        self.action = torch.tensor(action)
        self.reward = torch.tensor(reward).unsqueeze(-1)
        self.next_state = torch.tensor(next_state)
        self.done = torch.tensor(done).unsqueeze(-1)

        print(f"[Dataset] Transitions: {len(self.state)}")
        print(f"[Dataset] Unique actions: {self.num_actions}")

    def __len__(self):
        return len(self.state)

    def __getitem__(self, idx):
        return {
            "state": self.state[idx],
            "action": self.action[idx],
            "reward": self.reward[idx],
            "next_state": self.next_state[idx],
            "done": self.done[idx],
        }