# DEPRECATED: kept for reference only

# data/processed/build_kuairec_dataset.py

import sys
from pathlib import Path

# ------------------------------------------------------------------
# Make project root importable
# ------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(BASE_DIR))

import pickle
import pandas as pd
from env.trajectory_builder import build_rl_transitions

RAW_DIR = BASE_DIR / "data/raw/kuairec/data"
OUT_PATH = BASE_DIR / "data/processed/kuairec_rl_dataset.pkl"


def load_raw_tables():
    return pd.read_csv(RAW_DIR / "big_matrix.csv")


def sort_interactions(df):
    return df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)


def build_user_trajectories(df, min_length=5):
    for user_id, user_df in df.groupby("user_id"):
        if len(user_df) < min_length:
            continue
        yield {
            "user_id": user_id,
            "interactions": user_df.reset_index(drop=True)
        }


def main():
    print("Loading KuaiRec raw data...")
    big_matrix = load_raw_tables()

    print("big_matrix columns:")
    print(big_matrix.columns.tolist())

    print("Sorting interactions by time...")
    big_matrix = sort_interactions(big_matrix)

    print("Streaming RL transitions to disk...")

    total_transitions = 0
    with open(OUT_PATH, "wb") as f:
        for idx, traj in enumerate(build_user_trajectories(big_matrix)):
            transitions = build_rl_transitions(traj)
            pickle.dump(transitions, f, protocol=pickle.HIGHEST_PROTOCOL)

            total_transitions += len(transitions)

            if idx % 500 == 0:
                print(f"  Users processed: {idx}, transitions: {total_transitions}")

    print(f"Done. Saved {total_transitions} transitions to:")
    print(OUT_PATH)


if __name__ == "__main__":
    main()