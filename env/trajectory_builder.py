# DEPRECATED: kept for reference only

# env/trajectory_builder.py

from env.delayed_reward import compute_reward

def build_rl_transitions(trajectory, history_len=5):
    interactions = trajectory["interactions"]
    transitions = []

    for t in range(len(interactions) - 1):
        start = max(0, t - history_len + 1)

        def encode_state(df_slice):
            return [
                {
                    "video_id": row["video_id"],
                    "watch_ratio": row["watch_ratio"],
                    "play_duration": row["play_duration"],
                    "video_duration": row["video_duration"],
                }
                for _, row in df_slice.iterrows()
            ]

        state = encode_state(interactions.iloc[start:t+1])
        next_state = encode_state(interactions.iloc[start:t+2])

        transitions.append({
            "state": state,
            "action": interactions.iloc[t]["video_id"],
            "reward": compute_reward(interactions.iloc[t]),
            "next_state": next_state,
            "done": False
        })

    return transitions